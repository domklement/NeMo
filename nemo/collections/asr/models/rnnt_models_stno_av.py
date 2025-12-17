# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from math import ceil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToCharDALIDataset, DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.losses.rnnt import RNNTLoss, resolve_rnnt_default_loss_name
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.modules.rnnt import RNNTDecoderJoint
from nemo.collections.asr.modules.avhubert import AVHubertAVSR, make_non_pad_mask
from nemo.collections.asr.data.av_data_utils import AudioTransform, VideoTransform
from nemo.collections.asr.parts.mixins import (
    ASRModuleMixin,
    ASRTranscriptionMixin,
    TranscribeConfig,
    TranscriptionReturnType,
)
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecoding, RNNTDecodingConfig
from nemo.collections.asr.parts.utils.asr_batching import get_semi_sorted_batch_sampler
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.parts.preprocessing.parsers import make_parser
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, MaskType, NeuralType, SpectrogramType
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


class EncDecRNNTModelSTNOAV(ASRModel, ASRModuleMixin, ExportableEncDecModel, ASRTranscriptionMixin):
    """Base class for encoder decoder RNNT-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        self.audio_transform = AudioTransform(subset="test")

        self.train_video_transform = VideoTransform(subset="train")
        self.test_video_transform = VideoTransform(subset="test")

        # VISUAL EMBEDDING EXTRACTION CONFIGS
        self.extract_features_on_the_fly = cfg.get("extract_features_on_the_fly", False)
        self.visual_encoder_type = cfg.get("visual_encoder_type", None)
        self.visual_encoder_ckpt_path = cfg.get("visual_encoder_ckpt_path", None)
        self.freeze_visual_encoder = cfg.get("freeze_visual_encoder", False)
        self.replace_zero_video_frames_with_zero_embeds = cfg.get("replace_zero_video_frames_with_zero_embeds", False)
        if self.extract_features_on_the_fly and (self.visual_encoder_type is None or self.visual_encoder_ckpt_path is None):
            raise ValueError(
                "When `extract_features_on_the_fly` is set to True, "
                "`visual_encoder_type` and `visual_encoder_ckpt_path` must be provided."
            )

        if self.extract_features_on_the_fly:
            if hasattr(cfg.train_ds, 'return_visual_features'):
                cfg.train_ds.return_visual_features = False
            if hasattr(cfg.validation_ds, 'return_visual_features'):
                cfg.validation_ds.return_visual_features = False
            if hasattr(cfg.train_ds, 'return_video'):
                cfg.train_ds.return_video = True
            if hasattr(cfg.validation_ds, 'return_video'):
                cfg.validation_ds.return_video = True

        super().__init__(cfg=cfg, trainer=trainer)

        self.use_audio_encoder = cfg.get("use_audio_encoder", True)
        self.freeze_rnnt = cfg.get("freeze_rnnt", False)

        # Initialize components
        self.preprocessor = EncDecRNNTModelSTNOAV.from_config_dict(self.cfg.preprocessor)
        self.encoder = EncDecRNNTModelSTNOAV.from_config_dict(self.cfg.encoder)

        if self.extract_features_on_the_fly:
            if self.visual_encoder_type == 'avhubert':
                # This has to be here, otherwise the dataloader setup fails.
                self.vis_feat_extractor = AVHubertAVSR.from_pretrained(self.visual_encoder_ckpt_path)
                self.vis_feat_extractor.avsr.encoder.mask_emb.requires_grad = False
                self.vis_feat_extractor.avsr.encoder.label_embs_concat.requires_grad = False
                for p in self.vis_feat_extractor.avsr.encoder.feature_extractor_audio.proj.parameters():
                    p.requires_grad = False

                if self.freeze_visual_encoder:
                    self.vis_feat_extractor.eval()
                    for param in self.vis_feat_extractor.parameters():
                        param.requires_grad = False
                else:
                    self.vis_feat_extractor.train()

                self.vis_feat_extractor.avsr.encoder.feature_extractor_video.resnet.eval()
                for p in self.vis_feat_extractor.avsr.encoder.feature_extractor_video.resnet.parameters():
                    p.requires_grad = False
            else:
                raise ValueError(f"Unsupported visual_encoder_type: {self.visual_encoder_type}")

        # Update config values required by components dynamically
        with open_dict(self.cfg.decoder):
            self.cfg.decoder.vocab_size = len(self.cfg.labels)

        with open_dict(self.cfg.joint):
            self.cfg.joint.num_classes = len(self.cfg.labels)
            self.cfg.joint.vocabulary = self.cfg.labels
            self.cfg.joint.jointnet.encoder_hidden = self.cfg.model_defaults.enc_hidden
            self.cfg.joint.jointnet.pred_hidden = self.cfg.model_defaults.pred_hidden

        self.decoder = EncDecRNNTModelSTNOAV.from_config_dict(self.cfg.decoder)
        self.joint = EncDecRNNTModelSTNOAV.from_config_dict(self.cfg.joint)

        # Setup RNNT Loss
        loss_name, loss_kwargs = self.extract_rnnt_loss_cfg(self.cfg.get("loss", None))

        num_classes = self.joint.num_classes_with_blank - 1  # for standard RNNT and multi-blank

        if loss_name == 'tdt':
            num_classes = num_classes - self.joint.num_extra_outputs

        self.loss = RNNTLoss(
            num_classes=num_classes,
            loss_name=loss_name,
            loss_kwargs=loss_kwargs,
            reduction=self.cfg.get("rnnt_reduction", "mean_batch"),
        )

        if hasattr(self.cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecRNNTModelSTNOAV.from_config_dict(self.cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.cfg.decoding = self.set_decoding_type_according_to_loss(self.cfg.decoding)
        # Setup decoding objects
        self.decoding = RNNTDecoding(
            decoding_cfg=self.cfg.decoding,
            decoder=self.decoder,
            joint=self.joint,
            vocabulary=self.joint.vocabulary,
        )
        # Setup WER calculation
        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        # Whether to compute loss during evaluation
        if 'compute_eval_loss' in self.cfg:
            self.compute_eval_loss = self.cfg.compute_eval_loss
        else:
            self.compute_eval_loss = True

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        self.freeze_nonvision_parameters = self.cfg.get('freeze_nonvision_parameters', False)

        # pretrained_model_path = '/home/jovyan/NeMo/examples/asr/av_ts_asr_transducer/avhubert/model-bin/avsr_cocktail_mcorec_finetune'
        # self.vis_feat_extractor = AVHubertAVSR.from_pretrained(pretrained_model_path)
        # self.audio_transform = AudioTransform(subset="test")
        # self.video_transform = VideoTransform(subset="test")

        # Setup optimization normalization (if provided in config)
        self.setup_optim_normalization()

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        # Setup encoder adapters (from ASRAdapterModelMixin)
        self.setup_adapters()

    def setup(self, stage: Optional[str] = None):
        super().setup(stage=stage)

        if stage == 'fit' and self.freeze_nonvision_parameters:
            logging.info("Freezing non-visual parameters for optimizer setup.")
            self.eval()
            for _, param in self.named_parameters():
                param.requires_grad = False
            
            self.encoder.unfreeze_visual_parameters()

        if self.freeze_rnnt:
            logging.info("Freezing RNNT parameters for optimizer setup.")
            self.decoder.eval()
            self.joint.eval()

            for p in self.decoder.parameters():
                p.requires_grad = False

            for p in self.joint.parameters():
                p.requires_grad = False

        if not self.use_audio_encoder:
            logging.info("Freezing audio encoder parameters for optimizer setup.")
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

            # If we are not using audio encoder, we'll replace it by visual encoder.
            # There, we need to downsample representations by the same factor.
            # Hence, we will unfreeze it.
            # TODO: Make the visual pre-processor part of this module instead of the enocder and let's keep with a shared encoder for all the future experiments.
            assert self.cfg.get('encoder').get('share_visual_preprocessing')
            self.encoder.shared_visual_processing.train()
            for p in self.encoder.shared_visual_processing.parameters():
                p.requires_grad = True


    def setup_optim_normalization(self):
        """
        Helper method to setup normalization of certain parts of the model prior to the optimization step.

        Supported pre-optimization normalizations are as follows:

        .. code-block:: yaml

            # Variation Noise injection
            model:
                variational_noise:
                    std: 0.0
                    start_step: 0

            # Joint - Length normalization
            model:
                normalize_joint_txu: false

            # Encoder Network - gradient normalization
            model:
                normalize_encoder_norm: false

            # Decoder / Prediction Network - gradient normalization
            model:
                normalize_decoder_norm: false

            # Joint - gradient normalization
            model:
                normalize_joint_norm: false
        """
        # setting up the variational noise for the decoder
        if hasattr(self.cfg, 'variational_noise'):
            self._optim_variational_noise_std = self.cfg['variational_noise'].get('std', 0)
            self._optim_variational_noise_start = self.cfg['variational_noise'].get('start_step', 0)
        else:
            self._optim_variational_noise_std = 0
            self._optim_variational_noise_start = 0

        # Setup normalized gradients for model joint by T x U scaling factor (joint length normalization)
        self._optim_normalize_joint_txu = self.cfg.get('normalize_joint_txu', False)
        self._optim_normalize_txu = None

        # Setup normalized encoder norm for model
        self._optim_normalize_encoder_norm = self.cfg.get('normalize_encoder_norm', False)

        # Setup normalized decoder norm for model
        self._optim_normalize_decoder_norm = self.cfg.get('normalize_decoder_norm', False)

        # Setup normalized joint norm for model
        self._optim_normalize_joint_norm = self.cfg.get('normalize_joint_norm', False)

    def extract_rnnt_loss_cfg(self, cfg: Optional[DictConfig]):
        """
        Helper method to extract the rnnt loss name, and potentially its kwargs
        to be passed.

        Args:
            cfg: Should contain `loss_name` as a string which is resolved to a RNNT loss name.
                If the default should be used, then `default` can be used.
                Optionally, one can pass additional kwargs to the loss function. The subdict
                should have a keyname as follows : `{loss_name}_kwargs`.

                Note that whichever loss_name is selected, that corresponding kwargs will be
                selected. For the "default" case, the "{resolved_default}_kwargs" will be used.

        Examples:
            .. code-block:: yaml

                loss_name: "default"
                warprnnt_numba_kwargs:
                    kwargs2: some_other_val

        Returns:
            A tuple, the resolved loss name as well as its kwargs (if found).
        """
        if cfg is None:
            cfg = DictConfig({})

        loss_name = cfg.get("loss_name", "default")

        if loss_name == "default":
            loss_name = resolve_rnnt_default_loss_name()

        loss_kwargs = cfg.get(f"{loss_name}_kwargs", None)

        logging.info(f"Using RNNT Loss : {loss_name}\n" f"Loss {loss_name}_kwargs: {loss_kwargs}")

        return loss_name, loss_kwargs

    def set_decoding_type_according_to_loss(self, decoding_cfg):
        loss_name, loss_kwargs = self.extract_rnnt_loss_cfg(self.cfg.get("loss", None))

        if loss_name == 'tdt':
            decoding_cfg.durations = loss_kwargs.durations
        elif loss_name == 'multiblank_rnnt':
            decoding_cfg.big_blank_durations = loss_kwargs.big_blank_durations

        return decoding_cfg

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        timestamps: Optional[bool] = None,
        override_config: Optional[TranscribeConfig] = None,
    ) -> TranscriptionReturnType:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            audio: (a single or list) of paths to audio files or a np.ndarray/tensor audio array or path 
                to a manifest file.
                Can also be a dataloader object that provides values that can be consumed by the model.
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            partial_hypothesis: Optional[List['Hypothesis']] - A list of partial hypotheses to be used during rnnt
                decoding. This is useful for streaming rnnt decoding. If this is not None, then the length of this
                list should be equal to the length of the audio list.
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels 
                from multi-channel audio. If set to `'average'`, it performs averaging across channels. 
                Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            timestamps: Optional(Bool): timestamps will be returned if set to True as part of hypothesis object 
                (output.timestep['segment']/output.timestep['word']). Refer to `Hypothesis` class for more details. 
                Default is None and would retain the previous state set by using self.change_decoding_strategy().
            override_config: (Optional[TranscribeConfig]) override transcription config pre-defined by the user.
                **Note**: All other arguments in the function will be ignored if override_config is passed.
                You should call this argument as `model.transcribe(audio, override_config=TranscribeConfig(...))`.

        Returns:
            Returns a tuple of 2 items -
            * A list of greedy transcript texts / Hypothesis
            * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
        """

        timestamps = timestamps or (override_config.timestamps if override_config is not None else None)
        if timestamps is not None:
            if timestamps or (override_config is not None and override_config.timestamps):
                logging.info(
                    "Timestamps requested, setting decoding timestamps to True. Capture them in Hypothesis object, \
                        with output[0][idx].timestep['word'/'segment'/'char']"
                )
                return_hypotheses = True
                with open_dict(self.cfg.decoding):
                    self.cfg.decoding.compute_timestamps = True
                    self.cfg.decoding.preserve_alignments = True
            else:
                return_hypotheses = False
                with open_dict(self.cfg.decoding):
                    self.cfg.decoding.compute_timestamps = False
                    self.cfg.decoding.preserve_alignments = False

            self.change_decoding_strategy(self.cfg.decoding, verbose=False)

        return super().transcribe(
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            timestamps=timestamps,
            override_config=override_config,
            # Additional arguments
            partial_hypothesis=partial_hypothesis,
        )

    def change_vocabulary(self, new_vocabulary: List[str], decoding_cfg: Optional[DictConfig] = None):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning a 
        pre-trained model. This method changes only decoder and leaves encoder and pre-processing 
        modules unchanged. For example, you would use it if you want to use pretrained encoder when 
        fine-tuning on data in another language, or when you'd need model to learn capitalization, 
        punctuation and/or special characters.

        Args:
            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
                this is target alphabet.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.

        Returns: None

        """
        if self.joint.vocabulary == new_vocabulary:
            logging.warning(f"Old {self.joint.vocabulary} and new {new_vocabulary} match. Not changing anything.")
        else:
            if new_vocabulary is None or len(new_vocabulary) == 0:
                raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')

            joint_config = self.joint.to_config_dict()
            new_joint_config = copy.deepcopy(joint_config)
            new_joint_config['vocabulary'] = new_vocabulary
            new_joint_config['num_classes'] = len(new_vocabulary)
            del self.joint
            self.joint = EncDecRNNTModelSTNOAV.from_config_dict(new_joint_config)

            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            new_decoder_config.vocab_size = len(new_vocabulary)
            del self.decoder
            self.decoder = EncDecRNNTModelSTNOAV.from_config_dict(new_decoder_config)

            del self.loss
            loss_name, loss_kwargs = self.extract_rnnt_loss_cfg(self.cfg.get('loss', None))
            self.loss = RNNTLoss(
                num_classes=self.joint.num_classes_with_blank - 1, loss_name=loss_name, loss_kwargs=loss_kwargs
            )

            if decoding_cfg is None:
                # Assume same decoding config as before
                decoding_cfg = self.cfg.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(RNNTDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
            decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

            self.decoding = RNNTDecoding(
                decoding_cfg=decoding_cfg,
                decoder=self.decoder,
                joint=self.joint,
                vocabulary=self.joint.vocabulary,
            )

            self.wer = WER(
                decoding=self.decoding,
                batch_dim_index=self.wer.batch_dim_index,
                use_cer=self.wer.use_cer,
                log_prediction=self.wer.log_prediction,
                dist_sync_on_step=True,
            )

            # Setup fused Joint step
            if self.joint.fuse_loss_wer or (
                self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
            ):
                self.joint.set_loss(self.loss)
                self.joint.set_wer(self.wer)

            # Update config
            with open_dict(self.cfg.joint):
                self.cfg.joint = new_joint_config

            with open_dict(self.cfg.decoder):
                self.cfg.decoder = new_decoder_config

            with open_dict(self.cfg.decoding):
                self.cfg.decoding = decoding_cfg

            ds_keys = ['train_ds', 'validation_ds', 'test_ds']
            for key in ds_keys:
                if key in self.cfg:
                    with open_dict(self.cfg[key]):
                        self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

            logging.info(f"Changed decoder to output to {self.joint.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig, verbose=True):
        """
        Changes decoding strategy used during RNNT decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            verbose: (bool) whether to display logging information
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(RNNTDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
        decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

        self.decoding = RNNTDecoding(
            decoding_cfg=decoding_cfg,
            decoder=self.decoder,
            joint=self.joint,
            vocabulary=self.joint.vocabulary,
        )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Setup fused Joint step
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        self.joint.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        if verbose:
            logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='labels')

        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                # During transcription, the model is initially loaded on the CPU.
                # To ensure the correct global_rank and world_size are set,
                # these values must be passed from the configuration.
                global_rank=self.global_rank if not config.get("do_transcribe", False) else config.get("global_rank"),
                world_size=self.world_size if not config.get("do_transcribe", False) else config.get("world_size"),
                dataset=LhotseSpeechToTextBpeDataset(
                    tokenizer=make_parser(
                        labels=config.get('labels', None),
                        name=config.get('parser', 'en'),
                        unk_id=config.get('unk_index', -1),
                        blank_id=config.get('blank_index', -1),
                        do_normalize=config.get('normalize_transcripts', False),
                    ),
                    return_cuts=config.get("do_transcribe", False),
                ),
            )

        dataset = audio_to_text_dataset.get_audio_to_text_char_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            preprocessor_cfg=self._cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToCharDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        batch_sampler = None
        if config.get('use_semi_sorted_batching', False):
            if not isinstance(dataset, _AudioTextDataset):
                raise RuntimeError(
                    "Semi Sorted Batch sampler can be used with AudioToCharDataset or AudioToBPEDataset "
                    f"but found dataset of type {type(dataset)}"
                )
            # set batch_size and batch_sampler to None to disable automatic batching
            batch_sampler = get_semi_sorted_batch_sampler(self, dataset, config)
            config['batch_size'] = None
            config['drop_last'] = False
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            sampler=batch_sampler,
            batch_sampler=None,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.

        if (
            self._train_dl is not None
            and hasattr(self._train_dl, 'dataset')
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config, val=True)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config, val=True)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()

        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "stno_mask": NeuralType(('B', 'S', 'T'), MaskType(), optional=True),
            "stno_mask_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "visual_embeds": NeuralType(('B', 'T', 'S', 'C', 'D'), AcousticEncodedRepresentation(), optional=True),
            "visual_embed_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "num_speakers": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None, stno_mask=None, stno_mask_length=None, visual_embeds=None, visual_embed_lengths=None, num_speakers=None
    ):
        """
        Forward pass of the model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
        and this method only performs the first step - forward of the acoustic model.

        Please refer to the `training_step` in order to see the full `forward` step for training - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.

        Please refer to the `validation_step` in order to see the full `forward` step for inference - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 2 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal, 
            length=processed_signal_length, 
            stno_mask=stno_mask, 
            stno_mask_length=stno_mask_length, 
            visual_embeds=visual_embeds, 
            visual_embed_lengths=visual_embed_lengths,
            num_speakers=num_speakers,
        )
        return encoded, encoded_len

    # def avhubert_get_visual_feats(self, video_frames, video_lengths):
    #     # avhubert_audio_feats = torch.stack([self.audio_transform(signal[i].cpu()) for i in range(len(signal))], 
    #     #                                    dim=0).to(signal.device)
    #     avhubert_video_frames = video_frames
    #     attn_mask = make_non_pad_mask(video_lengths).to(video_frames.device)
    #     av_feats = self.vis_feat_extractor.avsr.encoder(
    #         input_features=None,
    #         video=avhubert_video_frames.permute(0, 2, 1, 3, 4),
    #         attention_mask=attn_mask,
    #     ).last_hidden_state.unsqueeze(2)
    #     return av_feats
    
    def get_visual_feats(self, video_frames, video_lengths, inference_mode='chunk', chunk_length=10, batched: bool = True):
        if self.visual_encoder_type != 'avhubert':
            raise ValueError(f"Unsupported visual_encoder_type: {self.visual_encoder_type}")

        # If not chunking, defer to single-call implementation
        if inference_mode != 'chunk' or chunk_length is None:
            # return self.avhubert_get_visual_feats(signal, signal_len, video_frames, video_lengths)
            raise NotImplementedError("Non-chunked visual feature extraction is not implemented yet.")

        # video_frames expected shape: (B, T, C, H, W)
        # video_lengths expected shape: (B,)
        B = video_frames.shape[0]
        T = video_frames.shape[1]
        fps = 25
        chunk_frames = max(1, int(chunk_length * fps))

        # number of chunks per sample
        n_chunks = (T + chunk_frames - 1) // chunk_frames

        device = video_frames.device
        lengths = video_lengths.to(device)

        if batched:
            # Pad time dimension to multiple of chunk_frames so we can batch all chunks
            pad_len = n_chunks * chunk_frames
            if pad_len != T:
                pad_amount = pad_len - T
                pad_shape = (B, pad_amount, video_frames.shape[2], video_frames.shape[3], video_frames.shape[4])
                pad_tensor = video_frames.new_zeros(pad_shape)
                video_frames_padded = torch.cat([video_frames, pad_tensor], dim=1)
            else:
                video_frames_padded = video_frames

            # reshape into chunks: (B, n_chunks, chunk_frames, C, H, W) -> (B*n_chunks, chunk_frames, C, H, W)
            _, Tp, C, H, W = video_frames_padded.shape
            video_chunks = video_frames_padded.view(B, n_chunks, chunk_frames, C, H, W)
            video_chunks = video_chunks.reshape(B * n_chunks, chunk_frames, C, H, W)

            # build per-chunk lengths so attention mask can be constructed
            chunk_lengths = []
            for i in range(B):
                L = int(lengths[i].item())
                for k in range(n_chunks):
                    start = k * chunk_frames
                    rem = max(0, L - start)
                    chunk_lengths.append(min(rem, chunk_frames))
            chunk_lengths = torch.tensor(chunk_lengths, dtype=torch.long, device=device)

            # attention mask: make_non_pad_mask expects lengths -> returns (B*n_chunks, chunk_frames)
            attn_mask = make_non_pad_mask(chunk_lengths).to(device)

            # permute to (batch, C, T, H, W) as expected by AVHubert encoder
            video_chunks_perm = video_chunks.permute(0, 2, 1, 3, 4)

            # run encoder on chunk-batch and recombine
            av_feats_chunks = self.vis_feat_extractor.avsr.encoder(
                input_features=None, video=video_chunks_perm, attention_mask=attn_mask
            ).last_hidden_state

            # av_feats_chunks: (B*n_chunks, chunk_frames, D) -> reshape to (B, pad_len, D)
            D = av_feats_chunks.shape[-1]
            av_feats = av_feats_chunks.view(B, n_chunks * chunk_frames, D)

            # crop to original T (remove last-chunk padding) and return (B, T, 1, D)
            av_feats = av_feats[:, :T, :]

            # Lately, to support all the speakers, we need to have visual features of shape (B, T, S, C, D)
            return av_feats.unsqueeze(2).unsqueeze(2)
        else:
            # Process chunks one-by-one (batched across samples) to avoid global padding.
            chunk_outputs = []
            for k in range(n_chunks):
                start = k * chunk_frames
                end = min(start + chunk_frames, T)
                cur_len = end - start

                # slice: (B, cur_len, C, H, W)
                chunk = video_frames[:, start:end, :, :, :]

                # per-sample valid lengths for this chunk
                # clip negative values to 0
                chunk_lengths_k = (lengths - start).clamp(min=0, max=cur_len)

                # attention mask: (B, cur_len)
                attn_mask_k = make_non_pad_mask(chunk_lengths_k).to(device)

                # permute to (B, C, T_chunk, H, W)
                chunk_perm = chunk.permute(0, 2, 1, 3, 4)

                # run encoder for this chunk-batch
                out = self.vis_feat_extractor.avsr.encoder(
                    input_features=None, video=chunk_perm, attention_mask=attn_mask_k
                ).last_hidden_state  # (B, cur_len, D)

                chunk_outputs.append(out)

            # concatenate along time to (B, T, D)
            av_feats = torch.cat(chunk_outputs, dim=1)

            # crop to original T (in case) and return (B, T, 1, D)
            av_feats = av_feats[:, :T, :]
            
            assert av_feats.shape[0] == video_frames.shape[0], f"Expected B={video_frames.shape[0]}, got {av_feats.shape[0]}"
            assert av_feats.shape[1] == video_frames.shape[1], f"Expected T={video_frames.shape[1]}, got {av_feats.shape[1]}"

            # Lately, to support all the speakers, we need to have visual features of shape (B, T, S, C, D)
            return av_feats.unsqueeze(2).unsqueeze(2)
        
    def get_visual_encoder_embeds(self, av_feats, video_lengths, num_speakers):
        assert self.extract_features_on_the_fly, "extract_features_on_the_fly must be True to use get_visual_encoder_embeds"

        # av_feats = self.get_visual_feats(video_frames, video_lengths, inference_mode='chunk', chunk_length=10, batched=True)
        processed_visual_embeds = self.encoder.shared_visual_processing(visual_embeds=av_feats, audio_signal=None)
        vis_downsampling_factor = self.encoder.shared_visual_processing.visual_downsampling_factor
        processed_visual_embed_lengths = ((video_lengths.float() / vis_downsampling_factor).ceil()).long()

        assert processed_visual_embeds.shape[2] == 1, f"Expected S=1, got {processed_visual_embeds.shape[2]}"
        
        # The output dimensionality of the encoder is: (B, D, T)
        return processed_visual_embeds.squeeze(2).permute(0, 2, 1), processed_visual_embed_lengths


    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if len(batch) == 16:
            signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_len, utt_ids, spk_ids, visual_embeds, visual_embed_lengths, video_frames, video_lengths, zero_frame_idxes, zero_frame_lengths, num_speakers, sample_id = batch
        else:
            signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_len, utt_ids, spk_ids, visual_embeds, visual_embed_lengths, video_frames, video_lengths, zero_frame_idxes, zero_frame_lengths, num_speakers = batch
            sample_id = None

        if self.extract_features_on_the_fly:
            av_feats = self.get_visual_feats(video_frames, video_lengths, inference_mode='chunk', chunk_length=10, batched=True)
            visual_embeds = av_feats
            visual_embed_lengths = video_lengths
            if self.replace_zero_video_frames_with_zero_embeds:
                for i, (zfi, zfi_len) in enumerate(zip(zero_frame_idxes, zero_frame_lengths)):
                    visual_embeds[i, zfi[:zfi_len], ...] = 0.0


        if self.use_audio_encoder:
            # forward() only performs encoder forwardf
            if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
                encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len, stno_mask=stno_mask, stno_mask_length=stno_mask_len, visual_embeds=visual_embeds, visual_embed_lengths=visual_embed_lengths, num_speakers=num_speakers)
            else:
                encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len, stno_mask=stno_mask, stno_mask_length=stno_mask_len, visual_embeds=visual_embeds, visual_embed_lengths=visual_embed_lengths, num_speakers=num_speakers)
        else:
            # We Assume that we are only using the vision encocder and will train it as a lip-reading model.
            encoded, encoded_len = self.get_visual_encoder_embeds(visual_embeds, video_lengths, num_speakers)
        
        del signal
        del signal_len
        del stno_mask
        del stno_mask_len
        del visual_embeds
        del visual_embed_lengths
        del video_frames
        del video_lengths
        del zero_frame_idxes
        del zero_frame_lengths
        del num_speakers

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            # if (sample_id + 1) % log_every_n_steps == 0:
            #     self.wer.update(
            #         predictions=encoded,
            #         predictions_lengths=encoded_len,
            #         targets=transcript,
            #         targets_lengths=transcript_len,
            #     )
            #     _, scores, words = self.wer.compute()
            #     self.wer.reset()
            #     tensorboard_logs.update({'training_batch_wer': scores.float() / words})
        else:
            # If experimental fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, sample_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        best_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=True
        )

        if isinstance(sample_id, torch.Tensor):
            sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, best_hyp_text))

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 16:
            signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_len, utt_ids, spk_ids, visual_embeds, visual_embed_lengths, video_frames, video_lengths, zero_frame_idxes, zero_frame_lengths, num_speakers, sample_id = batch
        else:
            signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_len, utt_ids, spk_ids, visual_embeds, visual_embed_lengths, video_frames, video_lengths, zero_frame_idxes, zero_frame_lengths, num_speakers = batch
            sample_id = None
        assert len(signal) == 1

        if self.extract_features_on_the_fly:
            av_feats = self.get_visual_feats(video_frames, video_lengths, inference_mode='chunk', chunk_length=10, batched=True)
            visual_embeds = av_feats
            visual_embed_lengths = video_lengths
            if self.replace_zero_video_frames_with_zero_embeds:
                for i, (zfi, zfi_len) in enumerate(zip(zero_frame_idxes, zero_frame_lengths)):
                    visual_embeds[i, zfi[:zfi_len], ...] = 0.0

        # forward() only performs encoder forward
        if self.use_audio_encoder:
            # forward() only performs encoder forwardf
            if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
                encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len, stno_mask=stno_mask, stno_mask_length=stno_mask_len, visual_embeds=visual_embeds, visual_embed_lengths=visual_embed_lengths, num_speakers=num_speakers)
            else:
                encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len, stno_mask=stno_mask, stno_mask_length=stno_mask_len, visual_embeds=visual_embeds, visual_embed_lengths=visual_embed_lengths, num_speakers=num_speakers)
        else:
            # We Assume that we are only using the vision encocder and will train it as a lip-reading model.
            encoded, encoded_len = self.get_visual_encoder_embeds(visual_embeds, video_lengths, num_speakers)

        del signal
        del signal_len
        del stno_mask
        del stno_mask_len
        del visual_embeds
        del visual_embed_lengths
        del video_frames
        del video_lengths
        del zero_frame_idxes
        del zero_frame_lengths
        del num_speakers

        tensorboard_logs = {}

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )

                tensorboard_logs['val_loss'] = loss_value

            # self.wer.update(
            #     predictions=encoded,
            #     predictions_lengths=encoded_len,
            #     targets=transcript,
            #     targets_lengths=transcript_len,
            # )
            # wer, wer_num, wer_denom = self.wer.compute()
            # self.wer.reset()

            self.meeteval_mt_wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                utt_ids=utt_ids,
                spk_ids=spk_ids,
            )

            # tensorboard_logs['val_wer_num'] = wer_num
            # tensorboard_logs['val_wer_denom'] = wer_denom
            # tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics
    
    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        torch.cuda.empty_cache()
        # self.meeteval_mt_wer.reset()
        
    
    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        # WithOptionalCudaGraphs.disable_cuda_graphs_recursive(self, attribute_path="decoding.decoding")
        torch.cuda.empty_cache()
    
    def on_validation_epoch_end(self, sync_metrics: bool = False) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Default DataLoader for Validation set which automatically supports multiple data loaders
        via `multi_validation_epoch_end`.

        If multi dataset support is not required, override this method entirely in base class.
        In such a case, there is no need to implement `multi_validation_epoch_end` either.

        .. note::
            If more than one data loader exists, and they all provide `val_loss`,
            only the `val_loss` of the first data loader will be used by default.
            This default can be changed by passing the special key `val_dl_idx: int`
            inside the `validation_ds` config.

        Args:
            outputs: Single or nested list of tensor outputs from one or more data loaders.

        Returns:
            A dictionary containing the union of all items from individual data_loaders,
            along with merged logs from all data loaders.
        """
        WithOptionalCudaGraphs.disable_cuda_graphs_recursive(self, attribute_path="decoding.decoding")

        print("on_validation_epoch_end")
        # Case where we dont provide data loaders
        if self.validation_step_outputs is not None and len(self.validation_step_outputs) == 0:
            return {}

        # Case where we provide exactly 1 data loader
        if isinstance(self.validation_step_outputs[0], dict):
            output_dict = self.multi_validation_epoch_end(self.validation_step_outputs, dataloader_idx=0)

            save_stm_path = f'{self.trainer.log_dir}/preds_{self.current_epoch}_{self.trainer.global_step}'
            logging.info(f"Saving predictions to {save_stm_path}")
            if not os.path.exists(save_stm_path):
                os.makedirs(save_stm_path, exist_ok=True)

            if self.cfg.train_ds.get('use_lhotse', False):
                gt_segments = self._validation_dl.dataset.segments_collection
            else:
                gt_segments = self._validation_dl.dataset.manifest_processor.collection

            cp_res, tcp_res = self.meeteval_mt_wer.compute(gt_segments, save_stm_path=save_stm_path)
            output_dict['log'].update({'val/cp_wer': cp_res['wer'], 'val/cp_ins': cp_res['ins'], 'val/cp_del': cp_res['del'], 'val/cp_sub': cp_res['sub'], 'val/cp_len': cp_res['len'],
                                    'val/tcp_wer': tcp_res['wer'], 'val/tcp_ins': tcp_res['ins'], 'val/tcp_del': tcp_res['del'], 'val/tcp_sub': tcp_res['sub'], 'val/tcp_len': tcp_res['len']})
            self.meeteval_mt_wer.reset()
            # output_dict['log'].update({'val/cp_wer': 1.0})

            if output_dict is not None and 'log' in output_dict:
                if is_global_rank_zero():
                    print('output_dict', output_dict)
                self.log_dict(output_dict.pop('log'), on_epoch=True, sync_dist=sync_metrics)

            # if torch.distributed.is_available() and torch.distributed.is_initialized():
            #     torch.distributed.barrier()

            self.validation_step_outputs.clear()  # free memory
            return output_dict

        else:  # Case where we provide more than 1 data loader
            output_dict = {'log': {}}

            # The output is a list of list of dicts, outer list corresponds to dataloader idx
            for dataloader_idx, val_outputs in enumerate(self.validation_step_outputs):
                # Get prefix and dispatch call to multi epoch end
                dataloader_prefix = self.get_validation_dataloader_prefix(dataloader_idx)
                dataloader_logs = self.multi_validation_epoch_end(val_outputs, dataloader_idx=dataloader_idx)

                # If result was not provided, generate empty dict
                dataloader_logs = dataloader_logs or {}

                # Perform `val_loss` resolution first (if provided outside logs)
                if 'val_loss' in dataloader_logs:
                    if 'val_loss' not in output_dict and dataloader_idx == self._val_dl_idx:
                        output_dict['val_loss'] = dataloader_logs['val_loss']

                # For every item in the result dictionary
                for k, v in dataloader_logs.items():
                    # If the key is `log`
                    if k == 'log':
                        # Parse every element of the log, and attach the prefix name of the data loader
                        log_dict = {}

                        for k_log, v_log in v.items():
                            # If we are logging the metric, but dont provide it at result level,
                            # store it twice - once in log and once in result level.
                            # Also mark log with prefix name to avoid log level clash with other data loaders
                            if k_log not in output_dict['log'] and dataloader_idx == self._val_dl_idx:
                                new_k_log = k_log

                                # Also insert duplicate key with prefix for ease of comparison / avoid name clash
                                log_dict[dataloader_prefix + k_log] = v_log

                            else:
                                # Simply prepend prefix to key and save
                                new_k_log = dataloader_prefix + k_log

                            # Store log value
                            log_dict[new_k_log] = v_log

                        # Update log storage of individual data loader
                        output_logs = output_dict['log']
                        output_logs.update(log_dict)

                        # Update global log storage
                        output_dict['log'] = output_logs

                    else:
                        # If any values are stored outside 'log', simply prefix name and store
                        new_k = dataloader_prefix + k
                        output_dict[new_k] = v

                self.validation_step_outputs[dataloader_idx].clear()  # free memory

            if 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True, sync_dist=sync_metrics)

            # return everything else
            return output_dict

    # def on_before_optimizer_step(self, optimizer):
    #     super().on_after_backward(optimizer)

        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(f"⚠️ No gradient: {name}")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}
        # wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        # wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        # tensorboard_logs = {**val_loss_log, 'val_wer': wer_num.float() / wer_denom}
        tensorboard_logs = {}
        return {**val_loss_log, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            test_loss_log = {'test_loss': test_loss_mean}
        else:
            test_loss_log = {}
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**test_loss_log, 'test_wer': wer_num.float() / wer_denom}
        return {**test_loss_log, 'log': tensorboard_logs}

    """ Transcription related methods """

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        encoded, encoded_len = self.forward(input_signal=batch[0], input_signal_length=batch[1])
        output = dict(encoded=encoded, encoded_len=encoded_len)
        return output

    def _transcribe_output_processing(
        self, outputs, trcfg: TranscribeConfig
    ) -> Union[List['Hypothesis'], List[List['Hypothesis']]]:
        encoded = outputs.pop('encoded')
        encoded_len = outputs.pop('encoded_len')

        hyp = self.decoding.rnnt_decoder_predictions_tensor(
            encoded,
            encoded_len,
            return_hypotheses=trcfg.return_hypotheses,
            partial_hypotheses=trcfg.partial_hypothesis,
        )
        # cleanup memory
        del encoded, encoded_len

        if trcfg.timestamps:
            hyp = process_timestamp_outputs(
                hyp, self.encoder.subsampling_factor, self.cfg['preprocessor']['window_stride']
            )

        return hyp

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.joint.vocabulary,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    # Compute gradient L2 norms (encoder/decoder/joint/total) and log
    @staticmethod
    def _module_grad_norm(module):
        norms = []
        for _, p in module.named_parameters():
            if p.grad is not None:
                g = p.grad
                try:
                    norms.append(g.detach().data.norm(2))
                except Exception:
                    pass
        if len(norms) == 0:
            return None
        return torch.sqrt(torch.sum(torch.stack([n * n for n in norms])))

    def on_after_backward(self):
        super().on_after_backward()

        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(f"⚠️ No gradient: {name}")

        enc_norm = self._module_grad_norm(self.encoder) if hasattr(self, 'encoder') else None
        dec_norm = self._module_grad_norm(self.decoder) if hasattr(self, 'decoder') else None
        jnt_norm = self._module_grad_norm(self.joint) if hasattr(self, 'joint') else None
        vis_enc_norm = self._module_grad_norm(self.vis_feat_extractor) if hasattr(self, 'vis_feat_extractor') else None

        # Total grad norm across all params
        total_norm_terms = []
        for _, p in self.named_parameters():
            if p.grad is not None:
                try:
                    n = p.grad.detach().data.norm(2)
                    total_norm_terms.append(n * n)
                except Exception:
                    pass
        total_norm = torch.sqrt(torch.sum(torch.stack(total_norm_terms))) if len(total_norm_terms) else None

        # Log via lightning's logger (appears in TensorBoard/WandB)
        if enc_norm is not None:
            self.log('grad_norm/encoder', enc_norm, prog_bar=False, on_step=True, on_epoch=False)
        if dec_norm is not None:
            self.log('grad_norm/decoder', dec_norm, prog_bar=False, on_step=True, on_epoch=False)
        if jnt_norm is not None:
            self.log('grad_norm/joint', jnt_norm, prog_bar=False, on_step=True, on_epoch=False)
        if vis_enc_norm is not None:
            self.log('grad_norm/visual_encoder', vis_enc_norm, prog_bar=False, on_step=True, on_epoch=False)
        if total_norm is not None:
            self.log('grad_norm/total', total_norm, prog_bar=False, on_step=True, on_epoch=False)

        if self._optim_variational_noise_std > 0 and self.global_step >= self._optim_variational_noise_start:
            for param_name, param in self.decoder.named_parameters():
                if param.grad is not None:
                    noise = torch.normal(
                        mean=0.0,
                        std=self._optim_variational_noise_std,
                        size=param.size(),
                        device=param.device,
                        dtype=param.dtype,
                    )
                    param.grad.data.add_(noise)

        if self._optim_normalize_joint_txu:
            T, U = self._optim_normalize_txu
            if T is not None and U is not None:
                for param_name, param in self.encoder.named_parameters():
                    if param.grad is not None:
                        param.grad.data.div_(U)

                for param_name, param in self.decoder.named_parameters():
                    if param.grad is not None:
                        param.grad.data.div_(T)

        if self._optim_normalize_encoder_norm:
            for param_name, param in self.encoder.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm()
                    param.grad.data.div_(norm)

        if self._optim_normalize_decoder_norm:
            for param_name, param in self.decoder.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm()
                    param.grad.data.div_(norm)

        if self._optim_normalize_joint_norm:
            for param_name, param in self.joint.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm()
                    param.grad.data.div_(norm)

    # EncDecRNNTModelSTNOAV is exported in 2 parts
    def list_export_subnets(self):
        return ['encoder', 'decoder_joint']

    # for export
    @property
    def decoder_joint(self):
        return RNNTDecoderJoint(self.decoder, self.joint)

    def set_export_config(self, args):
        if 'decoder_type' in args:
            if hasattr(self, 'change_decoding_strategy'):
                self.change_decoding_strategy(decoder_type=args['decoder_type'])
            else:
                raise Exception("Model does not have decoder type option")
        super().set_export_config(args)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="stt_zh_conformer_transducer_large",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_zh_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_conformer_transducer_large/versions/1.8.0/files/stt_zh_conformer_transducer_large.nemo",
        )
        results.append(model)

        return results

    @property
    def wer(self):
        return self._wer

    @wer.setter
    def wer(self, wer):
        self._wer = wer

    def setup_optimizer_param_groups(self):
        if not hasattr(self, "parameters"):
            self._optimizer_param_groups = None
            return

        known_groups = []
        param_groups = []
        fddt_group = []
        vis_preproc_group = []
        vis_feat_extractor_group = []

        processed_param_names = set()

        for n, p in self.named_parameters():
            if 'fddt' in n:
                processed_param_names.add(n)
                fddt_group.append(p)
            elif 'visual_preprocessing' in n:
                vis_preproc_group.append(p)
                processed_param_names.add(n)
            elif 'vis_feat_extractor' in n:
                vis_feat_extractor_group.append(p)
                processed_param_names.add(n)

        param_groups.append({
            "params": fddt_group, "lr": self.cfg.optim.lr * self.cfg.get('fddt_lr_multiplier', 1)
        })
        param_groups.append({
            "params": vis_preproc_group, "lr": self.cfg.optim.lr * self.cfg.get('vis_preproc_lr_multiplier', 1)
        })
        param_groups.append({
            "params": vis_feat_extractor_group, "lr": self.cfg.optim.lr * self.cfg.get('vis_feat_extractor_lr_multiplier', 1)
        })

        if "optim_param_groups" in self.cfg:
            param_groups_cfg = self.cfg.optim_param_groups
            for group, group_cfg in param_groups_cfg.items():
                module = getattr(self, group, None)
                if module is None:
                    raise ValueError(f"{group} not found in model.")
                elif hasattr(module, "parameters"):
                    known_groups.append(group)
                    new_group = {"params": list(module.parameters())}
                    for k, v in group_cfg.items():
                        new_group[k] = v
                    param_groups.append(new_group)
                else:
                    raise ValueError(f"{group} does not have parameters.")

            other_params = []
            for n, p in self.named_parameters():
                is_unknown = True
                for group in known_groups:
                    if n.startswith(group):
                        is_unknown = False
                if is_unknown:
                    other_params.append(p)

            if len(other_params):
                param_groups = [{"params": other_params}] + param_groups
        else:
            other_params = []
            for n, p in self.named_parameters():
                if n not in processed_param_names:
                    other_params.append(p)
            param_groups.append({"params": other_params})

        self._optimizer_param_groups = param_groups
