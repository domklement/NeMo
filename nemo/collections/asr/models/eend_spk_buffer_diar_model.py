# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=E1101
import itertools
from functools import partial
import math
import os
from pathlib import Path
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm
import torchaudio
from torch import nn

from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechE2ESpkDiarRandomChunkDataset
from nemo.collections.asr.data.audio_to_diar_label_lhotse import LhotseAudioToSpeechE2ESpkDiarDataset
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.metrics.meeteval_mt_der import MeetevalDER
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig, SpkDiarizationMixin
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_ats_targets, get_pil_targets, get_pil_targets_hungarian
from nemo.collections.asr.parts.utils.speaker_utils import generate_diarization_output_lines
from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging
from nemo.collections.asr.losses.pit_wrapper import PITLossWrapper
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerFeedForward
from nemo.collections.asr.parts.submodules.multi_head_attention import MultiHeadAttention

__all__ = ['EENDSpkBuffEncLabelModel']


def concat_and_pad(embs: List[torch.Tensor], lengths: List[torch.Tensor]):
    """
    Concatenates lengths[i] first embeddings of embs[i], and pads the rest elements with zeros.

    Args:
        embs: List of embeddings Tensors of (batch_size, n_frames, emb_dim) shape
        lengths: List of lengths Tensors of (batch_size,) shape

    Returns:
        output: concatenated embeddings Tensor of (batch_size, n_frames, emb_dim) shape
        total_lengths: output lengths Tensor of (batch_size,) shape
    """

    if len(embs) != len(lengths):
        raise ValueError(
            f"Length lists must have the same length, but got len(embs) - {len(embs)} "
            f"and len(lengths) - {len(lengths)}."
        )
    device, dtype = embs[0].device, embs[0].dtype
    batch_size, emb_dim = embs[0].shape[0], embs[0].shape[2]

    total_lengths = torch.sum(torch.stack(lengths), dim=0)
    sig_length = total_lengths.max().item()

    output = torch.zeros(batch_size, sig_length, emb_dim, device=device, dtype=dtype)
    start_indices = torch.zeros(batch_size, dtype=torch.int64, device=device)

    for emb, length in zip(embs, lengths):
        end_indices = start_indices + length
        for batch_idx in range(batch_size):
            output[batch_idx, start_indices[batch_idx] : end_indices[batch_idx]] = emb[batch_idx, : length[batch_idx]]
        start_indices = end_indices

    return output, total_lengths


class TALayer(nn.Module):
    def __init__(self, d_model: int, n_speakers: int, ff_expansion_factor: int = 4, n_heads: int = 4, dropout_att: float = 0.1, use_pytorch_sdpa: bool = False, use_pytorch_sdpa_backends: List[str] = None):
        super().__init__()
        self.d_model = d_model
        self.n_speakers = n_speakers
        self.n_heads = n_heads
        self.dropout_att = dropout_att
        self.use_pytorch_sdpa = use_pytorch_sdpa
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.ff_expansion_factor = ff_expansion_factor
        self.fst_lnorm = nn.LayerNorm(self.d_model)
        self.snd_lnorm = nn.LayerNorm(self.d_model)
        self.third_lnorm = nn.LayerNorm(self.d_model)
        self.ff = ConformerFeedForward(self.d_model, self.d_model*self.ff_expansion_factor, dropout=0.1)
        self.self_attn = MultiHeadAttention(
            n_head=self.n_heads,
            n_feat=self.d_model,
            dropout_rate=self.dropout_att,
            use_pytorch_sdpa=self.use_pytorch_sdpa,
            use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
        )
        self.cross_attn = MultiHeadAttention(
            n_head=self.n_heads,
            n_feat=self.d_model,
            dropout_rate=self.dropout_att,
            use_pytorch_sdpa=self.use_pytorch_sdpa,
            use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
        )

    def forward(self, attr_emb, acoustic_emb, ce_mask):
        x = self.fst_lnorm(self.self_attn(attr_emb, attr_emb, attr_emb, mask=None) + attr_emb)
        x = self.snd_lnorm(self.cross_attn(x, acoustic_emb, acoustic_emb, mask=ce_mask) + x)
        return self.third_lnorm(self.ff(x) + x)


class TransformerAttractors(nn.Module):
    def __init__(self, d_model: int, n_speakers: int, n_ta_layers: int = 4, n_heads: int = 4, dropout_att: float = 0.1, attr_dropout: float = 0.0, use_pytorch_sdpa: bool = False, use_pytorch_sdpa_backends: List[str] = None, ff_expansion_factor: int = 4, ta_weights_init_constant: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_speakers = n_speakers
        self.n_ta_layers = n_ta_layers
        self.attr_dropout_prob = attr_dropout

        self.global_embeddings = nn.Parameter(torch.randn(self.n_speakers + 1, self.d_model))
        self.ta_layers = nn.ModuleList([TALayer(self.d_model, self.n_speakers, ff_expansion_factor=ff_expansion_factor, n_heads=n_heads, dropout_att=dropout_att, use_pytorch_sdpa=use_pytorch_sdpa, use_pytorch_sdpa_backends=use_pytorch_sdpa_backends) for _ in range(self.n_ta_layers)])
        self.attractor_proj = nn.Linear(self.d_model, 1)
        self.attr_dropout = nn.Dropout(p=self.attr_dropout_prob)

        # for n, p in self.named_parameters():
        #     if 'norm' not in n:
        #         p.data = p.data * ta_weights_init_constant

        self.ta_layers[-1].third_lnorm.weight.data = self.ta_layers[-1].third_lnorm.weight.data * ta_weights_init_constant

        
    def forward_combiner(self, utt_embedding, alpha=1.0):
        return alpha * nn.functional.sigmoid(utt_embedding).unsqueeze(1) * self.global_embeddings.unsqueeze(0)
    
    def _create_ce_mask(self, emb_seq_lengths):
        max_length = emb_seq_lengths.max()
        att_mask = torch.zeros((len(emb_seq_lengths), self.n_speakers + 1, max_length), dtype=torch.bool, device=emb_seq_lengths.device)
        for i, length in enumerate(emb_seq_lengths):
            att_mask[i, :, length:] = True
        return att_mask

    def forward(self, utt_embed, emb_seq, emb_seq_lengths):
        """
        Args:
            utt_embed: (batch_size, d_model)
            emb_seq: (batch_size, n_frames, d_model)
            emb_seq_lengths: (batch_size,)

        Returns:
            emb_seq: (batch_size, n_frames, d_model)
            att_logits: (batch_size, n_frames, n_speakers + 1)
        """
        ce_mask = self._create_ce_mask(emb_seq_lengths)
        combined_utt_embs = self.forward_combiner(utt_embed)

        for _, layer in enumerate(self.ta_layers):
            combined_utt_embs = layer(combined_utt_embs, emb_seq, ce_mask)
        combined_utt_embs = self.attr_dropout(combined_utt_embs)

        return combined_utt_embs, self.attractor_proj(combined_utt_embs.detach())


class EENDSpkBuffEncLabelModel(ModelPT, ExportableEncDecModel, SpkDiarizationMixin):
    """
    Encoder class for Sortformer diarization model.
    Model class creates training, validation methods for setting up data performing model forward pass.

    This model class expects config dict for:
        * preprocessor
        * Transformer Encoder
        * FastConformer Encoder
        * Sortformer Modules
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly
        from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initialize an Sortformer Diarizer model and a pretrained NEST encoder.
        In this init function, training and validation datasets are prepared.
        """
        random.seed(42)
        self._trainer = trainer if trainer else None
        self._cfg = cfg

        if self._trainer:
            self.world_size = trainer.num_nodes * trainer.num_devices
        else:
            self.world_size = 1

        if self._trainer is not None and self._cfg.get('augmentor', None) is not None:
            self.augmentor = process_augmentations(self._cfg.augmentor)
        else:
            self.augmentor = None
        super().__init__(cfg=self._cfg, trainer=trainer)
        self.preprocessor = EENDSpkBuffEncLabelModel.from_config_dict(self._cfg.preprocessor)

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EENDSpkBuffEncLabelModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.encoder = EENDSpkBuffEncLabelModel.from_config_dict(self._cfg.encoder).to(self.device)
        
        self._init_loss_weights()

        self.eps = 1e-3
        self.negative_init_val = -99
        self.loss = instantiate(self._cfg.loss)
        self.pit_loss_wrapper = PITLossWrapper(self.loss, pit_from="pw_pt")

        self.async_streaming = self._cfg.get("async_streaming", False)
        self.streaming_mode = self._cfg.get("streaming_mode", False)
        self.save_hyperparameters("cfg")
        self._init_eval_metrics()
        speaker_inds = list(range(self._cfg.max_num_of_spks))
        # self.speaker_permutations = torch.tensor(list(itertools.permutations(speaker_inds)))  # Get all permutations

        self.max_batch_dur = self._cfg.get("max_batch_dur", 20000)
        self.concat_and_pad_script = torch.jit.script(concat_and_pad)

        self.force_first_k_streams_to_be_active = self._cfg.get("force_first_k_streams_to_be_active", False)
        self.save_predictions = self._cfg.get("save_predictions", False)
        self.max_num_of_spks = self._cfg.get("max_num_of_spks", 4)
        self.use_bce_for_hungarian = self._cfg.get("use_bce_for_hungarian", False)
        self.use_transformer_attractors = self._cfg.get("use_transformer_attractors", False)
        self.ta_weights_init_constant = self._cfg.get("ta_weights_init_constant", 1)

        if self.use_transformer_attractors:
            self.transformer_attractors = TransformerAttractors(
                d_model=self._cfg.model_defaults.d_model,
                n_speakers=self.max_num_of_spks,
                n_ta_layers=4,
                n_heads=4,
                dropout_att=self._cfg.get("ta_dropout_att", 0.0),
                attr_dropout=self._cfg.get("ta_attr_dropout", 0.0),
                ta_weights_init_constant=self.ta_weights_init_constant,
            )
            # self.emb_seq_ln = nn.LayerNorm(self._cfg.model_defaults.d_model)
            # self.emb_seq_ln.weight.data = self.emb_seq_ln.weight.data * self.ta_weights_init_constant

            self.use_scaled_cos_sim_for_attr_dot = self._cfg.get("use_scaled_cos_sim_for_attr_dot", False)
            if self.use_scaled_cos_sim_for_attr_dot:
                self.attr_dot_scale = nn.Parameter(torch.tensor(self._cfg.get("attr_dot_scale_init_val", 1.0)))

        else:
            self.sortformer_modules = EENDSpkBuffEncLabelModel.from_config_dict(self._cfg.sortformer_modules).to(
                self.device
            )
            self.sortformer_modules.hidden_to_spks = None
            self.sortformer_modules.encoder_proj = None


    def _init_loss_weights(self):
        pil_weight = self._cfg.get("pil_weight", 0.0)
        ats_weight = self._cfg.get("ats_weight", 1.0)
        attr_weight = self._cfg.get("attr_loss_weight", 0.0)

        if pil_weight + ats_weight == 0:
            raise ValueError(f"weights for PIL {pil_weight} and ATS {ats_weight} cannot sum to 0")
        
        self.pil_weight = pil_weight / (pil_weight + ats_weight + attr_weight)
        self.ats_weight = ats_weight / (pil_weight + ats_weight + attr_weight)
        self.attr_weight = attr_weight / (pil_weight + ats_weight + attr_weight)

    def _init_eval_metrics(self):
        """
        If there is no label, then the evaluation metrics will be based on Permutation Invariant Loss (PIL).
        """
        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()

        # self._mlap_train = MultilabelAveragePrecision()
        # self._mlap_valid = MultilabelAveragePrecision()

        self._der_train = MeetevalDER()
        self._der_valid = MeetevalDER()

        self._accuracy_test_ats = MultiBinaryAccuracy()
        self._accuracy_train_ats = MultiBinaryAccuracy()
        self._accuracy_valid_ats = MultiBinaryAccuracy()

    def _reset_train_metrics(self):
        self._accuracy_train.reset()
        self._accuracy_train_ats.reset()

    def _reset_valid_metrics(self):
        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()

    def __setup_dataloader_from_config(self, config, training=False):
        # Switch to lhotse dataloader if specified in the config
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseAudioToSpeechE2ESpkDiarDataset(cfg=config),
            )

        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=self.augmentor
        )

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        logging.info(f"Loading dataset from {config.manifest_filepath}")

        if self._trainer is not None:
            global_rank = self._trainer.global_rank
        else:
            global_rank = 0

        dataset = AudioToSpeechE2ESpkDiarRandomChunkDataset(
            manifest_filepath=config.manifest_filepath,
            soft_label_thres=config.soft_label_thres,
            session_len_sec=config.session_len_sec,
            num_spks=config.num_spks,
            featurizer=featurizer,
            window_stride=self._cfg.preprocessor.window_stride,
            global_rank=global_rank,
            soft_targets=config.soft_targets if 'soft_targets' in config else False,
            device=self.device,
            equalize_recording_lengths=config.get('equalize_recording_lengths', False),
        )

        self.data_collection = dataset.collection
        self.collate_ds = dataset

        dataloader_instance = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=self.collate_ds.eesd_train_collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 1),
            pin_memory=config.get('pin_memory', False),
        )
        return dataloader_instance

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(
            config=train_data_config,
            training=True
        )

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self._validation_dl = self.__setup_dataloader_from_config(
            config=val_data_layer_config,
        )

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        self._test_dl = self.__setup_dataloader_from_config(
            config=test_data_config,
        )

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl
        return None

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "audio_signal": NeuralType(('B', 'T'), audio_eltype),
            "audio_signal_length": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return OrderedDict(
            {
                "preds": NeuralType(('B', 'T', 'C'), ProbsType()),
            }
        )

    def frontend_encoder(self, processed_signal, processed_signal_length, bypass_pre_encode: bool = False):
        """
        Generate encoder outputs from frontend encoder.

        Args:
            processed_signal (torch.Tensor):
                tensor containing audio-feature (mel spectrogram, mfcc, etc.).
            processed_signal_length (torch.Tensor):
                tensor containing lengths of audio signal in integers.

        Returns:
            emb_seq (torch.Tensor):
                tensor containing encoder outputs.
            emb_seq_length (torch.Tensor):
                tensor containing lengths of encoder outputs.
        """
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
        emb_seq, emb_seq_length, global_tokens = self.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length,
            bypass_pre_encode=bypass_pre_encode,
            output_prepend_global_tokens=True,
        )
        emb_seq = emb_seq.transpose(1, 2)
        if hasattr(self, 'sortformer_modules') and self.sortformer_modules.encoder_proj is not None:
            emb_seq = self.sortformer_modules.encoder_proj(emb_seq)
        return emb_seq, emb_seq_length, global_tokens

    def forward_infer(self, emb_seq, emb_seq_length):
        """
        The main forward pass for diarization for offline diarization inference.

        Args:
            emb_seq (torch.Tensor): Tensor containing FastConformer encoder states (embedding vectors).
                Shape: (batch_size, diar_frame_count, emb_dim)
            emb_seq_length (torch.Tensor): Tensor containing lengths of FastConformer encoder states.
                Shape: (batch_size,)

        Returns:
            preds (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
        """
        logits = self.sortformer_modules.forward_speaker_logits(emb_seq)
        return logits

    def forward_infer_transformer_attractors(self, global_tokens, emb_seq, emb_seq_length):
        attractors, attr_logits = self.transformer_attractors(global_tokens[:, 0, :], emb_seq, emb_seq_length)
        attractors = attractors[:, :-1, :] # Remove the last attractor, which should be inactive.
        # EMB_SEQ is (B, T, D)
        if self.use_scaled_cos_sim_for_attr_dot:
            cos_sims = torch.bmm(emb_seq, attractors.transpose(-1,-2)) / (emb_seq.norm(dim=-1).unsqueeze(-1) * attractors.norm(dim=-1).unsqueeze(dim=1))
            logits = cos_sims * self.attr_dot_scale
        else:
            logits = torch.bmm(emb_seq, attractors.transpose(-1,-2))
        return logits, attractors, attr_logits

    def _diarize_forward(self, batch: Any):
        """
        A counterpart of `_transcribe_forward` function in ASR.
        This function is a wrapper for forward pass functions for compataibility
        with the existing classes.

        Args:
            batch (Any): The input batch containing audio signal and audio signal length.

        Returns:
            preds (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
        """
        with torch.no_grad():
            preds = self.forward(audio_signal=batch[0], audio_signal_length=batch[1])
            preds = preds.to('cpu')
            torch.cuda.empty_cache()
        return preds

    def _diarize_output_processing(
        self, outputs, uniq_ids, diarcfg: DiarizeConfig
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[torch.Tensor]]]:
        """
        Processes the diarization outputs and generates RTTM (Real-time Text Markup) files.
        TODO: Currently, this function is not included in mixin test because of
              `ts_vad_post_processing` function.
              (1) Implement a test-compatible function
              (2) `vad_utils.py` has `predlist_to_timestamps` function that is close to this function.
                  Needs to consolute differences and implement the test-compatible function.

        Args:
            outputs (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            uniq_ids (List[str]): List of unique identifiers for each audio file.
            diarcfg (DiarizeConfig): Configuration object for diarization.

        Returns:
            diar_output_lines_list (List[List[str]]): A list of lists, where each inner list contains
                                                      the RTTM lines for a single audio file.
            preds_list (List[torch.Tensor]): A list of tensors containing the diarization outputs
                                             for each audio file.
        """
        preds_list, diar_output_lines_list = [], []
        if outputs.shape[0] == 1:  # batch size = 1
            preds_list.append(outputs)
        else:
            preds_list.extend(torch.split(outputs, [1] * outputs.shape[0]))

        for sample_idx, uniq_id in enumerate(uniq_ids):
            offset = self._diarize_audio_rttm_map[uniq_id]['offset']
            speaker_assign_mat = preds_list[sample_idx].squeeze(dim=0)
            speaker_timestamps = [[] for _ in range(speaker_assign_mat.shape[-1])]
            for spk_id in range(speaker_assign_mat.shape[-1]):
                ts_mat = ts_vad_post_processing(
                    speaker_assign_mat[:, spk_id],
                    cfg_vad_params=diarcfg.postprocessing_params,
                    unit_10ms_frame_count=int(self._cfg.encoder.subsampling_factor),
                    bypass_postprocessing=False,
                )
                ts_mat = ts_mat + offset
                ts_seg_raw_list = ts_mat.tolist()
                ts_seg_list = [[round(stt, 2), round(end, 2)] for (stt, end) in ts_seg_raw_list]
                speaker_timestamps[spk_id].extend(ts_seg_list)

            diar_output_lines = generate_diarization_output_lines(
                speaker_timestamps=speaker_timestamps, model_spk_num=len(speaker_timestamps)
            )
            diar_output_lines_list.append(diar_output_lines)
        if diarcfg.include_tensor_outputs:
            return (diar_output_lines_list, preds_list)
        else:
            return diar_output_lines_list

    def _setup_diarize_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            - manifest_filepath: Path to the manifest file containing audio file paths
              and corresponding speaker labels.

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
            'num_spks': config.get('num_spks', self._cfg.max_num_of_spks),
            'batch_size': batch_size,
            'shuffle': False,
            'soft_label_thres': 0.5,
            'session_len_sec': config['session_len_sec'],
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
        }
        temporary_datalayer = self.__setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def oom_safe_feature_extraction(self, input_signal, input_signal_length):
        """
        This function divides the input signal into smaller sub-batches and processes them sequentially
        to prevent out-of-memory errors during feature extraction.

        Args:
            input_signal (torch.Tensor): The input audio signal.
            input_signal_length (torch.Tensor): The lengths of the input audio signals.

        Returns:
            processed_signal (torch.Tensor): The aggregated audio signal.
                                             The length of this tensor should match the original batch size.
            processed_signal_length (torch.Tensor): The lengths of the processed audio signals.
        """
        input_signal = input_signal.cpu()
        processed_signal_list, processed_signal_length_list = [], []
        max_batch_sec = input_signal.shape[1] / self.preprocessor._cfg.sample_rate
        org_batch_size = input_signal.shape[0]
        div_batch_count = min(int(max_batch_sec * org_batch_size // self.max_batch_dur + 1), org_batch_size)
        div_size = math.ceil(org_batch_size / div_batch_count)

        for div_count in range(div_batch_count):
            start_idx = int(div_count * div_size)
            end_idx = int((div_count + 1) * div_size)
            if start_idx >= org_batch_size:
                break
            input_signal_div = input_signal[start_idx:end_idx, :].to(self.device)
            input_signal_length_div = input_signal_length[start_idx:end_idx]
            processed_signal_div, processed_signal_length_div = self.preprocessor(
                input_signal=input_signal_div, length=input_signal_length_div
            )
            processed_signal_div = processed_signal_div.detach().cpu()
            processed_signal_length_div = processed_signal_length_div.detach().cpu()
            processed_signal_list.append(processed_signal_div)
            processed_signal_length_list.append(processed_signal_length_div)

        processed_signal = torch.cat(processed_signal_list, 0)
        processed_signal_length = torch.cat(processed_signal_length_list, 0)
        assert processed_signal.shape[0] == org_batch_size, (
            f"The resulting batch size of processed signal - {processed_signal.shape[0]} "
            f"is not equal to original batch size: {org_batch_size}"
        )
        processed_signal = processed_signal.to(self.device)
        processed_signal_length = processed_signal_length.to(self.device)
        return processed_signal, processed_signal_length

    def process_signal(self, audio_signal, audio_signal_length):
        """
        Extract audio features from time-series signal for further processing in the model.

        This function performs the following steps:
        1. Moves the audio signal to the correct device.
        2. Normalizes the time-series audio signal.
        3. Extrac audio feature from from the time-series audio signal using the model's preprocessor.

        Args:
            audio_signal (torch.Tensor): The input audio signal.
                Shape: (batch_size, num_samples)
            audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                Shape: (batch_size,)

        Returns:
            processed_signal (torch.Tensor): The preprocessed audio signal.
                Shape: (batch_size, num_features, num_frames)
            processed_signal_length (torch.Tensor): The length of each processed signal.
                Shape: (batch_size,)
        """
        audio_signal, audio_signal_length = audio_signal.to(self.device), audio_signal_length.to(self.device)
        if not self.streaming_mode:
            audio_signal = (1 / (audio_signal.max() + self.eps)) * audio_signal

        batch_total_dur = audio_signal.shape[0] * audio_signal.shape[1] / self.preprocessor._cfg.sample_rate
        if self.max_batch_dur > 0 and self.max_batch_dur < batch_total_dur:
            processed_signal, processed_signal_length = self.oom_safe_feature_extraction(
                input_signal=audio_signal, input_signal_length=audio_signal_length
            )
        else:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=audio_signal, length=audio_signal_length
            )
        # This cache clearning can significantly slow down the training speed.
        # Only perform `empty_cache()` when the input file is extremely large for streaming mode.
        if not self.training and self.streaming_mode:
            del audio_signal, audio_signal_length
            torch.cuda.empty_cache()
        return processed_signal, processed_signal_length

    def forward(
        self,
        audio_signal,
        audio_signal_length,
    ):
        """
        Forward pass for training and inference.

        Args:
            audio_signal (torch.Tensor): Tensor containing audio waveform
                Shape: (batch_size, num_samples)
            audio_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)

        Returns:
            preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Shape: (batch_size, max. diar frame count, num_speakers)
        """
        processed_signal, processed_signal_length = self.process_signal(
            audio_signal=audio_signal, audio_signal_length=audio_signal_length
        )
        processed_signal = processed_signal[:, :, : processed_signal_length.max()]
        if self.streaming_mode:
            raise NotImplementedError("Streaming mode is not implemented for TransformerAttractors")
            # preds = self.forward_streaming(processed_signal, processed_signal_length)
        else:
            emb_seq, emb_seq_length, global_tokens = self.frontend_encoder(
                processed_signal=processed_signal, processed_signal_length=processed_signal_length
            )

            attractors = None
            attr_logits = None
            if self.use_transformer_attractors:
                # preds are logits here!
                preds, attractors, attr_logits = self.forward_infer_transformer_attractors(global_tokens, emb_seq, emb_seq_length)
            else:
                # preds are logits here as well now!
                preds = self.forward_infer(emb_seq, emb_seq_length)

        return preds, attractors, attr_logits

    @property
    def input_names(self):
        return ["chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"]

    @property
    def output_names(self):
        return ["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs", "chunk_pre_encode_lengths"]

    def streaming_input_examples(self):
        """Input tensor examples for exporting streaming version of model"""
        batch_size = 4
        chunk = torch.rand([batch_size, 120, 80]).to(self.device)
        chunk_lengths = torch.tensor([120] * batch_size).to(self.device)
        spkcache = torch.randn([batch_size, 188, 512]).to(self.device)
        spkcache_lengths = torch.tensor([40, 188, 0, 68]).to(self.device)
        fifo = torch.randn([batch_size, 188, 512]).to(self.device)
        fifo_lengths = torch.tensor([50, 88, 0, 90]).to(self.device)
        return chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths

    def streaming_export(self, output: str):
        """Exports the model for streaming inference."""
        input_example = self.streaming_input_examples()
        export_out = self.export(output, input_example=input_example)
        return export_out

    def forward_for_export(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
        """
        This forward pass is for ONNX model export.

        Args:
            chunk (torch.Tensor): Tensor containing audio waveform.
                The term "chunk" refers to the "input buffer" in the speech processing pipeline.
                The size of chunk (input buffer) determines the latency introduced by buffering.
                Shape: (batch_size, feature frame count, dimension)
            chunk_lengths (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)
            spkcache (torch.Tensor): Tensor containing speaker cache embeddings from start
                Shape: (batch_size, spkcache_len, emb_dim)
            spkcache_lengths (torch.Tensor): Tensor containing lengths of speaker cache
                Shape: (batch_size,)
            fifo (torch.Tensor): Tensor containing embeddings from latest chunks
                Shape: (batch_size, fifo_len, emb_dim)
            fifo_lengths (torch.Tensor): Tensor containing lengths of FIFO queue embeddings
                Shape: (batch_size,)

        Returns:
            spkcache_fifo_chunk_preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Shape: (batch_size, max. diar frame count, num_speakers)
            chunk_pre_encode_embs (torch.Tensor): Tensor containing pre-encoded embeddings from the chunk
                Shape: (batch_size, num_frames, emb_dim)
            chunk_pre_encode_lengths (torch.Tensor): Tensor containing lengths of pre-encoded embeddings
                from the chunk (=input buffer).
                Shape: (batch_size,)
        """
        # pre-encode the chunk
        chunk_pre_encode_embs, chunk_pre_encode_lengths = self.encoder.pre_encode(x=chunk, lengths=chunk_lengths)
        chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)

        # concat the embeddings from speaker cache, FIFO queue and the chunk
        spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths = self.concat_and_pad_script(
            [spkcache, fifo, chunk_pre_encode_embs], [spkcache_lengths, fifo_lengths, chunk_pre_encode_lengths]
        )

        # encode the concatenated embeddings
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.frontend_encoder(
            processed_signal=spkcache_fifo_chunk_pre_encode_embs,
            processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
            bypass_pre_encode=True,
        )

        # forward pass for inference
        spkcache_fifo_chunk_preds = self.forward_infer(
            spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths
        )
        return spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths

    def forward_streaming(
        self,
        processed_signal,
        processed_signal_length,
    ):
        """
        The main forward pass for diarization inference in streaming mode.

        Args:
            processed_signal (torch.Tensor): Tensor containing audio waveform
                Shape: (batch_size, num_samples)
            processed_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)

        Returns:
            total_preds (torch.Tensor): Tensor containing predicted speaker labels for the current chunk
                and all previous chunks
                Shape: (batch_size, pred_len, num_speakers)
        """
        streaming_state = self.sortformer_modules.init_streaming_state(
            batch_size=processed_signal.shape[0], async_streaming=self.async_streaming, device=self.device
        )

        batch_size, ch, sig_length = processed_signal.shape
        processed_signal_offset = torch.zeros((batch_size,), dtype=torch.long, device=self.device)

        if dist.is_available() and dist.is_initialized():
            local_tensor = torch.tensor([sig_length], device=processed_signal.device)
            dist.all_reduce(
                local_tensor, op=dist.ReduceOp.MAX, async_op=False
            )  # get max feature length across all GPUs
            max_n_frames = local_tensor.item()
            if dist.get_rank() == 0:
                logging.info(f"Maximum feature length across all GPUs: {max_n_frames}")
        else:
            max_n_frames = sig_length

        if sig_length < max_n_frames:  # need padding to have the same feature length for all GPUs
            pad_tensor = torch.full(
                (batch_size, ch, max_n_frames - sig_length),
                self.negative_init_val,
                dtype=processed_signal.dtype,
                device=processed_signal.device,
            )
            processed_signal = torch.cat([processed_signal, pad_tensor], dim=2)

        att_mod = False
        if self.training:
            rand_num = random.random()
            if rand_num < self.sortformer_modules.causal_attn_rate:
                self.encoder.att_context_size = [-1, self.sortformer_modules.causal_attn_rc]
                # self.transformer_encoder.diag = self.sortformer_modules.causal_attn_rc
                att_mod = True

        total_preds = torch.zeros((batch_size, 0, self.sortformer_modules.n_spk), device=self.device)

        feat_len = processed_signal.shape[2]
        num_chunks = math.ceil(
            feat_len / (self.sortformer_modules.chunk_len * self.sortformer_modules.subsampling_factor)
        )
        streaming_loader = self.sortformer_modules.streaming_feat_loader(
            feat_seq=processed_signal,
            feat_seq_length=processed_signal_length,
            feat_seq_offset=processed_signal_offset,
        )
        for _, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in tqdm(
            streaming_loader,
            total=num_chunks,
            desc="Streaming Steps",
            disable=self.training,
        ):
            streaming_state, total_preds = self.forward_streaming_step(
                processed_signal=chunk_feat_seq_t,
                processed_signal_length=feat_lengths,
                streaming_state=streaming_state,
                total_preds=total_preds,
                left_offset=left_offset,
                right_offset=right_offset,
            )

        if att_mod:
            self.encoder.att_context_size = [-1, -1]
            # self.transformer_encoder.diag = None

        del processed_signal, processed_signal_length

        if sig_length < max_n_frames:  # Discard preds corresponding to padding
            n_frames = math.ceil(sig_length / self.encoder.subsampling_factor)
            total_preds = total_preds[:, :n_frames, :]
        return total_preds

    def forward_streaming_step(
        self,
        processed_signal,
        processed_signal_length,
        streaming_state,
        total_preds,
        left_offset=0,
        right_offset=0,
    ):
        """
        One-step forward pass for diarization inference in streaming mode.

        Args:
            processed_signal (torch.Tensor): Tensor containing audio waveform
                Shape: (batch_size, num_samples)
            processed_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)
            streaming_state (SortformerStreamingState):
                    Tensor variables that contain the streaming state of the model.
                    Find more details in the `SortformerStreamingState` class in `sortformer_modules.py`.

                Attributes:
                    spkcache (torch.Tensor): Speaker cache to store embeddings from start
                    spkcache_lengths (torch.Tensor): Lengths of the speaker cache
                    spkcache_preds (torch.Tensor): The speaker predictions for the speaker cache parts
                    fifo (torch.Tensor): FIFO queue to save the embedding from the latest chunks
                    fifo_lengths (torch.Tensor): Lengths of the FIFO queue
                    fifo_preds (torch.Tensor): The speaker predictions for the FIFO queue parts
                    spk_perm (torch.Tensor): Speaker permutation information for the speaker cache

            total_preds (torch.Tensor): Tensor containing total predicted speaker activity probabilities
                Shape: (batch_size, cumulative pred length, num_speakers)
            left_offset (int): left offset for the current chunk
            right_offset (int): right offset for the current chunk

        Returns:
            streaming_state (SortformerStreamingState):
                    Tensor variables that contain the updated streaming state of the model from
                    this function call.
            total_preds (torch.Tensor):
                Tensor containing the updated total predicted speaker activity probabilities.
                Shape: (batch_size, cumulative pred length, num_speakers)
        """
        chunk_pre_encode_embs, chunk_pre_encode_lengths = self.encoder.pre_encode(
            x=processed_signal, lengths=processed_signal_length
        )

        if self.async_streaming:
            spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths = concat_and_pad(
                [streaming_state.spkcache, streaming_state.fifo, chunk_pre_encode_embs],
                [streaming_state.spkcache_lengths, streaming_state.fifo_lengths, chunk_pre_encode_lengths],
            )
        else:
            spkcache_fifo_chunk_pre_encode_embs = self.sortformer_modules.concat_embs(
                [streaming_state.spkcache, streaming_state.fifo, chunk_pre_encode_embs], dim=1, device=self.device
            )
            spkcache_fifo_chunk_pre_encode_lengths = (
                streaming_state.spkcache.shape[1] + streaming_state.fifo.shape[1] + chunk_pre_encode_lengths
            )

        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.frontend_encoder(
            processed_signal=spkcache_fifo_chunk_pre_encode_embs,
            processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
            bypass_pre_encode=True,
        )
        spkcache_fifo_chunk_preds = self.forward_infer(
            emb_seq=spkcache_fifo_chunk_fc_encoder_embs, emb_seq_length=spkcache_fifo_chunk_fc_encoder_lengths
        )

        spkcache_fifo_chunk_preds = self.sortformer_modules.apply_mask_to_preds(
            spkcache_fifo_chunk_preds, spkcache_fifo_chunk_fc_encoder_lengths
        )
        if self.async_streaming:
            streaming_state, chunk_preds = self.sortformer_modules.streaming_update_async(
                streaming_state=streaming_state,
                chunk=chunk_pre_encode_embs,
                chunk_lengths=chunk_pre_encode_lengths,
                preds=spkcache_fifo_chunk_preds,
                lc=round(left_offset / self.encoder.subsampling_factor),
                rc=math.ceil(right_offset / self.encoder.subsampling_factor),
            )
        else:
            streaming_state, chunk_preds = self.sortformer_modules.streaming_update(
                streaming_state=streaming_state,
                chunk=chunk_pre_encode_embs,
                preds=spkcache_fifo_chunk_preds,
                lc=round(left_offset / self.encoder.subsampling_factor),
                rc=math.ceil(right_offset / self.encoder.subsampling_factor),
            )
        total_preds = torch.cat([total_preds, chunk_preds], dim=1)

        return streaming_state, total_preds

    def _get_aux_train_evaluations(self, preds, targets, target_lens, attr_logits=None) -> dict:
        """
        Compute auxiliary training evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        Args:
            preds (torch.Tensor): Predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)

        Returns:
            (dict): A dictionary containing the following training metrics.
        """
        n_speakers, targets_pil, perm_inds = self._get_permuted_labels(preds, targets, target_lens)

        loss = self.compute_loss(preds, targets_pil, target_lens, attr_logits)

        self._accuracy_train(torch.nn.functional.sigmoid(preds), targets_pil, target_lens)
        train_f1_acc, train_precision, train_recall = self._accuracy_train.compute()

        train_metrics = {
            'trainer/learning_rate': self._optimizer.param_groups[0]['lr'],
            'train_metrics/f1_acc': train_f1_acc,
            'train_metrics/precision': train_precision,
            'train_metrics/recall': train_recall,
        }
        for k in loss.keys():
            train_metrics[f'loss/{k}'] = loss[k]
        return train_metrics

    def training_step(self, batch: list, batch_idx: int) -> dict:
        """
        Performs a single training step.

        Args:
            batch (list): A list containing the following elements:
                - audio_signal (torch.Tensor): The input audio signal in time-series format.
                - audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                - targets (torch.Tensor): The target labels for the batch.
                - target_lens (torch.Tensor): The length of each target sequence in the batch.
            batch_idx (int): The index of the current batch.

        Returns:
            (dict): A dictionary containing the 'loss' key with the calculated loss value.
        """
        audio_signal, audio_signal_length, targets, target_lens, uniq_ids, offsets, rttm_file_paths = batch

        preds, attractors, attr_logits = self.forward(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        with torch.amp.autocast(enabled=False, device_type='cuda' if self.trainer.accelerator.__class__.__name__ == 'CUDAAccelerator' else 'cpu'):
            train_metrics = self._get_aux_train_evaluations(preds.float(), targets.float(), target_lens, attr_logits=attr_logits)

        total_silence = 0
        total_length = 0
        for i in range(targets.shape[0]):
            total_silence += ((targets[i][:target_lens[i]] > 0).sum(-1) == 0).sum()
            total_length += target_lens[i]
        train_metrics['stats/perc_silence'] = total_silence / total_length
        train_metrics['stats/min_logit'] = preds.min()
        train_metrics['stats/max_logit'] = preds.max()

        self._reset_train_metrics()
        self.log_dict(train_metrics, sync_dist=True, on_step=True, on_epoch=False, logger=True)
        return {'loss': train_metrics['loss/loss']}
    
    def _get_permuted_labels(self, preds, targets, target_lens):
        if self.force_first_k_streams_to_be_active or self.use_transformer_attractors:
            n_speakers = (targets.sum(1) > 0).sum(-1)
        else:
            n_speakers = torch.ones((targets.shape[0], ), device=targets.device) * self.max_num_of_spks

        if self.use_bce_for_hungarian:
            targets_pil, perm_inds = get_pil_targets_hungarian(labels=targets.clone(), 
                                                               preds=preds, 
                                                               n_speakers=n_speakers, 
                                                               return_perm_inds=True, 
                                                               use_bce_for_cost_mx_construction=self.use_bce_for_hungarian,
                                                               max_n_speakers=self.max_num_of_spks if self.use_transformer_attractors else None, 
                                                               input_is_probs=False)
        else:
            targets_pil, perm_inds = get_pil_targets_hungarian(labels=targets.clone(), 
                                                               preds=preds, 
                                                               n_speakers=n_speakers, 
                                                               return_perm_inds=True, 
                                                               use_bce_for_cost_mx_construction=self.use_bce_for_hungarian,
                                                               max_n_speakers=self.max_num_of_spks if self.use_transformer_attractors else None, 
                                                               input_is_probs=False)

        return n_speakers, targets_pil, perm_inds
    
    def compute_loss(self, preds, targets_pil, target_lens, attr_logits=None):
        if self.use_transformer_attractors:
            if attr_logits is None:
                raise ValueError("attr_probs is required when use_transformer_attractors is True")

            max_num_spks = 0
            for i in range(targets_pil.shape[0]):
                n_speakers = (targets_pil[i].long().sum(0) > 0).sum()
                targets_pil[i, :, n_speakers:] = -1
                max_num_spks = max(max_num_spks, n_speakers)

            logits_list = [preds[k, : target_lens[k], :] for k in range(preds.shape[0])]
            targets_list = [targets_pil[k, : target_lens[k], :] for k in range(targets_pil.shape[0])]
            logits = torch.cat(logits_list, dim=0)
            labels = torch.cat(targets_list, dim=0)

            # loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, (labels>0).float(), reduction='none')
            loss[torch.where(labels == -1)] = 0
            loss = torch.sum(loss, axis=0) / (labels != -1).sum(axis=0)
            loss[max_num_spks:] = 0

            loss = loss.mean()

            attr_labels = torch.ones_like(attr_logits)
            for i in range(targets_pil.shape[0]):
                n_speakers = (targets_pil[i].long().sum(0) > 0).sum()
                attr_labels[i, n_speakers:, :] = 0
            attr_loss = nn.functional.binary_cross_entropy_with_logits(attr_logits, attr_labels)

            return {
                'loss': self.pil_weight * loss + self.attr_weight * attr_loss,
                'diar_loss': loss,
                'attr_loss': attr_loss,
            }
        else:
            # loss = self.loss(logits=preds, labels=targets_pil, target_lens=target_lens)
            logits_list = [preds[k, : target_lens[k], :] for k in range(preds.shape[0])]
            targets_list = [targets_pil[k, : target_lens[k], :] for k in range(targets_pil.shape[0])]
            logits = torch.cat(logits_list, dim=0)
            labels = torch.cat(targets_list, dim=0)

            # loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, (labels>0).float(), reduction='none')
            loss[torch.where(labels == -1)] = 0
            loss = torch.sum(loss, axis=0) / (labels != -1).sum(axis=0)

            loss = loss.mean()
            return {
                'loss': loss,
                'diar_loss': loss,
            }


    def _get_aux_validation_evaluations(self, preds, targets, target_lens, uniq_ids, attr_logits=None) -> dict:
        """
        Compute auxiliary validation evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        Args:
            preds (torch.Tensor): Predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)

        Returns:
            val_metrics (dict): A dictionary containing the following validation metrics
        """
        n_speakers, targets_pil, perm_inds = self._get_permuted_labels(preds, targets, target_lens)

        if self.save_predictions:
            if not hasattr(self, 'pred_num'):
                self.pred_num = 0
                if os.path.exists(f'{self.trainer.log_dir}'):
                    self.pred_num = len(list(filter(lambda x: x.startswith('pred_matrices'), os.listdir(f'{self.trainer.log_dir}')))) + 1
                
            os.makedirs(f'{self.trainer.log_dir}/pred_matrices_{self.pred_num}/{self.current_epoch}_{self.trainer.global_step}', exist_ok=True)
            for i, uniq_id in enumerate(uniq_ids):
                fnum = len(list(filter(lambda x: x.endswith('.wav'), os.listdir(f'{self.trainer.log_dir}/pred_matrices_{self.pred_num}/{self.current_epoch}_{self.trainer.global_step}')))) + 1
                save_path = f'{self.trainer.log_dir}/pred_matrices_{self.pred_num}/{self.current_epoch}_{self.trainer.global_step}/{uniq_id}_{fnum}.wav'
                print(f'Saving predictions to {save_path}')

                output = torch.empty((targets.shape[1], 2*self.max_num_of_spks), dtype=preds.dtype, device='cpu')
                output[:, 0::2] = targets_pil[i].detach().cpu()*0.8  # Even indices get tensor a
                output[:, 1::2] = torch.nn.functional.sigmoid(preds[i]).detach().cpu() # Odd indices get tensor b
                torchaudio.save(save_path, output.T.float(), sample_rate=16000, format="wav", encoding="PCM_F")

        loss = self.compute_loss(preds, targets_pil, target_lens, attr_logits)

        self._accuracy_valid(torch.nn.functional.sigmoid(preds), targets_pil, target_lens)
        val_f1_acc, val_precision, val_recall = self._accuracy_valid.compute()

        self._accuracy_valid.reset()

        val_metrics = {
            'val_metrics/f1_acc': val_f1_acc,
            'val_metrics/precision': val_precision,
            'val_metrics/recall': val_recall,
        }
        for k in loss.keys():
            val_metrics[f'val_loss/{k}'] = loss[k]

        return val_metrics

    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        """
        Performs a single validation step.

        This method processes a batch of data during the validation phase. It forward passes
        the audio signal through the model, computes various validation metrics, and stores
        these metrics for later aggregation.

        Args:
            batch (list): A list containing the following elements:
                - audio_signal (torch.Tensor): The input audio signal.
                - audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                - targets (torch.Tensor): The target labels for the batch.
                - target_lens (torch.Tensor): The length of each target sequence in the batch.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int, optional): The index of the dataloader in case of multiple
                                            validation dataloaders. Defaults to 0.

        Returns:
            dict: A dictionary containing various validation metrics for this batch.
        """
        audio_signal, audio_signal_length, targets, target_lens, uniq_ids, offsets, rttm_file_paths = batch
        preds, attractors, attr_logits = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )

        if self.use_transformer_attractors:
            # We need to estimate the number of speakers for each utterance.
            # The only thing we need to do is to zero out the logits for the inactive speakers.
            attr_probs = torch.sigmoid(attr_logits)
            for i in range(attr_probs.shape[0]):
                n_speakers = torch.where(attr_probs[i] < .5)[0]
                if not n_speakers.numel():
                    n_speakers = self.max_num_of_spks
                else:
                    n_speakers = n_speakers[0]
                
                preds[i, :, n_speakers:] = -10

        with torch.amp.autocast(enabled=False, device_type='cuda' if self.trainer.accelerator.__class__.__name__ == 'CUDAAccelerator' else 'cpu'):
            val_metrics = self._get_aux_validation_evaluations(preds.float(), targets.float(), target_lens, uniq_ids, attr_logits=attr_logits)
            self._der_valid.update(torch.nn.functional.sigmoid(preds), targets, target_lens, utt_ids=uniq_ids, offsets=offsets, rttm_file_paths=rttm_file_paths)
        if isinstance(self.trainer.val_dataloaders, list) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(val_metrics)
        else:
            self.validation_step_outputs.append(val_metrics)
        return val_metrics

    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        if not outputs:
            logging.warning(f"`outputs` is None; empty outputs for dataloader={dataloader_idx}")
            return None

        val_loss_mean = torch.stack([x['val_loss/loss'] for x in outputs]).mean()
        # val_ats_loss_mean = torch.stack([x['val_ats_loss'] for x in outputs]).mean()
        val_pil_loss_mean = torch.stack([x['val_loss/diar_loss'] for x in outputs]).mean()
        if self.use_transformer_attractors:
            val_attr_loss_mean = torch.stack([x['val_loss/attr_loss'] for x in outputs]).mean()
        val_f1_acc_mean = torch.stack([x['val_metrics/f1_acc'] for x in outputs]).mean()
        val_precision_mean = torch.stack([x['val_metrics/precision'] for x in outputs]).mean()
        val_recall_mean = torch.stack([x['val_metrics/recall'] for x in outputs]).mean()
        # val_f1_acc_ats_mean = torch.stack([x['val_f1_acc_ats'] for x in outputs]).mean()

        self._reset_valid_metrics()

        multi_val_metrics = {
            'val_loss/loss': val_loss_mean,
            # 'val_ats_loss': val_ats_loss_mean,
            'val_loss/diar_loss': val_pil_loss_mean,
            'val_metrics/f1_acc': val_f1_acc_mean,
            'val_metrics/precision': val_precision_mean,
            'val_metrics/recall': val_recall_mean,
        }
        if self.use_transformer_attractors:
            multi_val_metrics['val_loss/attr_loss'] = val_attr_loss_mean
        return {'log': multi_val_metrics}

    def _get_aux_test_batch_evaluations(self, batch_idx: int, preds, targets, target_lens):
        """
        Compute auxiliary validation evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        Args:
            preds (torch.Tensor): Predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)
        """
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        self._accuracy_test(preds, targets_pil, target_lens)
        f1_acc, precision, recall = self._accuracy_test.compute()
        self.batch_f1_accs_list.append(f1_acc)
        self.batch_precision_list.append(precision)
        self.batch_recall_list.append(recall)
        logging.info(f"batch {batch_idx}: f1_acc={f1_acc}, precision={precision}, recall={recall}")

        self._accuracy_test_ats(preds, targets_ats, target_lens)
        f1_acc_ats, precision_ats, recall_ats = self._accuracy_test_ats.compute()
        self.batch_f1_accs_ats_list.append(f1_acc_ats)
        logging.info(
            f"batch {batch_idx}: f1_acc_ats={f1_acc_ats}, precision_ats={precision_ats}, recall_ats={recall_ats}"
        )

        self._accuracy_test.reset()
        self._accuracy_test_ats.reset()

    def test_batch(
        self,
    ):
        """
        Perform batch testing on the model.

        This method iterates through the test data loader, making predictions for each batch,
        and calculates various evaluation metrics. It handles both single and multi-sample batches.
        """
        (
            self.preds_total_list,
            self.batch_f1_accs_list,
            self.batch_precision_list,
            self.batch_recall_list,
            self.batch_f1_accs_ats_list,
        ) = ([], [], [], [], [])

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self._test_dl)):
                audio_signal, audio_signal_length, targets, target_lens = batch
                audio_signal = audio_signal.to(self.device)
                audio_signal_length = audio_signal_length.to(self.device)
                targets = targets.to(self.device)
                preds = self.forward(
                    audio_signal=audio_signal,
                    audio_signal_length=audio_signal_length,
                )
                self._get_aux_test_batch_evaluations(batch_idx, preds, targets, target_lens)
                preds = preds.detach().to('cpu')
                if preds.shape[0] == 1:  # batch size = 1
                    self.preds_total_list.append(preds)
                else:
                    self.preds_total_list.extend(torch.split(preds, [1] * preds.shape[0]))
                torch.cuda.empty_cache()

        logging.info(f"Batch F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_list))}")
        logging.info(f"Batch Precision MEAN: {torch.mean(torch.tensor(self.batch_precision_list))}")
        logging.info(f"Batch Recall MEAN: {torch.mean(torch.tensor(self.batch_recall_list))}")
        logging.info(f"Batch ATS F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_ats_list))}")

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
        # Case where we dont provide data loaders
        if self.validation_step_outputs is not None and len(self.validation_step_outputs) == 0:
            return {}

        # Case where we provide exactly 1 data loader
        if isinstance(self.validation_step_outputs[0], dict):
            output_dict = self.multi_validation_epoch_end(self.validation_step_outputs, dataloader_idx=0)

            other_pred_rttm_files = Path(self.trainer.log_dir).glob("preds*.rttm")
            preds_count = 0
            for fold in other_pred_rttm_files:
                if fold.is_file():
                    preds_count += 1

            der_output = self._der_valid.compute(pred_rttm_path=f'{self.trainer.log_dir}/preds_val_{preds_count}.rttm')
            der_output = {k: float(v) for k, v in der_output.items()}
            der_output_collar = self._der_valid.compute(collar=0.25)
            der_output_collar = {k: float(v) for k, v in der_output_collar.items()}
            self._der_valid.reset()

            output_dict['log']['val_metrics/der'] = der_output['der']
            output_dict['log']['val_metrics/der_scored_speaker_time'] = der_output['scored_speaker_time']
            output_dict['log']['val_metrics/der_missed_speaker_time'] = der_output['missed_speaker_time'] / der_output['scored_speaker_time']
            output_dict['log']['val_metrics/der_falarm_speaker_time'] = der_output['falarm_speaker_time'] / der_output['scored_speaker_time']
            output_dict['log']['val_metrics/der_speaker_error_time'] = der_output['speaker_error_time'] / der_output['scored_speaker_time']
            output_dict['log']['val_metrics/der_collar_0.25'] = der_output_collar['der']
            output_dict['log']['val_metrics/der_collar_0.25_scored_speaker_time'] = der_output_collar['scored_speaker_time']
            output_dict['log']['val_metrics/der_collar_0.25_missed_speaker_time'] = der_output_collar['missed_speaker_time'] / der_output_collar['scored_speaker_time']
            output_dict['log']['val_metrics/der_collar_0.25_falarm_speaker_time'] = der_output_collar['falarm_speaker_time'] / der_output_collar['scored_speaker_time']
            output_dict['log']['val_metrics/der_collar_0.25_speaker_error_time'] = der_output_collar['speaker_error_time'] / der_output_collar['scored_speaker_time']

            if output_dict is not None and 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True, sync_dist=sync_metrics)

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
    
    def on_before_optimizer_step(self, optimizer):
        for p_name, p in self.named_parameters():
            if p.grad is None:
                print('NO GRAD PARAM:', p_name)
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        per_layer_norms = [[] for _ in range(len(self.encoder.layers))]
        pre_encode_layer_norms = []
        for k, v in norms.items():
            if 'pre_encode' in k and v is not None:
                pre_encode_layer_norms.append(v)
            elif 'encoder.layers' in k and v is not None:
                layer_num = int(k.split('encoder.layers.')[1].split('.')[0])
                assert layer_num < len(self.encoder.layers)
                per_layer_norms[layer_num].append(v)

        for l in range(len(self.encoder.layers)):
            if per_layer_norms[l]:
                    per_layer_norms[l] = torch.stack(per_layer_norms[l]).norm(2)

        if len(pre_encode_layer_norms) > 0:
            log_dict = {
                'trainer/grad_l2_norm': norms['grad_2.0_norm_total'],
                'per_block_grad_norms/pre_encode_grad_l2_norm': torch.stack(pre_encode_layer_norms).norm(2),
            }

        for l in range(len(self.encoder.layers)):
            if per_layer_norms[l]:
                log_dict[f'per_block_grad_norms/layer_{l}_grad_l2_norm'] = per_layer_norms[l]

        self.log_dict(log_dict)

    def setup_optimizer_param_groups(self):
        if not hasattr(self, "parameters"):
            self._optimizer_param_groups = None
            return

        def get_all_module_names_of_type(module, instance, name_prefix=''):
            if isinstance(module, instance):
                return [(name_prefix + '.' + n if name_prefix else n) for n, p in module.named_parameters()]

            res = []
            for n, ch in module.named_children():
                new_prefix = f'{name_prefix}.{n}' if name_prefix else n
                res.extend(get_all_module_names_of_type(ch, instance, new_prefix))

            return res

        known_groups = []
        param_groups = []
        
        layer_norm_names = set(get_all_module_names_of_type(self, nn.LayerNorm))
        embed_names = set(get_all_module_names_of_type(self, nn.Embedding))

        learnable_vectors_group = []
        other_group_no_decay = []
        other_group_decay = []
        attr_dot_scale_group = []
        for n, p in self.named_parameters():
            if 'attr_dot_scale' in n:
                attr_dot_scale_group.append(p)
            elif '.spk_buffer.' in n or 'extra_global_tokens' in n:
                print('LEARNABLE VECTOR:', n)
                learnable_vectors_group.append(p)
            elif n in layer_norm_names or n in embed_names:
                other_group_no_decay.append(p)
            else:
                other_group_decay.append(p)

        param_groups = [
            {
                "params": other_group_decay,
            },
            {
                "params": other_group_no_decay,
                "weight_decay": 0.0
            },
            {
                "params": learnable_vectors_group, 
                "lr": self.cfg.optim.lr * self.cfg.get('learnable_vectors_lr_multiplier', 1),
                "weight_decay": 0.0
            },
            {
                "params": attr_dot_scale_group,
                "lr": self.cfg.optim.lr * self.cfg.get('attr_dot_scale_lr_multiplier', 1),
                "weight_decay": 0.0
            },
        ]

        if "optim_param_groups" in self.cfg:
            raise NotImplementedError("optim_param_groups is not implemented")

        # if "optim_param_groups" in self.cfg:
        #     param_groups_cfg = self.cfg.optim_param_groups
        #     for group, group_cfg in param_groups_cfg.items():
        #         module = getattr(self, group, None)
        #         if module is None:
        #             raise ValueError(f"{group} not found in model.")
        #         elif hasattr(module, "parameters"):
        #             known_groups.append(group)
        #             new_group = {"params": list(module.parameters())}
        #             for k, v in group_cfg.items():
        #                 new_group[k] = v
        #             param_groups.append(new_group)
        #         else:
        #             raise ValueError(f"{group} does not have parameters.")

        #     other_params = []
        #     for n, p in self.named_parameters():
        #         is_unknown = True
        #         for group in known_groups:
        #             if n.startswith(group):
        #                 is_unknown = False
        #         if is_unknown:
        #             other_params.append(p)

        #     if len(other_params):
        #         param_groups = [{"params": other_params}] + param_groups
        # else:
        #     param_groups.append({"params": list(filter(lambda x: x not in learnable_vectors_group, self.parameters()))})

        self._optimizer_param_groups = param_groups

    @torch.no_grad()
    def diarize(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        batch_size: int = 1,
        include_tensor_outputs: bool = False,
        postprocessing_yaml: Optional[str] = None,
        num_workers: int = 0,
        verbose: bool = True,
        override_config: Optional[DiarizeConfig] = None,
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[torch.Tensor]]]:
        """One-click runner function for diarization.

        Args:
            audio: (a single or list) of paths to audio files or path to a manifest file.
            batch_size: (int) Batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            include_tensor_outputs: (bool) Include raw speaker activity probabilities to the output.
                See Returns: for more details.
            postprocessing_yaml: Optional(str) Path to .yaml file with postprocessing parameters.
            num_workers: (int) Number of workers for DataLoader.
            verbose: (bool) Whether to display tqdm progress bar.
            override_config: (Optional[DiarizeConfig]) A config to override the default config.

        Returns:
            *if include_tensor_outputs is False: A list of lists of speech segments with a corresponding speaker index,
                in format "[begin_seconds, end_seconds, speaker_index]".
            *if include_tensor_outputs is True: A tuple of the above list
                and list of tensors of raw speaker activity probabilities.
        """
        return super().diarize(
            audio=audio,
            batch_size=batch_size,
            include_tensor_outputs=include_tensor_outputs,
            postprocessing_yaml=postprocessing_yaml,
            num_workers=num_workers,
            verbose=verbose,
            override_config=override_config,
        )
