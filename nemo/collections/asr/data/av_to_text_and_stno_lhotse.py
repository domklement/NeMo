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
from collections.abc import Iterable as IterableABC
from functools import reduce
import io
import json
import math
import multiprocessing
import os
import random
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import braceexpand
import numpy as np
import torch
from tqdm import tqdm
from torchcodec.decoders import AudioDecoder, VideoDecoder

from nemo.collections.asr.data.text_norm import get_text_norm
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.asr.parts.preprocessing.segment import available_formats as valid_sf_formats
from nemo.collections.common import tokenizers
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.data_utils import DataStoreObject, datastore_object_get, is_datastore_cache_shared, is_datastore_path
from nemo.utils.get_rank import is_global_rank_zero
from lhotse import load_manifest, CutSet, MonoCut
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

__all__ = [
    'LhotseAVToBPEAndSTNODataset',
]

VALID_FILE_FORMATS = ';'.join(['wav', 'mp3', 'flac', 'opus'] + [fmt.lower() for fmt in valid_sf_formats.keys()])


def _speech_collate_fn(batch, pad_id):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    packed_batch = list(zip(*batch))
    # Expecting dataset to return fields in this order:
    # 0: audio_signal, 1: audio_length,
    # 2: tokens, 3: tokens_length,
    # 4: stno_mask, 5: stno_mask_length,
    # 6: utt_id, 7: spk_id,
    # 8: visual_embeds, 9: visual_embeds_length,
    # 10: video_frames, 11: video_frames_length,
    # optional 12: sample_id
    if len(packed_batch) == 13:
        audio_lengths = packed_batch[1]
        tokens_lengths = packed_batch[3]
        stno_mask_lengths = packed_batch[5]
        utt_ids = packed_batch[6]
        spk_ids = packed_batch[7]
        visual_embed_lengths = packed_batch[9]
        video_frame_lengths = packed_batch[11]
        sample_ids = packed_batch[12]
    elif len(packed_batch) == 12:
        audio_lengths = packed_batch[1]
        tokens_lengths = packed_batch[3]
        stno_mask_lengths = packed_batch[5]
        utt_ids = packed_batch[6]
        spk_ids = packed_batch[7]
        visual_embed_lengths = packed_batch[9]
        video_frame_lengths = packed_batch[11]
        sample_ids = None
    else:
        raise ValueError("Expects 12 or 13 tensors in the batch!")

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    has_tokens = tokens_lengths[0] is not None
    if has_tokens:
        max_tokens_len = max(tokens_lengths).item()
    has_stno_mask = stno_mask_lengths[0] is not None
    if has_stno_mask:
        max_stno_mask_len = max(stno_mask_lengths).item()
    has_visual_embed = visual_embed_lengths[0] is not None
    if has_visual_embed:
        max_visual_embed_len = max(visual_embed_lengths).item()
    has_video = video_frame_lengths[0] is not None
    if has_video:
        max_video_frame_len = max(video_frame_lengths).item()

    audio_signal, tokens, stno_masks, visual_embeds, video_frames_list = [], [], [], [], []
    for b in batch:
        # unpack according to returned tuple length
        if len(b) == 12:
            sig, sig_len, tokens_i, tokens_i_len, stno_mask_i, stno_mask_i_len, utt_id, spk_id, visual_embed_i, visual_embed_i_len, video_frames_i, video_frames_i_len = b
        else:
            sig, sig_len, tokens_i, tokens_i_len, stno_mask_i, stno_mask_i_len, utt_id, spk_id, visual_embed_i, visual_embed_i_len, video_frames_i, video_frames_i_len, _ = b

        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        if has_tokens:
            tokens_i_len = tokens_i_len.item()
            if tokens_i_len < max_tokens_len:
                pad = (0, max_tokens_len - tokens_i_len)
                tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
            tokens.append(tokens_i)
        if has_stno_mask:
            stno_mask_i_len = stno_mask_i_len.item()
            if stno_mask_i_len < max_stno_mask_len:
                pad = (0, max_stno_mask_len - stno_mask_i_len)
                stno_mask_i = torch.nn.functional.pad(stno_mask_i, pad)
            stno_masks.append(stno_mask_i)
        if has_visual_embed:
            visual_embed_i_len = visual_embed_i_len.item()
            if visual_embed_i_len < max_visual_embed_len:
                pad = (0, 0, 0, 0, 0, max_visual_embed_len - visual_embed_i_len)
                visual_embed_i = torch.nn.functional.pad(visual_embed_i, pad)
            visual_embeds.append(visual_embed_i)
        
        # We don't have to normalize the empty videos. If one of them is empty, all of the videos will be empty due to the way the dataset is written.
        if has_video:
            # video_frames_i is expected shape (T, H, W, C) or empty tensor
            video_len = video_frames_i_len.item()
            if video_len < max_video_frame_len:
                pad = (0, 0, 0, 0, 0, 0, 0, max_video_frame_len - video_len)
                video_frames_i = torch.nn.functional.pad(video_frames_i, pad)
            video_frames_list.append(video_frames_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    if has_tokens:
        tokens = torch.stack(tokens)
        tokens_lengths = torch.stack(tokens_lengths)
    else:
        tokens = None
        tokens_lengths = None
    if has_stno_mask:
        stno_masks = torch.stack(stno_masks)
        stno_mask_lengths = torch.stack(stno_mask_lengths)
    else:
        stno_masks, stno_mask_lengths = None, None
    if has_visual_embed:
        visual_embeds = torch.stack(visual_embeds)
        visual_embed_lengths = torch.stack(visual_embed_lengths)
    else:
        visual_embeds, visual_embed_lengths = None, None

    if has_video:
        video_frames = torch.stack(video_frames_list)
        video_frame_lengths = torch.stack(video_frame_lengths)
    else:
        video_frames, video_frame_lengths = None, None

    utt_ids = torch.tensor(utt_ids, dtype=torch.int32)
    spk_ids = torch.tensor(spk_ids, dtype=torch.int32)
    
    if sample_ids is None:
        return (
            audio_signal,
            audio_lengths,
            tokens,
            tokens_lengths,
            stno_masks,
            stno_mask_lengths,
            utt_ids,
            spk_ids,
            visual_embeds,
            visual_embed_lengths,
            video_frames,
            video_frame_lengths,
        )
    else:
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
        return (
            audio_signal,
            audio_lengths,
            tokens,
            tokens_lengths,
            stno_masks,
            stno_mask_lengths,
            utt_ids,
            spk_ids,
            visual_embeds,
            visual_embed_lengths,
            video_frames,
            video_frame_lengths,
            sample_ids,
        )


class LhotseAVToBPEAndSTNODataset(torch.utils.data.Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    In practice, the dataset and manifest used for character encoding and byte pair encoding
    are exactly the same. The only difference lies in how the dataset tokenizes the text in
    the manifest.

    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        tokenizer: A subclass of the Tokenizer wrapper found in the common collection,
            nemo.collections.common.tokenizers.TokenizerSpec. ASR Models support a subset of
            all available tokenizers.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        trim: Whether to trim silence segments
        use_start_end_token: Boolean which dictates whether to add [BOS] and [EOS]
            tokens to beginning and ending of speech respectively.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
        channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
        manifest_parse_func: Optional function to parse manifest entries. Defaults to None.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'stno_masks': NeuralType(('B', 'S', 'T'), MaskType()),
            'stno_mask_length': NeuralType(tuple('B'), LengthsType()),
            'utterance_id': NeuralType(tuple('B'), VoidType()),
            'speaker_id': NeuralType(tuple('B'), VoidType()),
            'visual_embeds': NeuralType(('B', 'T', 'N', 'C'), AudioSignal()),
            'visual_embeds_length': NeuralType(tuple('B'), LengthsType()),
            'video_frames': NeuralType(('B', 'T', 'H', 'W', 'C'), AudioSignal()),
            'video_frames_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        use_start_end_token: bool = True,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        manifest_parse_func: Optional[Callable] = None,
        audio_downsampling_factor: int = 1,
        max_training_rand_seg_duration: Optional[int] = None,
        val: bool = False,
        # Default values are in the get function that passes args from config.
        return_audio: bool = True,
        return_stno: bool = True,
        return_visual_features: bool = True,
        return_video: bool = False,
        visual_features_key: Optional[str] = 'av_hubert_lip_features',
        video_key: Optional[str] = 'per_spk_face_crop_videos',
        use_asd_for_stno: bool = False,
    ):
        print("VAL:", val)
        if use_start_end_token and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            self.bos_id = tokenizer.bos_id
        else:
            self.bos_id = None

        if use_start_end_token and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            self.eos_id = tokenizer.eos_id
        else:
            self.eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            self.pad_id = tokenizer.pad_id
        else:
            self.pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer
                # TODO: Make this configurable
                self.text_norm = get_text_norm('whisper_nsf')

            def __call__(self, *args):
                if isinstance(args[0], List) and self.is_aggregate:
                    t = []
                    for span in args[0]:
                        t.extend(self._tokenizer.text_to_ids(span['str'], span['lang']))
                    return t

                args = tuple(self.text_norm(x) for x in args)

                t = self._tokenizer.text_to_ids(*args)
                return t
        
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)

        self.cutset = load_manifest(manifest_filepath)
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.sample_rate = sample_rate
        self.max_training_rand_seg_duration = max_training_rand_seg_duration
        self.return_sample_id = return_sample_id
        self.trim = trim
        self.channel_selector = channel_selector
        self.audio_downsampling_factor = audio_downsampling_factor
        self.val = val
        self.return_audio = return_audio
        self.return_stno = return_stno
        self.return_visual_features = return_visual_features
        self.return_video = return_video
        self.visual_features_key = visual_features_key
        self.video_key = video_key
        self.use_asd_for_stno = use_asd_for_stno
        
        self.VIDEO_FPS = 25
        
        self.spk_cut_list = []

        for i, c in enumerate(self.cutset):
            spks = sorted(CutSet.from_cuts([c]).speakers)
            for s in spks:
                self.spk_cut_list.append((i, s, c))

        # self.spk_cut_list = self.spk_cut_list[:10]
            
    def __len__(self):
        return len(self.spk_cut_list)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.spk_cut_list):
            raise IndexError("Index out of range")

        utt_id, spk, cut = self.spk_cut_list[idx]
        if abs(cut.recording.has_video and cut.recording.sources[0].video.fps - self.VIDEO_FPS) > 1e-2:
            raise ValueError(f"Cut video fps {cut.recording.sources[0].video.fps} does not match dataset fps {self.VIDEO_FPS}")

        cut_duration = cut.duration
        spk_specific_supervisions = list(filter(lambda s: s.speaker == spk, cut.supervisions ))
        
        if self.val:
            rand_start = 0.0
            rand_end = cut_duration
        else:
            if self.max_training_rand_seg_duration is None or cut_duration <= self.max_training_rand_seg_duration:
                rand_start = 0.0
                rand_end = cut_duration
            else:
                rand_start = random.uniform(0, cut_duration - self.max_training_rand_seg_duration)
                rand_end = rand_start + self.max_training_rand_seg_duration

        start_sample = int(rand_start * self.sample_rate)
        start_second = start_sample / self.sample_rate
        end_sample = int(rand_end * self.sample_rate)
        end_second = end_sample / self.sample_rate
        start_vid_idx = int(start_second * self.VIDEO_FPS) # Assuming 25 fps
        end_vid_idx = int(end_second * self.VIDEO_FPS)

        selected_supervisions = []
        for sup in spk_specific_supervisions:
            sup_start = sup.start
            sup_end = sup.end
            if sup_start >= rand_start and sup_end <= rand_end:
                selected_supervisions.append(sup)

        spk_to_id = dict([a[::-1] for a in enumerate(sorted(CutSet.from_cuts([cut]).speakers))])
        target_spk_id = spk_to_id[spk]
        # spk_activity_mask = cut.speakers_audio_mask(speaker_to_idx_map=spk_to_id)[:, start_sample:end_sample]

        audio_data = self.featurizer.process(
            cut.recording.sources[0].source,
            offset=start_second,
            duration=end_second - start_second,
            trim=self.trim,
            orig_sr=cut.recording.sampling_rate,
            channel_selector=self.channel_selector,
        )
        audio_data_len = torch.tensor(audio_data.shape[0], dtype=torch.long)

        downsampled_freq = self.sample_rate / self.audio_downsampling_factor
        downsampled_fl_length = audio_data_len if audio_data_len % self.audio_downsampling_factor == 0 else audio_data_len + (self.audio_downsampling_factor - (audio_data_len % self.audio_downsampling_factor))
        downsampled_fl_length = int(downsampled_fl_length / self.audio_downsampling_factor)

        # From now on, rand_end is adjusted to match the padded signal better.
        rand_end = rand_start + downsampled_fl_length / downsampled_freq
        start_sample = int(rand_start * self.sample_rate)
        start_second = start_sample / self.sample_rate
        end_sample = int(rand_end * self.sample_rate)
        end_second = end_sample / self.sample_rate
        start_vid_idx = int(start_second * self.VIDEO_FPS) # Assuming 25 fps
        end_vid_idx = int(end_second * self.VIDEO_FPS)
        
        tokenized_transcript = torch.tensor(self.tokenizer(' '.join([sup.text for sup in selected_supervisions]))).long()
        tokenized_transcript_len = torch.tensor(len(tokenized_transcript), dtype=torch.long)

        spk_activity_mask = torch.zeros((len(spk_to_id), downsampled_fl_length))
        if self.use_asd_for_stno:
            spk_to_asd_logits = dict()
            all_speakers = spk_to_id.keys()
            max_len = 0
            for speaker in all_speakers:
                with open(cut.custom['per_spk_asd'][speaker], 'r') as f:
                    spk_to_asd_logits[speaker] = json.load(f)
                    if len(spk_to_asd_logits[speaker]) > max_len:
                        max_len = len(spk_to_asd_logits[speaker])
            
            assert self.VIDEO_FPS % downsampled_freq == 0, f"Video FPS {self.VIDEO_FPS} is not divisible by downsampled frequency {downsampled_freq}"
            video_downsampling_factor = int(self.VIDEO_FPS // downsampled_freq)
            for speaker in all_speakers:
                assert len(spk_to_asd_logits[speaker]) == max_len, f"ASD length mismatch for speaker {speaker} in cut {cut.id}"
                # FPS - 25Hz, We need 12.5 -> avg downsample.
                # We need to either shorten or pad the logits to be able to perform the downsampling well.
                
                # We need to pad and downsample the ASD logits.
                spk_asd_logits = list(spk_to_asd_logits[speaker].values())[start_vid_idx:end_vid_idx]
                if len(spk_asd_logits) < video_downsampling_factor*downsampled_fl_length:
                    spk_asd_logits = spk_asd_logits + [0.0] * (video_downsampling_factor*downsampled_fl_length - len(spk_asd_logits))
                elif len(spk_asd_logits) > video_downsampling_factor*downsampled_fl_length:
                    spk_asd_logits = spk_asd_logits[:video_downsampling_factor*downsampled_fl_length]
                spk_asd_logits = torch.tensor(spk_asd_logits, dtype=torch.float32).reshape(downsampled_fl_length, video_downsampling_factor).mean(dim=1)

                assert len(spk_asd_logits) == downsampled_fl_length, f"ASD length after downsampling mismatch for speaker {speaker} in cut {cut.id}"
                spk_activity_mask[spk_to_id[speaker], :] = (spk_asd_logits > 0).float()
        else:
            for s in cut.supervisions:
                if not self.tokenizer(s.text):
                    continue
                if s.start < rand_start or s.end > rand_end:
                    continue
                sup_start = s.start - rand_start
                sup_end = s.end - rand_start
                start_idx = int(sup_start * downsampled_freq)
                end_idx = int(sup_end * downsampled_freq)
                spk_activity_mask[spk_to_id[s.speaker], start_idx:end_idx] = 1.
        
        stno_mask = self._create_stno_masks(spk_activity_mask, spk_to_id[spk])
        stno_len = torch.tensor(stno_mask.shape[1], dtype=torch.long)

        if self.return_visual_features and self.visual_features_key is not None:
            if self.visual_features_key not in cut.custom:
                raise ValueError(f"Visual features key {self.visual_features_key} not found in cut custom.")
            
            per_spk_vis_feat_paths = cut.custom[self.visual_features_key]
            if spk not in per_spk_vis_feat_paths:
                raise ValueError(f"Speaker {spk} not found in visual features paths.")

            visual_embeds = torch.load(per_spk_vis_feat_paths[spk], map_location='cpu', mmap=True)
            if len(visual_embeds.shape) == 2: # Shape: (time, layers, feature_dim)
                visual_embeds = visual_embeds.unsqueeze(1)

            visual_embeds = visual_embeds[start_vid_idx:end_vid_idx, ...]

            if len(visual_embeds.shape) == 2: # Shape: (time, layers, feature_dim)
                visual_embeds = visual_embeds.unsqueeze(1)
        else:
            visual_embeds = torch.tensor([])

        if self.return_video and self.video_key is not None:
            if spk not in cut.custom[self.video_key]:
                raise ValueError(f"Speaker {spk} not found in video paths.")
            vid_dec = VideoDecoder(cut.custom[self.video_key][spk])
            video_frames = vid_dec[start_vid_idx:end_vid_idx]
        else:
            video_frames = torch.tensor([])

        return (
            audio_data, 
            audio_data_len, 
            tokenized_transcript, 
            tokenized_transcript_len, 
            stno_mask, 
            stno_len, 
            torch.tensor(utt_id, dtype=torch.long),
            torch.tensor(target_spk_id, dtype=torch.long),
            visual_embeds,
            torch.tensor(len(visual_embeds), dtype=torch.long),
            video_frames,
            torch.tensor(len(video_frames), dtype=torch.long),
        )
    
    @property
    def segments_collection(self):
        """
        This provides an access to the tokenized segments collection to fit the non-lhotse style of WER computation.
        """
        segment_list = []
        for i, c in enumerate(self.cutset):
            segment_list.append(
                SimpleNamespace(
                    id=i,
                    text_tokens= [
                        {
                            'speaker': s.speaker,
                            'text': self.tokenizer(s.text),
                            'start': s.start,
                            'duration': s.duration
                        } for s in c.supervisions
                    ]
                )
            )
        return segment_list
            
        
        
    @staticmethod
    def _create_stno_masks(spk_mask: torch.Tensor, s_index: int):
        non_target_mask = torch.ones(spk_mask.shape[0], dtype=torch.bool)
        non_target_mask[s_index] = False
        sil_frames = (1 - spk_mask).prod(dim=0)
        anyone_else = (1 - spk_mask[non_target_mask]).prod(dim=0)
        target_spk = spk_mask[s_index] * anyone_else
        non_target_spk = (1 - spk_mask[s_index]) * (1 - anyone_else)
        overlapping_speech = spk_mask[s_index] - target_spk
        stno_mask = torch.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], dim=0)
        return stno_mask

        

    def collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=self.pad_id)            
        
