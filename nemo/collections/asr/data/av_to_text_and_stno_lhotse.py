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
import cv2
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
from lhotse import load_manifest, CutSet, MonoCut, fastcopy
from lhotse.cut import MixedCut
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
    # 12: zero_frame_idxes, 13: zero_frame_idxes_length,
    # 14: num_speakers
    # optional 15: sample_id
    if len(packed_batch) == 16:
        audio_lengths = packed_batch[1]
        tokens_lengths = packed_batch[3]
        stno_mask_lengths = packed_batch[5]
        utt_ids = packed_batch[6]
        spk_ids = packed_batch[7]
        visual_embed_lengths = packed_batch[9]
        video_frame_lengths = packed_batch[11]
        zero_frame_idxes_length = packed_batch[13]
        num_speakers = packed_batch[14]
        sample_ids = packed_batch[15]
    elif len(packed_batch) == 15:
        audio_lengths = packed_batch[1]
        tokens_lengths = packed_batch[3]
        stno_mask_lengths = packed_batch[5]
        utt_ids = packed_batch[6]
        spk_ids = packed_batch[7]
        visual_embed_lengths = packed_batch[9]
        video_frame_lengths = packed_batch[11]
        zero_frame_idxes_length = packed_batch[13]
        num_speakers = packed_batch[14]
        sample_ids = None
    else:
        raise ValueError(f"Expects 15 or 16 tensors in the batch, got {len(packed_batch)}!")
    
    # Get max number of speakers for padding
    max_num_speakers = max(num_speakers).item()

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
    has_zero_frame_idxes = zero_frame_idxes_length[0] is not None
    if has_zero_frame_idxes:
        max_zero_frame_idxes_len = max(zero_frame_idxes_length).item()

    audio_signal, tokens, stno_masks, visual_embeds, video_frames_list, zero_frame_idxes_list = [], [], [], [], [], []
    for b in batch:
        # unpack according to returned tuple length
        if len(b) == 15:
            sig, sig_len, tokens_i, tokens_i_len, stno_mask_i, stno_mask_i_len, utt_id, spk_id, visual_embed_i, visual_embed_i_len, video_frames_i, video_frames_i_len, zero_frame_idxes_i, zero_frame_idxes_i_len, num_speakers_i = b
        else:
            sig, sig_len, tokens_i, tokens_i_len, stno_mask_i, stno_mask_i_len, utt_id, spk_id, visual_embed_i, visual_embed_i_len, video_frames_i, video_frames_i_len, zero_frame_idxes_i, zero_frame_idxes_i_len, num_speakers_i, _ = b

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
            # visual_embed_i shape: (T, S, N, C)
            # Pad time dimension if needed (T_right)
            if visual_embed_i_len < max_visual_embed_len:
                pad = (0, 0, 0, 0, 0, 0, 0, max_visual_embed_len - visual_embed_i_len)
                visual_embed_i = torch.nn.functional.pad(visual_embed_i, pad)
            # Pad speaker dimension if needed (S_right)
            current_num_speakers = visual_embed_i.shape[1] if visual_embed_i.numel() > 0 else 0

            # If the vis embeds tensor is empty, we can't do padding.
            if current_num_speakers < max_num_speakers and visual_embed_i.numel() > 0:
                pad = (0, 0, 0, 0, 0, max_num_speakers - current_num_speakers, 0, 0)
                visual_embed_i = torch.nn.functional.pad(visual_embed_i, pad)
            visual_embeds.append(visual_embed_i)
        
        # We don't have to normalize the empty videos. If one of them is empty, all of the videos will be empty due to the way the dataset is written.
        if has_video:
            # video_frames_i is expected shape (Time, Speakers, Channels, Height, Width) or empty tensor
            video_len = video_frames_i_len.item()
            if video_len < max_video_frame_len:
                pad = (0, 0, 0, 0, 0, 0, 0, 0, 0, max_video_frame_len - video_len)
                video_frames_i = torch.nn.functional.pad(video_frames_i, pad)
            
            current_num_speakers = video_frames_i.shape[1] if video_frames_i.numel() > 0 else 0
            if current_num_speakers < max_num_speakers and video_frames_i.numel() > 0:
                pad = (0, 0, 0, 0, 0, 0, 0, max_num_speakers - current_num_speakers, 0, 0)
                video_frames_i = torch.nn.functional.pad(video_frames_i, pad)

            video_frames_list.append(video_frames_i)

        if has_zero_frame_idxes:
            zero_frame_idxes_len = zero_frame_idxes_i_len.item()
            if zero_frame_idxes_len < max_zero_frame_idxes_len:
                pad = (0, max_zero_frame_idxes_len - zero_frame_idxes_len)
                zero_frame_idxes_i = torch.nn.functional.pad(zero_frame_idxes_i, pad, value=-1)

            zero_frame_idxes_list.append(zero_frame_idxes_i)

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

    if has_zero_frame_idxes:
        zero_frame_idxes = torch.stack(zero_frame_idxes_list)
        zero_frame_idxes_length = torch.stack(zero_frame_idxes_length)
    else:
        zero_frame_idxes, zero_frame_idxes_length = None, None

    utt_ids = torch.tensor(utt_ids, dtype=torch.int32)
    spk_ids = torch.tensor(spk_ids, dtype=torch.int32)
    num_speakers = torch.stack(num_speakers)
    
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
            zero_frame_idxes,
            zero_frame_idxes_length,
            num_speakers,
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
            zero_frame_idxes,
            zero_frame_idxes_length,
            num_speakers,
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
            'visual_embeds': NeuralType(('B', 'T', 'S', 'N', 'C'), AudioSignal()),
            'visual_embeds_length': NeuralType(tuple('B'), LengthsType()),
            'video_frames': NeuralType(('B', 'T', 'S', 'C', 'H', 'W'), AudioSignal()),
            'video_frames_length': NeuralType(tuple('B'), LengthsType()),
            'zero_frame_idxes': NeuralType(('B','T'), LengthsType()),
            'zero_frame_idxes_length': NeuralType(tuple('B'), LengthsType()),
            'num_speakers': NeuralType(tuple('B'), LengthsType()),
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
        video_key: Optional[str] = 'per_spk_lip_crop_videos',
        use_asd_for_stno: bool = False,
        replace_path_prefixes: Optional[List[str]] = None,
        replace_path_replacements: Optional[List[str]] = None,
        audio_transform: Optional[callable] = None,
        video_transform: Optional[callable] = None,
        randomly_mask_audio_signal: bool = False,
        max_random_audio_mask_span_seconds: float = 0.0,
        max_random_audio_mask_ratio: float = 0.3,
        return_all_spks: bool = False,
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
        self.replace_path_prefixes = replace_path_prefixes
        self.replace_path_replacements = replace_path_replacements
        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.randomly_mask_audio_signal = randomly_mask_audio_signal
        self.max_random_audio_mask_span_seconds = max_random_audio_mask_span_seconds
        self.max_random_audio_mask_ratio = max_random_audio_mask_ratio
        self.return_all_spks = return_all_spks

        # Disable audio masking during validation/validation-like usage
        if self.val:
            self.randomly_mask_audio_signal = False
        self.VIDEO_FPS = 25
        
        self.spk_cut_list = []

        for i, c in enumerate(self.cutset):
            spks = sorted(CutSet.from_cuts([c]).speakers)
            for s in spks:
                self.spk_cut_list.append((i, s, c))

        # self.spk_cut_list = self.spk_cut_list[:10]

    def _get_random_mask_range(self, audio_len_samples: int):
        """Return (start_idx, end_idx) for a random contiguous mask or None.

        The returned mask length is sampled uniformly up to
        `max_random_audio_mask_span_seconds` and capped at 30% of actual audio length.
        """
        if not self.randomly_mask_audio_signal or self.max_random_audio_mask_span_seconds <= 0.0:
            return None
        if audio_len_samples <= 0:
            return None

        audio_len_seconds = audio_len_samples / float(self.sample_rate) if self.sample_rate > 0 else 0.0
        max_allowed_seconds = min(self.max_random_audio_mask_span_seconds, self.max_random_audio_mask_ratio * audio_len_seconds)
        if max_allowed_seconds <= 0.0:
            return None

        mask_seconds = random.uniform(0.0, max_allowed_seconds)
        mask_samples = int(mask_seconds * float(self.sample_rate))
        if mask_samples <= 0:
            return None

        if mask_samples >= audio_len_samples:
            mask_samples = max(0, audio_len_samples - 1)
        if mask_samples <= 0:
            return None

        start_idx = random.randint(0, audio_len_samples - mask_samples) if audio_len_samples - mask_samples > 0 else 0
        end_idx = start_idx + mask_samples
        return (start_idx, end_idx)

    def _apply_mask_to_audio(self, audio_data, mask_range):
        """Apply zero mask to `audio_data` for the provided (start,end) range.

        Returns audio_data (possibly converted to a tensor) with masked region zeroed.
        """
        if mask_range is None:
            return audio_data
        start_idx, end_idx = mask_range
        try:
            audio_data[start_idx:end_idx] = 0
            return audio_data
        except Exception:
            ad = torch.as_tensor(audio_data)
            ad[start_idx:end_idx] = 0
            return ad

    def _replace_path(self, path: str) -> str:
        if self.replace_path_prefixes is not None and self.replace_path_replacements is not None:
            for prefix, replacement in zip(self.replace_path_prefixes, self.replace_path_replacements):
                if path.startswith(prefix):
                    path = path.replace(prefix, replacement, 1)
                    break
        return path
    
    def _pad_video_frames_for_track(self, video_frames: np.ndarray, track, total_duration: float) -> np.ndarray:
        """
        Pad video frames before and after based on track offset and duration.
        
        Args:
            video_frames: Video frames array with shape (T, H, W)
            track: The track object containing offset information
            total_duration: Total duration of the mixed cut
            
        Returns:
            Padded video frames with shape (T_padded, H, W)
        """
        track_offset = track.offset
        track_duration = track.cut.duration
        
        # Calculate number of frames to pad before and after
        frames_before = int(round(track_offset * self.VIDEO_FPS))
        frames_after = int(round((total_duration - track_offset - track_duration) * self.VIDEO_FPS))
        
        # Create zero padding frames with same height and width as video_frames
        if len(video_frames) > 0:
            h, w = video_frames.shape[1], video_frames.shape[2]
            pad_before = np.zeros((frames_before, h, w), dtype=video_frames.dtype)
            pad_after = np.zeros((frames_after, h, w), dtype=video_frames.dtype)
            padded_frames = np.concatenate([pad_before, video_frames, pad_after], axis=0)
        else:
            # If video_frames is empty, create all zero frames
            # Use a default frame size or raise an error
            raise ValueError("Cannot pad empty video frames without knowing frame dimensions")
        
        return padded_frames
            
    def __len__(self):
        return len(self.spk_cut_list)
    
    def __getitem__(self, idx):
        """
        MixedCuts are used only for simulated data. 
        """
        if idx < 0 or idx >= len(self.spk_cut_list):
            raise IndexError("Index out of range")

        utt_id, spk, cut = self.spk_cut_list[idx]
        cut = fastcopy(cut)

        if isinstance(cut, MonoCut):
            if cut.recording.has_video and abs(cut.recording.sources[0].video.fps - self.VIDEO_FPS) > 1e-2:
                raise ValueError(f"Cut video fps {cut.recording.sources[0].video.fps} does not match dataset fps {self.VIDEO_FPS}")
        elif isinstance(cut, MixedCut):
            for t in cut.tracks:
                if t.cut.has_video and abs(t.cut.video.fps - self.VIDEO_FPS) > 1e-2:
                    raise ValueError(f"Cut video fps {t.cut.video.fps} does not match dataset fps {self.VIDEO_FPS}")
                
            # We need to check if the tracks have unique speakers.
            # This assumption is later used when retrieving per-speaker videos or embeddings.
            spks = set()
            for t in cut.tracks:
                cut_spks = CutSet.from_cuts([t.cut]).speakers
                for spk in cut_spks:
                    if spk in spks:
                        raise ValueError(f"Speaker {spk} appears in multiple tracks of the same MixedCut {cut.id}, which is not supported.")
                    
                    spks.add(spk)

            # We need to build the per_spk video dict.
            per_spk_videos = dict()
            for t in cut.tracks:
                if t.cut.has_video:
                    per_spk_videos[t.cut.supervisions[0].speaker] = t.cut.recording.sources[0].source

        cut_duration = cut.duration
        spk_specific_supervisions = list(filter(lambda s: s.speaker == spk, cut.supervisions ))

        if self.val:
            rand_start = cut.start
            rand_end = cut.start + cut.duration
        else:
            if self.max_training_rand_seg_duration is None or cut_duration <= self.max_training_rand_seg_duration:
                rand_start = cut.start
                rand_end = cut.start + cut.duration
            else:
                rand_start = random.uniform(cut.start, cut.start + cut.duration - self.max_training_rand_seg_duration)
                # rand_start = cut.start # DEBUG
                rand_end = rand_start + self.max_training_rand_seg_duration

        selected_supervisions = []
        for sup in spk_specific_supervisions:
            sup_start = sup.start
            sup_end = sup.end
            if sup_start >= rand_start and sup_end <= rand_end:
                selected_supervisions.append(sup)

        # We need to adjust the rand end according to the last spoken supervision.
        rand_end = max(sup.end for sup in selected_supervisions) + 0.3 if selected_supervisions else rand_end

        start_sample = int(rand_start * self.sample_rate)
        start_second = start_sample / self.sample_rate
        end_sample = int(rand_end * self.sample_rate)
        end_second = end_sample / self.sample_rate
        start_vid_idx = int(start_second * self.VIDEO_FPS) # Assuming 25 fps
        end_vid_idx = int(end_second * self.VIDEO_FPS)

        spk_to_id = dict([a[::-1] for a in enumerate(sorted(CutSet.from_cuts([cut]).speakers))])
        all_speakers = spk_to_id.keys()
        target_spk_id = spk_to_id[spk]
        # spk_activity_mask = cut.speakers_audio_mask(speaker_to_idx_map=spk_to_id)[:, start_sample:end_sample]

        # RETURN AUDIO
        if self.return_audio:
            if isinstance(cut, MonoCut):
                recording_source = self._replace_path(cut.recording.sources[0].source)
                audio_data = self.featurizer.process(
                    recording_source,
                    offset=start_second,
                    duration=end_second - start_second,
                    trim=self.trim,
                    orig_sr=cut.recording.sampling_rate,
                    channel_selector=self.channel_selector,
                )
            elif isinstance(cut, MixedCut):
                for t in cut.tracks:
                    t.cut.recording.sources[0].source = self._replace_path(t.cut.recording.sources[0].source)
                
                audio_data = cut.load_audio()
                assert audio_data.shape[0] == 1, "Only single channel audio is supported in MixedCut for now."
                audio_data = torch.from_numpy(audio_data[0, start_sample:end_sample])
        else:
            audio_data = torch.tensor([])

        audio_data_len = torch.tensor(audio_data.shape[0], dtype=torch.long)

        if self.return_audio:
            # Optionally apply a random contiguous zero-mask to the audio signal (refactored)
            mask_range = self._get_random_mask_range(int(audio_data.shape[0]))
            if mask_range is not None:
                audio_data = self._apply_mask_to_audio(audio_data, mask_range)

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

        # STNO MASK CREATION
        if self.return_stno:
            spk_activity_mask = torch.zeros((len(spk_to_id), downsampled_fl_length))
            if self.use_asd_for_stno:
                spk_to_asd_logits = dict()
                max_len = 0
                for speaker in all_speakers:
                    with open(self._replace_path(cut.custom['per_spk_asd'][speaker]), 'r') as f:
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
        else:
            stno_mask = torch.tensor([[]])
            stno_len = torch.tensor(0, dtype=torch.long)

        # VISUAL FEATURES LOADING
        if self.return_visual_features and self.visual_features_key is not None:
            assert not isinstance(cut, MixedCut), "MixedCut visual features loading not implemented yet."
            
            if self.return_all_spks:
                # Load visual features for all speakers and stack them
                all_spk_visual_embeds = [self._get_spk_visual_feats(cut, spk, start_vid_idx, end_vid_idx)]
                for speaker in sorted(all_speakers):
                    if speaker == spk:
                        continue
                    spk_visual_embeds = self._get_spk_visual_feats(cut, speaker, start_vid_idx, end_vid_idx)
                    all_spk_visual_embeds.append(spk_visual_embeds)
                # Each spk tensor is (T, N, C). Stack along speaker dim=1 to get (T, S, N, C)
                visual_embeds = torch.stack(all_spk_visual_embeds, dim=1)
            else:
                # Load only target speaker and add speaker dimension of size 1 at dim=1
                visual_embeds = self._get_spk_visual_feats(cut, spk, start_vid_idx, end_vid_idx)
                visual_embeds = visual_embeds.unsqueeze(1)  # (T, 1, N, C)
        else:
            visual_embeds = torch.tensor([])

        # VIDEO FRAMES LOADING
        if self.return_video and self.video_key is not None:
            per_spk_videos = self._build_per_spk_vid_paths(cut)

            if isinstance(cut, MixedCut):
                if spk not in per_spk_videos:
                    raise ValueError(f"Speaker {spk} not found in video paths.")
                
                assert set(per_spk_videos.keys()) == all_speakers, "Mismatch between video paths and speakers in the cut."

                # if self.return_all_spks:
                # Load all video frames for the track
                per_spk_tracks = dict([(t.cut.supervisions[0].speaker, t) for t in cut.tracks])
                track = per_spk_tracks[spk]

                # Target speaker is always going to be the first one.
                if self.return_all_spks:
                    all_spk_video_frames = [
                        self._get_transformed_spk_video_from_mixed_cut(
                            cut_duration, track, spk, per_spk_videos, start_vid_idx, end_vid_idx
                        )[0]
                    ]
                    for other_speaker in sorted(all_speakers):
                        if other_speaker == spk:
                            continue
                        
                        track = per_spk_tracks[other_speaker]
                        spk_video_frames, _ = self._get_transformed_spk_video_from_mixed_cut(
                            cut_duration, track, other_speaker, per_spk_videos, start_vid_idx, end_vid_idx
                        )
                        all_spk_video_frames.append(spk_video_frames)
                    video_frames = torch.stack(all_spk_video_frames, dim=1)  # (T, S, C, H, W)
                else:
                    video_frames, zero_frame_idxes = self._get_transformed_spk_video_from_mixed_cut(
                        cut_duration, track, spk, per_spk_videos, start_vid_idx, end_vid_idx
                    )
                    video_Frames = video_frames.unsqueeze(1)  # 1 speaker.

                zero_frame_idxes = np.array([], dtype=np.int64)  # Not tracking zero frames for all speakers
            else:
                if spk not in cut.custom[self.video_key]:
                    raise ValueError(f"Speaker {spk} not found in video paths.")
                
                if self.return_all_spks:
                    all_spk_video_frames = [
                        self._get_transformed_spk_video_from_mono_cut(
                            cut, spk, start_vid_idx, end_vid_idx
                        )[0]
                    ]
                    for other_speaker in sorted(all_speakers):
                        if other_speaker == spk:
                            continue
                        
                        spk_video_frames, _ = self._get_transformed_spk_video_from_mono_cut(
                            cut, other_speaker, start_vid_idx, end_vid_idx
                        )
                        all_spk_video_frames.append(spk_video_frames)

                    video_frames = torch.stack(all_spk_video_frames, dim=1)  # (T, S, C, H, W)
                    zero_frame_idxes = np.array([], dtype=np.int64)
                else:
                    video_frames, zero_frame_idxes = self._get_transformed_spk_video_from_mono_cut(
                        cut, spk, start_vid_idx, end_vid_idx
                    )

                    video_frames = video_frames.unsqueeze(1) # 1 speaker.
        else:
            video_frames = torch.tensor([])
            zero_frame_idxes = np.array([], dtype=np.int64)

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
            torch.from_numpy(zero_frame_idxes).long(),
            torch.tensor(len(zero_frame_idxes), dtype=torch.long),
            torch.tensor(len(all_speakers), dtype=torch.long) if self.return_all_spks else torch.tensor(1, dtype=torch.long),
        )
    
    def _get_transformed_spk_video_from_mono_cut(self, cut, spk, start_vid_idx, end_vid_idx) -> torch.Tensor:
        vid_dec = VideoDecoder(self._replace_path(cut.custom[self.video_key][spk]), dimension_order="NHWC")
        video_frames = vid_dec[start_vid_idx:end_vid_idx]
        video_frames = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video_frames.numpy()])
        zero_frame_idxes = np.where(video_frames.reshape(video_frames.shape[0], -1).sum(-1) == 0)[0]

        if self.video_transform is not None:
            video_frames = self.video_transform(torch.from_numpy(video_frames).unsqueeze(1)) # Add channel dim

        return video_frames, zero_frame_idxes
    
    def _get_transformed_spk_video_from_mixed_cut(self, cut_duration, track, spk, per_spk_videos, start_vid_idx, end_vid_idx) -> torch.Tensor:
        vid_dec = VideoDecoder(self._replace_path(per_spk_videos[spk]), dimension_order="NHWC")
        video_frames = vid_dec[:]  # Load all frames without indexing
        video_frames = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video_frames.numpy()])
        
        # Pad video frames based on track offset and total cut duration
        video_frames = self._pad_video_frames_for_track(video_frames, track, cut_duration)
        
        # Now extract the relevant segment from the padded video
        video_frames = video_frames[start_vid_idx:end_vid_idx]
        zero_frame_idxes = np.where(video_frames.reshape(video_frames.shape[0], -1).sum(-1) == 0)[0]

        if self.video_transform is not None:
            video_frames = self.video_transform(torch.from_numpy(video_frames).unsqueeze(1)) # Add channel dim

        return video_frames, zero_frame_idxes
    
    def _build_per_spk_vid_paths(self, cut):
        if isinstance(cut, MixedCut):
            per_spk_videos = dict()
            for t in cut.tracks:
                t_cut = t.cut
                if self.video_key in t_cut.custom:
                    per_spk_videos.update(t_cut.custom[self.video_key])
                else:
                    logging.warning(f"Video key {self.video_key} not found in track cut custom for cut {t_cut.id}. Using recording source instead.")
                    per_spk_videos[t_cut.supervisions[0].speaker] = t_cut.recording.sources[0].source
        else:
            per_spk_videos = cut.custom.get(self.video_key, dict())

        return per_spk_videos
    
    def _get_spk_visual_feats(self, cut, spk, start_vid_idx, end_vid_idx):
        if self.visual_features_key not in cut.custom:
                raise ValueError(f"Visual features key {self.visual_features_key} not found in cut custom.")
            
        per_spk_vis_feat_paths = cut.custom[self.visual_features_key]
        if spk not in per_spk_vis_feat_paths:
            raise ValueError(f"Speaker {spk} not found in visual features paths.")

        visual_embeds = torch.load(self._replace_path(per_spk_vis_feat_paths[spk]), map_location='cpu', mmap=True)
        visual_embeds = visual_embeds[start_vid_idx:end_vid_idx, ...]

        if len(visual_embeds.shape) == 2: # Shape: (time, layers, feature_dim)
            visual_embeds = visual_embeds.unsqueeze(1)

        
        return visual_embeds
    
    
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
        
