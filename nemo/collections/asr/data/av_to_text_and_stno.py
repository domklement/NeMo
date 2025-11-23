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
from functools import reduce
import io
import json
import math
import multiprocessing
import os
import random
from collections.abc import Iterable as IterableABC
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import braceexpand
import numpy as np
import torch
from tqdm import tqdm

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

__all__ = [
    'AVToBPEAndSTNODataset',
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
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 11:
        _, audio_lengths, _, tokens_lengths, _, stno_mask_lengths, utt_ids, spk_ids, _, visual_embed_lengths, sample_ids = packed_batch
    elif len(packed_batch) == 10:
        sample_ids = None
        _, audio_lengths, _, tokens_lengths, _, stno_mask_lengths, utt_ids, spk_ids, _, visual_embed_lengths = packed_batch
    else:
        raise ValueError("Expects 10 or 11 tensors in the batch!")
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

    audio_signal, tokens, stno_masks, visual_embeds = [], [], [], []
    for b in batch:
        if len(b) == 10:
            sig, sig_len, tokens_i, tokens_i_len, stno_mask_i, stno_mask_i_len, utt_id, spk_id, visual_embed_i, visual_embed_i_len = b
        else:
            sig, sig_len, tokens_i, tokens_i_len, stno_mask_i, stno_mask_i_len, utt_id, spk_id, visual_embed_i, visual_embed_i_len, _ = b
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

    utt_ids = torch.tensor(utt_ids, dtype=torch.int32)
    spk_ids = torch.tensor(spk_ids, dtype=torch.int32)
    
    if sample_ids is None:
        return audio_signal, audio_lengths, tokens, tokens_lengths, stno_masks, stno_mask_lengths, utt_ids, spk_ids, visual_embeds, visual_embed_lengths
    else:
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
        return audio_signal, audio_lengths, tokens, tokens_lengths, stno_masks, stno_mask_lengths, utt_ids, spk_ids, visual_embeds, visual_embed_lengths, sample_ids


class ASRManifestProcessor:
    """
    Class that processes a manifest json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        bos_id: Id of beginning of sequence symbol to append if not None.
        eos_id: Id of end of sequence symbol to append if not None.
        pad_id: Id of pad symbol. Defaults to 0.
    """

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        index_by_file_id: bool = False,
        manifest_parse_func: Optional[Callable] = None,
    ):
        self.parser = parser

        self.collection = collections.ASRAVTextSTNO(
            manifests_files=manifest_filepath,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
            parse_func=manifest_parse_func,
        )

        ids_to_pop = []
        for i in range(len(self.collection)):
            if not self.collection[i].text_tokens:
                ids_to_pop.append(i)
            
        ids_to_pop.sort()
        print(f'Popping {len(ids_to_pop)} samples')
        print('Before popping: ', len(self.collection))
        
        for i in reversed(ids_to_pop):
            self.collection.pop(i)

        print('After popping: ', len(self.collection))

        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

    def process_text_by_id(self, index: int) -> Tuple[List[int], int]:
        sample = self.collection[index]
        return self.process_text_by_sample(sample)

    def process_text_by_file_id(self, file_id: str) -> Tuple[List[int], int]:
        manifest_idx = self.collection.mapping[file_id][0]
        sample = self.collection[manifest_idx]
        return self.process_text_by_sample(sample)

    def process_text_by_sample(self, text_tokens: List[int]) -> Tuple[List[int], int]:
        t, tl = text_tokens, len(text_tokens)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return t, tl


def expand_sharded_filepaths(sharded_filepaths, shard_strategy: str, world_size: int, global_rank: int):
    valid_shard_strategies = ['scatter', 'replicate']
    if shard_strategy not in valid_shard_strategies:
        raise ValueError(f"`shard_strategy` must be one of {valid_shard_strategies}")

    if isinstance(sharded_filepaths, str):
        # Replace '(' and '[' with '{'
        brace_keys_open = ['(', '[', '<', '_OP_']
        for bkey in brace_keys_open:
            if bkey in sharded_filepaths:
                sharded_filepaths = sharded_filepaths.replace(bkey, "{")

        # Replace ')' and ']' with '}'
        brace_keys_close = [')', ']', '>', '_CL_']
        for bkey in brace_keys_close:
            if bkey in sharded_filepaths:
                sharded_filepaths = sharded_filepaths.replace(bkey, "}")

    if isinstance(sharded_filepaths, str):
        # Brace expand, set escape=False for Windows compatibility
        sharded_filepaths = list(braceexpand.braceexpand(sharded_filepaths, escape=False))

    # Check for distributed and partition shards accordingly
    if world_size > 1:
        if shard_strategy == 'scatter':
            logging.info("All tarred dataset shards will be scattered evenly across all nodes.")

            if len(sharded_filepaths) % world_size != 0:
                logging.warning(
                    f"Number of shards in tarred dataset ({len(sharded_filepaths)}) is not divisible "
                    f"by number of distributed workers ({world_size})."
                )

            begin_idx = (len(sharded_filepaths) // world_size) * global_rank
            end_idx = begin_idx + len(sharded_filepaths) // world_size
            sharded_filepaths = sharded_filepaths[begin_idx:end_idx]
            logging.info(
                "Partitioning tarred dataset: process (%d) taking shards [%d, %d)", global_rank, begin_idx, end_idx
            )

        elif shard_strategy == 'replicate':
            logging.info("All tarred dataset shards will be replicated across all nodes.")
        else:
            raise ValueError(f"Invalid shard strategy ! Allowed values are : {valid_shard_strategies}")

    return sharded_filepaths


def cache_datastore_manifests(
    manifest_filepaths: Union[str, List[str]],
    cache_audio: bool = False,
    shared_cache: Optional[bool] = None,
    num_workers: Optional[int] = None,
    max_num_workers: int = 20,
):
    """Cache manifests and audio from an object store.
    It is assumed that remote manifests are using relative paths.

    Args:
        manifest_filepaths: list of paths to manifest files (list of strings or a string with `,` as separator)
        cache_audio: If True, audio from manifest will also be cached
        shared_cache: Optional, True if cache is shared across all nodes
        num_workers: Optional, number of workers to be used for download
        max_num_workers: max number of workers to be used for download, used when setting num_workers automatically
    """
    if isinstance(manifest_filepaths, str):
        manifest_filepaths = manifest_filepaths.split(',')

    num_datastore_manifests = sum([is_datastore_path(f) for f in manifest_filepaths])

    if num_datastore_manifests > 0:
        # Local utility function
        def cache_data(manifest_filepaths, cache_audio, num_workers, max_num_workers):
            """Cache manifests and audio data from object store."""
            # Determine the number of workers to use
            if num_workers is None:
                num_workers = os.cpu_count() - 1
            num_workers = min(num_workers, max_num_workers)

            # Process each manifest file
            for manifest_file in manifest_filepaths:
                # If manifest is on a data store, then cache it.
                # Otherwise, nothing to do.
                if is_datastore_path(manifest_file):
                    logging.info('Cache manifest file: %s', manifest_file)
                    cached_manifest_file = DataStoreObject(manifest_file).get()
                    logging.info('Cached at: %s', str(cached_manifest_file))

                    if cache_audio:
                        # Each audio file from manifest will be cached.
                        logging.info('Cache audio from manifest file: %s', manifest_file)
                        # Assumes that manifest is using relative paths
                        manifest_dir = os.path.dirname(manifest_file)
                        # Prepare all store objects
                        audio_objects = []
                        with open(cached_manifest_file, 'r') as f:
                            for line in f:
                                item = json.loads(line)
                                store_path = os.path.join(manifest_dir, item['audio_filepath'])
                                audio_objects.append(DataStoreObject(store_path=store_path))

                        if num_workers is not None and num_workers > 1:
                            logging.debug('Using multiprocessing with num_workers: %d.', num_workers)
                            with multiprocessing.Pool(processes=num_workers) as p:
                                result = list(
                                    tqdm(p.imap(datastore_object_get, audio_objects), total=len(audio_objects))
                                )
                        else:
                            logging.debug('Using a single process.')
                            result = []
                            for audio_object in tqdm(audio_objects):
                                result.append(audio_object.get() is not None)

                        if not all(result):
                            raise RuntimeError('Some files not downloaded successfully')
                        logging.info('Caching complete')

                else:
                    # Nothing to do here
                    logging.debug('Manifest is not on a data store: %s', manifest_file)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            logging.debug('Distributed environment is available and initialized.')

            # Handle distributed environment
            if shared_cache is None:
                shared_cache = is_datastore_cache_shared()

            if shared_cache:
                logging.debug('Cache is shared among nodes, cache data on global rank zero.')
                is_rank_zero = is_global_rank_zero()
            else:
                logging.debug('Cache is not shared among nodes, cache data on local rank zero.')
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                is_rank_zero = local_rank == 0

            if is_rank_zero:
                logging.info('Cache data from %s rank 0', 'global' if shared_cache else 'local')
                cache_data(
                    manifest_filepaths=manifest_filepaths,
                    cache_audio=cache_audio,
                    num_workers=num_workers,
                    max_num_workers=max_num_workers,
                )
            logging.debug('Reached barrier')
            torch.distributed.barrier()

        elif is_global_rank_zero():
            # Handle non-distributed environment, e.g., if running on a single GPU
            logging.warning(
                'Torch distributed is not initialized and caching may be prone to data race conditions. '
                'Now caching data from global rank 0. If there are other ranks and they pass this '
                'before rank 0, errors might result.'
            )
            cache_data(
                manifest_filepaths=manifest_filepaths,
                cache_audio=cache_audio,
                num_workers=num_workers,
                max_num_workers=max_num_workers,
            )
        else:
            raise RuntimeError(
                'Torch distributed is not initialized and caching on nodes other than global rank zero is disabled '
                'to avoid race condition between different ranks. To ensure distributed environment is '
                'initialized, please update data config to use `defer_setup = True`.'
            )


"""Optionally expand / shard the list of manifests
    This is made to use the same notation as the sharded audio files

    Args:
        manifest_filepaths: list of manifest files (the sharded notation)
        shard_strategy: scatter or replicate (scatter by default)
        shard_manifests: bool, if False, no sharding / manifest filepath expansion will be attempted
        global_rank: int, the rank of this worker
        world_size: int, total number of workers
"""


def shard_manifests_if_needed(
    manifest_filepaths: Union[str, List[str]],
    shard_strategy: str,
    shard_manifests: bool,
    global_rank: int,
    world_size: int,
):
    if shard_manifests:
        if not torch.distributed.is_available():
            logging.warning("Not running in torch.distributed mode. Manifest sharding not available")
            return manifest_filepaths

        if not torch.distributed.is_initialized():
            logging.warning(
                'Manifest sharding was requested but torch.distributed is not initialized '
                'Did you intend to set the defer_setup flag?'
            )
            return manifest_filepaths

        manifest_filepaths = expand_sharded_filepaths(
            sharded_filepaths=manifest_filepaths,
            shard_strategy=shard_strategy,
            world_size=world_size,
            global_rank=global_rank,
        )

    return manifest_filepaths


class _AVTextDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment loaded
            audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include in dataset
        max_utts: Limit number of utterances
        trim: whether or not to trim silence. Defaults to False
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        pad_id: Id of pad symbol. Defaults to 0
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
            'utterance_id': NeuralType(tuple('B'), LengthsType()),
            'speaker_id': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        manifest_parse_func: Optional[Callable] = None,
        audio_downsampling_factor: int = 1,
        max_training_rand_seg_duration: Optional[int] = None,
        val: bool = False,
    ):
        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(",")

        self.VIDEO_FPS = 25

        # If necessary, cache manifests and audio from object store
        cache_datastore_manifests(manifest_filepaths=manifest_filepath, cache_audio=True)

        self.manifest_processor = ASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            manifest_parse_func=manifest_parse_func,
        )

        if val:
            self.per_spk_collection = []
            # Create new collection by unflattening text_tokens
            for sample in self.manifest_processor.collection:
                # Get unique speakers from text_tokens
                speakers = sorted(list(set(x['speaker'] for x in sample.text_tokens)))
                for s in speakers:
                    self.per_spk_collection.append((sample, s))
            

        self.max_training_rand_seg_duration = max_training_rand_seg_duration
        self.val = val
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.return_sample_id = return_sample_id
        self.channel_selector = channel_selector
        self.audio_downsampling_factor = audio_downsampling_factor

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __getitem__(self, index):
        if isinstance(index, IterableABC):
            return [self._process_sample(_index) for _index in index]
        else:
            return self._process_sample(index)
        
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

    def _process_sample(self, index):
        if self.val:
            sample, spk = self.per_spk_collection[index]
        else:
            sample = self.manifest_processor.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(
            sample.audio_file,
            offset=offset,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr,
            channel_selector=self.channel_selector,
        )
        f, fl = features, torch.tensor(features.shape[0]).long()

        speakers = sorted(list(set(x['speaker'] for x in sample.text_tokens)))
        speakers_idx = {spk: i for i, spk in enumerate(speakers)}
        rand_spk = spk if self.val else random.choice(list(speakers))
        visual_embeds = torch.load(sample.per_spk_feature_files[rand_spk], map_location='cpu')
        if len(visual_embeds.shape) == 2: # Shape: (time, layers, feature_dim)
            visual_embeds = visual_embeds.unsqueeze(1)
        start_idx = int(sample.offset * self.VIDEO_FPS) # Assuming 25 fps
        end_idx = start_idx + int(sample.duration * self.VIDEO_FPS)
        visual_embeds = visual_embeds[start_idx:end_idx, :, :]
        
        speakers_tokens = []
        downsampled_freq = 16000 / self.audio_downsampling_factor
        downsampled_fl_length = fl if fl % self.audio_downsampling_factor == 0 else fl + (self.audio_downsampling_factor - (fl % self.audio_downsampling_factor))
        downsampled_fl_length = int(downsampled_fl_length / self.audio_downsampling_factor)

        spk_activity_mask = torch.zeros((len(speakers), downsampled_fl_length))

        for tt in sample.text_tokens:
            if tt['speaker'] == rand_spk:
                speakers_tokens.extend(tt['text'])
            spk_activity_mask[speakers_idx[tt['speaker']], int(tt['start']*downsampled_freq):int((tt['start'] + tt['duration'])*downsampled_freq)] = 1
        
        stno_mask = self._create_stno_masks(spk_activity_mask, speakers_idx[rand_spk])
        t, tl = self.manifest_processor.process_text_by_sample(speakers_tokens)

        """
        DUMMY subsampling so I can start training ASAP. A proper (more optimal) way of generating random segments will be implemented later.
        The point is to see if the sample duration is above the required sample duration. 
        If so, then we need to sample some random {max_lemgth} chunk and select all the words from the target speaker that fall into that chunk.
        We don't have any word-level alignment, so we need to randomly select some starting segment and then take all the segments within (seg_start, seg_start + max_length). STNO and other tensors can be generated accordingly, same goes for the audio.
        """
        if self.max_training_rand_seg_duration is not None and not self.val and \
                sample.duration > self.max_training_rand_seg_duration:

            segment_start = random.uniform(0, sample.duration - self.max_training_rand_seg_duration)
            segment_end = segment_start + self.max_training_rand_seg_duration

            visual_embeds = visual_embeds[int(segment_start*self.VIDEO_FPS):int(segment_end*self.VIDEO_FPS), ...]
            # Temporary assert making sure we're not using any feat extractor apart from loading a raw waveform.
            assert fl.item() / sample.duration - 16000 < 1
            f = f[int(segment_start*16000):int(segment_end*16000)]
            fl = torch.tensor(f.shape[0]).long()
            stno_sr = 12.5
            stno_mask = stno_mask[:, int(segment_start*stno_sr):int(segment_end*stno_sr)]

            # Now, we need to select text tokens
            tt = filter(lambda x: x['speaker'] == rand_spk and x['start'] >= segment_start and x['start'] + x['duration'] <= segment_end, sample.text_tokens)
            t = reduce(lambda a,b: a + b['text'], tt, [])
            tl = len(t)

        if self.return_sample_id:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), stno_mask, torch.tensor(stno_mask.shape[-1]).long(), sample.id, speakers_idx[rand_spk], visual_embeds, torch.tensor(visual_embeds.shape[0]), index
        else:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), stno_mask, torch.tensor(stno_mask.shape[-1]).long(), sample.id, speakers_idx[rand_spk], visual_embeds, torch.tensor(visual_embeds.shape[0])

        return output

    def __len__(self):
        if self.val:
            return len(self.per_spk_collection)
        return len(self.manifest_processor.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=self.manifest_processor.pad_id)


class AVToBPEAndSTNODataset(_AVTextDataset):
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
    ):
        if use_start_end_token and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

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

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            trim=trim,
            return_sample_id=return_sample_id,
            channel_selector=channel_selector,
            manifest_parse_func=manifest_parse_func,
            audio_downsampling_factor=audio_downsampling_factor,
            max_training_rand_seg_duration=max_training_rand_seg_duration,
            val=val,
        )
