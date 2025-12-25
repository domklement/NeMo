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
import json
import random
from math import isclose
from typing import Any, List, Optional, Union

import numpy as np
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.listconfig import ListConfig
from torch.utils.data import ChainDataset

from nemo.collections.asr.data import audio_to_text_dali, audio_to_text_and_stno, av_to_text_and_stno
from nemo.collections.asr.data.huggingface.hf_audio_to_text_dataset import (
    get_hf_audio_to_text_bpe_dataset,
    get_hf_audio_to_text_char_dataset,
)
from nemo.collections.asr.parts.preprocessing.perturb import AudioAugmentor, process_augmentations
from nemo.collections.common.data.dataset import CodeSwitchedDataset, ConcatDataset
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.utils import logging
from nemo.collections.asr.data.av_to_text_and_stno_lhotse import LhotseAVToBPEAndSTNODataset

def get_av_to_text_and_stno_lhotse_dataset(
    cfg: DictConfig,
    tokenizer: Optional[TokenizerSpec] = None,
    global_rank: int = 0,
    world_size: int = 1,
    augmentor: Optional[AudioAugmentor] = None,
    shuffle: bool = True,
    audio_transform: Optional[callable] = None,
    video_transform: Optional[callable] = None,
    video_transform_type: Optional[str] = None,
    **kwargs: Any,
) -> Union[LhotseAVToBPEAndSTNODataset, ConcatDataset, CodeSwitchedDataset]:
    return LhotseAVToBPEAndSTNODataset(
        manifest_filepath=cfg.manifest_filepath,
        tokenizer=tokenizer,
        max_training_rand_seg_duration=cfg.get("max_training_rand_seg_duration", None),
        channel_selector=cfg.get("channel_selector", None),
        trim=cfg.get("trim", False),
        return_audio=cfg.get("return_audio", True),
        return_stno=cfg.get("return_stno", True),
        return_visual_features=cfg.get("return_visual_features", True),
        return_video=cfg.get("return_video", False),
        visual_features_key=cfg.get("visual_features_key", "av_hubert_lip_features"),
        video_key=cfg.get("video_key", "per_spk_lip_crop_videos"),
        use_asd_for_stno=cfg.get("use_asd_for_stno", False),
        randomly_mask_audio_signal=cfg.get("randomly_mask_audio_signal", False),
        max_random_audio_mask_span_seconds=cfg.get("max_random_audio_mask_span_seconds", 0.0),
        max_random_audio_mask_ratio=cfg.get("max_random_audio_mask_ratio", 0.3),
        replace_path_prefixes=cfg.get("replace_path_prefixes", None),
        replace_path_replacements=cfg.get("replace_path_replacements", None),
        return_all_spks=cfg.get("return_all_spks", False),
        audio_transform=audio_transform,
        video_transform=video_transform,
        video_transform_type=video_transform_type,
        **kwargs,
    )