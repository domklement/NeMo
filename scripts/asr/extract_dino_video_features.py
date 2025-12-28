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

"""
Extract last hidden state from DINO model for video cutsets using DDP and multiple GPUs.
Features are extracted as: Time x TokenSequence x EmbedDim

Example usage:
python extract_dino_video_features.py \
    --cutset_path=/path/to/cutset.jsonl \
    --video_dir=/path/to/videos \
    --output_path=/path/to/output \
    --num_shards=4 \
    --shard_idx=0 \
    --model_name=dinov3-vith16plus-pretrain-lvd1689m \
    --num_workers=4
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from lhotse import CutSet, load_manifest
from torch.utils.data import DataLoader, IterableDataset
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
from transformers import AutoModel

from nemo.collections.asr.modules.dinov3 import DINOv3VRSEncoder
from nemo.utils import logging

torch.set_float32_matmul_precision('high')


class VideoFeatureDataset(IterableDataset):
    """Dataset for extracting video features from Lhotse CutSet."""

    def __init__(self, cutset: CutSet, video_dir: str, video_transform, num_shards: int = 1, shard_idx: int = 0):
        self.cutset = cutset
        self.video_dir = Path(video_dir)
        self.video_transform = video_transform
        self.num_shards = num_shards
        self.shard_idx = shard_idx
        
        # Distribute cuts across shards
        cuts_list = list(cutset)
        self.cuts = cuts_list[shard_idx::num_shards]
        logging.info(f"Shard {shard_idx}/{num_shards}: Processing {len(self.cuts)} cuts")

    def __iter__(self):
        for cut_idx, cut in enumerate(self.cuts):
            try:
                # Get all speakers' videos
                speakers = sorted(cut.speakers)
                
                for spk_idx, speaker in enumerate(speakers):
                    # Access per_spk_lip_crops for this speaker
                    if hasattr(cut, 'per_spk_lip_crops') and speaker in cut.per_spk_lip_crops:
                        video_path = cut.per_spk_lip_crops[speaker]
                    else:
                        logging.warning(f"No video found for speaker {speaker} in cut {cut.id}")
                        continue

                    # Compute relative path from video_dir
                    video_path_abs = Path(video_path)
                    try:
                        relative_path = video_path_abs.relative_to(self.video_dir)
                    except ValueError:
                        # If not relative, use the full path structure
                        relative_path = video_path_abs

                    # Load video frames using torchcodec with NCHW dimension order
                    try:
                        vid_dec = VideoDecoder(video_path, dimension_order="NCHW")
                        video_frames = vid_dec[:]  # [T, C, H, W]
                        
                    except Exception as e:
                        logging.warning(f"Failed to load video {video_path}: {e}")
                        continue

                    # Apply video transform (same as in av_to_text_and_stno_lhotse)
                    # video_transform expects [T, C, H, W] and returns processed frames
                    try:
                        video_frames = self.video_transform(video_frames, return_tensors="pt")['pixel_values']
                    except Exception as e:
                        logging.warning(f"Failed to transform video {video_path}: {e}")
                        continue
                    
                    if video_frames.numel() > 0:
                        # video_frames shape: [T, C, H, W]
                        yield {
                            'cut_id': cut.id,
                            'speaker': speaker,
                            'video_path': str(video_path),
                            'relative_path': str(relative_path),
                            'video_frames': video_frames,  # [T, C, H, W]
                        }
            except Exception as e:
                logging.warning(f"Error processing cut {cut.id}: {e}")
                continue


class DINOFeatureExtractor(pl.LightningModule):
    """Lightning module for DINO feature extraction using raw DINO model."""

    def __init__(self, model_name: str = "dinov3-vith16plus-pretrain-lvd1689m"):
        super().__init__()
        # Load the base DINO model
        self.dino_model = AutoModel.from_pretrained(f"facebook/{model_name}")
        self.dino_model.eval()
        
        # Freeze the model
        for param in self.dino_model.parameters():
            param.requires_grad = False

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video frames.
        
        Args:
            video_frames: [T, C, H, W] video frames (batch dim = time)
            
        Returns:
            features: [T, num_tokens, embed_dim] entire last hidden state (all tokens including CLS)
        """
        with torch.no_grad():
            output = self.dino_model(pixel_values=video_frames)
            # output.last_hidden_state: [T, num_tokens, embed_dim]
            # Return entire last hidden state (CLS + all patch tokens)
            features = output.last_hidden_state
        
        return features

    def configure_optimizers(self):
        return None


def main(args):
    # Setup DDP
    pl.seed_everything(42)
    
    # Load cutset
    logging.info(f"Loading cutset from {args.cutset_path}")
    cutset = load_manifest(args.cutset_path)
    
    # Create video transform using DINOv3VRSEncoder
    video_transform = DINOv3VRSEncoder.get_image_processor(args.model_name)
    
    # Create dataset
    dataset = VideoFeatureDataset(
        cutset=cutset,
        video_dir=args.video_dir,
        video_transform=video_transform,
        num_shards=args.num_shards,
        shard_idx=args.shard_idx,
    )
    
    # Create dataloader with multiple workers (no batching, batch_size=1)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,  # No batching
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Setup output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup model
    model = DINOFeatureExtractor(model_name=args.model_name)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Extract features
    num_processed = 0
    
    for sample in tqdm(dataloader, desc=f"Extracting features (shard {args.shard_idx})"):
        if sample is None:
            continue
        
        try:
            cut_id = sample['cut_id']
            speaker = sample['speaker']
            video_path = sample['video_path']
            relative_path = sample['relative_path']
            video_frames = sample['video_frames'].to(device)  # [T, C, H, W]
            
            # Extract features: [T, num_tokens, embed_dim]
            with torch.no_grad():
                features = model(video_frames)
            
            # Save features with directory structure
            _save_single_feature(output_dir, relative_path, features.cpu())
            
            num_processed += 1
                
        except Exception as e:
            logging.warning(f"Failed to process video: {e}")
            continue
    
    logging.info(f"Extraction complete. Processed {num_processed} videos")


def _save_single_feature(output_dir: Path, relative_path: str, features: torch.Tensor):
    """Save extracted features preserving directory structure."""
    # Convert video path to feature path (.pt)
    relative_path = Path(relative_path)
    feature_path = output_dir / relative_path.with_suffix('.pt')
    
    # Create parent directories
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save features as .pt file (torch tensor)
    # Shape: [T, num_tokens, embed_dim]
    torch.save(features, feature_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DINO video features from Lhotse cutset")
    parser.add_argument(
        "--cutset_path",
        type=str,
        required=True,
        help="Path to Lhotse cutset manifest (JSONL)"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Base directory for videos (used to compute relative paths)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for extracted features"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards for data distribution"
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Index of current shard (0-indexed)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="dinov3-vith16plus-pretrain-lvd1689m",
        help="DINO model name from HuggingFace"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading"
    )
    
    args = parser.parse_args()
    main(args)
