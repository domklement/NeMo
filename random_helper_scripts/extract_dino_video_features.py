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
import torch.distributed as dist
import torch.nn as nn
from lhotse import CutSet, load_manifest
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
from transformers import AutoModel

from nemo.collections.asr.modules.dinov3 import DINOv3VRSEncoder
from nemo.utils import logging

torch.set_float32_matmul_precision('high')


class VideoFeatureDataset(Dataset):
    """Indexable Dataset for extracting video features from a Lhotse CutSet.

    This dataset builds a flat index of (cut, speaker) pairs across the provided
    CutSet and applies contiguous sharding over that flattened list.
    The shard range is computed over the total number of (cut, speaker) items.
    """

    def __init__(
        self,
        cutset: CutSet,
        video_dir: str,
        video_transform,
        num_shards: int = 1,
        shard_idx: int = 0,
    ):
        self.video_dir = Path(video_dir)
        self.video_transform = video_transform

        # Build global flat list of items: (cut, speaker)
        all_items = []
        cuts_list = list(cutset)
        for cut in cuts_list:
            try:
                speakers = sorted(CutSet.from_cuts([cut]).speakers)
            except Exception:
                speakers = []

            for speaker in speakers:
                # Ensure the speaker has an associated video in the cut metadata
                if hasattr(cut, 'per_spk_lip_crop_videos') and speaker in getattr(cut, 'per_spk_lip_crop_videos'):
                    all_items.append((cut, speaker))

        self.total_items = len(all_items)

        # Compute contiguous shard ranges over the flattened list
        num_shards = max(1, int(num_shards))
        shard_idx = min(max(0, int(shard_idx)), num_shards - 1)

        base = self.total_items // num_shards
        rem = self.total_items % num_shards
        # Distribute remainder across the first `rem` shards
        start = shard_idx * base + min(shard_idx, rem)
        end = start + base + (1 if shard_idx < rem else 0)

        self.items = all_items[start:end]
        logging.info(
            f"Shard {shard_idx}/{num_shards}: Global items={self.total_items}, "
            f"range=[{start}:{end}), local items={len(self.items)}"
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        cut, speaker = self.items[index]

        # Resolve speaker-specific video path
        if not hasattr(cut, 'per_spk_lip_crop_videos') or speaker not in cut.per_spk_lip_crop_videos:
            raise KeyError(f"No video found for speaker {speaker} in cut {getattr(cut, 'id', 'unknown')}")

        video_path = cut.per_spk_lip_crop_videos[speaker]
        video_path_abs = Path(video_path)
        try:
            relative_path = video_path_abs.relative_to(self.video_dir)
        except ValueError:
            relative_path = video_path_abs

        # Load video frames using torchcodec with NCHW dimension order
        vid_dec = VideoDecoder(video_path, dimension_order="NCHW")
        video_frames = vid_dec[:]  # [T, C, H, W]

        # Apply video transform
        video_frames = self.video_transform(video_frames, return_tensors="pt")[
            'pixel_values'
        ]

        return {
            'cut_id': getattr(cut, 'id', ''),
            'speaker': speaker,
            'video_path': str(video_path),
            'relative_path': str(relative_path),
            'video_frames': video_frames,  # [T, C, H, W]
        }


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


def _get_rank_world_size():
    """Best-effort detection of `rank`, `local_rank`, and `world_size` across torchrun, SLURM, and MPI.

    Returns:
        rank (int): Global rank in [0, world_size-1]
        local_rank (int): Local rank on the current node/GPU
        world_size (int): Total number of ranks
    """
    # Defaults
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', os.environ.get('OMPI_COMM_WORLD_RANK', 0))))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))))
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', os.environ.get('OMPI_COMM_WORLD_SIZE', 1))))

    # Fallback to CUDA device count when launching single-node multi-GPU without env (rare)
    if world_size == 1 and torch.cuda.is_available():
        try:
            # If torchrun wasn't used, attempt to infer from device count; keep world_size=1 to avoid mis-sharding
            pass
        except Exception:
            pass

    return rank, local_rank, world_size


def main(args):
    # Setup DDP and device mapping
    pl.seed_everything(42)
    
    # Determine ranks and device
    rank, local_rank, world_size = _get_rank_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    logging.info(f"Using device: {device} (RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size})")
    
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
        # Across-node sharding only; per-rank distribution is handled by DistributedSampler
        num_shards=args.num_shards,
        shard_idx=args.shard_idx,
    )
    
    # Create DistributedSampler for per-rank data distribution
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    
    # Create dataloader with multiple workers (no batching, batch_size=1)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Setup output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup model and move to device immediately
    model = DINOFeatureExtractor(model_name=args.model_name)
    model = model.to(device)
    
    # Detect available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Setup Lightning Trainer with DDP
    trainer = pl.Trainer(
        accelerator='gpu' if num_gpus > 0 else 'cpu',
        devices=num_gpus if num_gpus > 0 else 1,
        strategy='ddp' if num_gpus > 1 else 'auto',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        max_epochs=1,
    )
    
    # Pass device to extraction function
    return _extract_features_distributed(
        trainer=trainer,
        model=model,
        dataloader=dataloader,
        output_dir=output_dir,
        device=device,
        args=args,
    )


def _extract_features_distributed(trainer, model, dataloader, output_dir, device, args):
    """Extract features using distributed inference.
    
    Args:
        trainer: PyTorch Lightning Trainer instance
        model: DINOFeatureExtractor model (already on device)
        dataloader: DataLoader for the dataset
        output_dir: Output directory for saving features
        device: Torch device to use for processing
        args: Command-line arguments
    """
    
    # Get rank and world size from environment
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Try initializing torch.distributed so we can aggregate progress across ranks
    try:
        if world_size > 1 and dist.is_available() and not dist.is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend, init_method='env://')
    except Exception:
        pass
    
    # Ensure model is on device and in eval mode
    model = model.to(device)
    model.eval()
    
    num_processed_local = 0
    update_every = 1

    # Iterate over local shard
    for sample in tqdm(dataloader, disable=(rank != 0), desc=f"Rank {rank} processing"):
        try:
            # sample is a dict, but wrapped in a list when batch_size=1
            if isinstance(sample, list) and len(sample) == 1 and isinstance(sample[0], dict):
                sample = sample[0]

            cut_id = sample['cut_id'][0] if isinstance(sample['cut_id'], list) else sample['cut_id']
            speaker = sample['speaker'][0] if isinstance(sample['speaker'], list) else sample['speaker']
            relative_path = sample['relative_path'][0] if isinstance(sample['relative_path'], list) else sample['relative_path']
            video_frames = sample['video_frames'][0]
            video_frames = video_frames.to(device, dtype=torch.float32)  # [T, C, H, W]

            # Extract features: [T, num_tokens, embed_dim]
            with torch.no_grad():
                # Ensure video_frames is on correct device before passing to model
                video_frames = video_frames.to(device)
                features = model(video_frames)

            # Save features
            _save_single_feature(output_dir, relative_path, speaker, features.cpu())

            num_processed_local += 1

        except Exception as e:
            logging.warning(f"Failed to process video: {e}")
            continue

    # Aggregate processed counts across ranks
    global_processed = num_processed_local
    if dist.is_available() and dist.is_initialized():
        try:
            t = torch.tensor([num_processed_local], dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            global_processed = int(t.item())
        except Exception:
            pass

    if rank == 0:
        logging.info(
            f"Extraction complete. Processed {global_processed} videos across {world_size} rank(s) on device {device}"
        )


def _save_single_feature(output_dir: Path, relative_path: str, speaker: str, features: torch.Tensor):
    """Save extracted features preserving directory structure with video name and speaker subdirectory."""
    # Convert video path to feature path
    relative_path = Path(relative_path)
    
    # Get video name without extension
    video_name = relative_path.stem
    
    # Create path: output_dir/relative_path_parent/video_name/speaker_X.pt
    feature_dir = output_dir / relative_path.parent / video_name
    feature_path = feature_dir / f"speaker_{speaker}.pt"
    
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
