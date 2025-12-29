#!/usr/bin/env python3
"""
Add DINO v3 feature paths to cutset custom metadata.

This script loads a cutset and populates cut.custom['dinov3_b_lip_features'] with
a dictionary mapping speaker IDs to their corresponding DINO feature file paths.

The script expects features to be organized as:
    {features_dir}/{relative_video_path_parent}/{video_name}/speaker_{speaker_id}.pt

Example usage:
    python add_dinov3_feature_paths_to_cuts.py \
        --cutset_path /path/to/cuts.jsonl.gz \
        --features_dir /path/to/dino_feats \
        --video_dir /path/to/videos \
        --output_path /path/to/output_cuts.jsonl.gz \
        --feature_key dinov3_b_lip_features \
        --video_key per_spk_lip_crop_videos
"""

import argparse
import os
from pathlib import Path

from tqdm import tqdm
from lhotse import load_manifest, CutSet
from nemo.utils import logging


def find_feature_path(features_dir: Path, video_path: str, video_dir: Path, speaker: str) -> str:
    """
    Find the feature path for a given video and speaker.
    
    Args:
        features_dir: Base directory containing extracted features
        video_path: Absolute path to the video file
        video_dir: Base directory for videos (to compute relative paths)
        speaker: Speaker ID
        
    Returns:
        Absolute path to the feature file as string
    """
    video_path_abs = Path(video_path)
    
    # Compute relative path
    try:
        relative_path = video_path_abs.relative_to(video_dir)
    except ValueError:
        # If video_path is not relative to video_dir, use the path as-is
        relative_path = video_path_abs
    
    # Build feature path: features_dir/relative_path_parent/video_name/speaker_{speaker}.pt
    video_name = relative_path.stem
    feature_path = features_dir / relative_path.parent / video_name / f"speaker_{speaker}.pt"
    
    return str(feature_path)


def main():
    parser = argparse.ArgumentParser(description="Add DINO feature paths to cutset")
    parser.add_argument(
        "--cutset_path",
        type=str,
        required=True,
        help="Path to input cutset manifest (JSONL or JSONL.GZ)"
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="Base directory containing extracted DINO features"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output cutset manifest with updated feature paths"
    )
    parser.add_argument(
        "--feature_key",
        type=str,
        default="dinov3_b_lip_features",
        help="Key name for feature paths in cut.custom (default: dinov3_b_lip_features)"
    )
    parser.add_argument(
        "--video_key",
        type=str,
        default="per_spk_lip_crop_videos",
        help="Key name for per-speaker video paths in cut.custom (e.g., per_spk_lip_crop_videos)"
    )
    parser.add_argument(
        "--verify_existence",
        action="store_true",
        help="Verify that feature files exist (slower but safer)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    features_dir = Path(args.features_dir)
    
    # Load cutset
    logging.info(f"Loading cutset from {args.cutset_path}")
    cutset = load_manifest(args.cutset_path)
    total_cuts = len(cutset)
    
    logging.info(f"Processing {total_cuts} cuts")
    
    # Track statistics
    num_updated = 0
    num_speakers_updated = 0
    num_missing = 0
    
    updated_cuts = []
    
    for cut in tqdm(cutset, desc="Processing cuts"):        
        speakers = CutSet.from_cuts([cut]).speakers
        
        # Build feature path mapping
        per_spk_features = {}
        for speaker in speakers:
            vid_path = cut.custom[args.video_key].get(speaker)
            if vid_path is None:
                raise ValueError(f"Video path for speaker {speaker} not found in cut.custom[{args.video_key}] for cut {cut.id}")

            feature_path = os.path.join(
                features_dir,
                os.path.basename(vid_path).split('.')[0],
                f"speaker_{speaker}.pt"
            )
            
            # Optionally verify existence
            if args.verify_existence and not Path(feature_path).exists():
                raise FileNotFoundError(f"Feature file not found: {feature_path} for speaker {speaker} in cut {cut.id}")
            
            per_spk_features[speaker] = feature_path
            num_speakers_updated += 1

        assert len(per_spk_features) == len(speakers), \
            f"Number of feature paths ({len(per_spk_features)}) does not match number of speakers ({len(speakers)}) in cut {cut.id}"
        
        cut.custom[args.feature_key] = per_spk_features
        updated_cuts.append(cut)
    
    # Create updated cutset
    updated_cutset = CutSet.from_cuts(updated_cuts)
    
    # Save updated cutset
    logging.info(f"Saving updated cutset to {args.output_path}")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    updated_cutset.to_file(args.output_path)
    
    # Print statistics
    logging.info("=" * 60)
    logging.info(f"Processing complete!")
    logging.info(f"Total cuts: {total_cuts}")


if __name__ == "__main__":
    main()
