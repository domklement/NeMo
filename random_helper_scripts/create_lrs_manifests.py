#!/usr/bin/env python3
"""
Script to create Lhotse CutSets from LRS2 dataset directory structure.

The script scans through directories in the LRS2 root directory and processes
video files along with their corresponding labels and sample IDs to create
Lhotse CutSets for each dataset partition.
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from lhotse import AudioSource, Recording, SupervisionSegment
from lhotse.audio.utils import VideoInfo
from lhotse.cut import MonoCut, CutSet
from torchcodec.decoders import AudioDecoder, VideoDecoder


def find_video_files(dataset_dir: Path) -> List[Path]:
    """
    Find all .video files in the given directory.

    Args:
        dataset_dir: Path to the dataset directory

    Returns:
        List of paths to .video files
    """
    video_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.video'):
                video_files.append(Path(root) / file)
    return video_files


def read_label_file(label_path: Path) -> str:
    """
    Read the transcript from a .label file.

    Args:
        label_path: Path to the .label file

    Returns:
        Transcript text
    """
    with open(label_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def read_sample_id_file(sample_id_path: Path) -> str:
    """
    Read the sample ID from a .sample_id file.

    Args:
        sample_id_path: Path to the .sample_id file

    Returns:
        Sample ID string (format: "main/speaker_id/session_id")
    """
    with open(sample_id_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def create_cut_from_video(video_path: Path, base_name: str, dataset_part: str) -> Optional[MonoCut]:
    """
    Create a Lhotse MonoCut from a video file and its associated metadata.

    Args:
        video_path: Path to the .video file
        base_name: Base name of the file (without extension)
        dataset_part: Name of the dataset partition

    Returns:
        MonoCut object or None if processing fails
    """
    try:
        # Construct paths to associated files
        label_path = video_path.with_suffix('.label')
        sample_id_path = video_path.with_suffix('.sample_id')

        # Read metadata
        transcript = ""
        sample_id = ""

        if label_path.exists():
            transcript = read_label_file(label_path)
        else:
            raise FileNotFoundError(f"Label file not found for {video_path}")

        if sample_id_path.exists():
            sample_id = read_sample_id_file(sample_id_path)
        else:
            raise FileNotFoundError(f"Sample ID file not found for {video_path}")

        # Extract speaker ID from sample_id (format: "main/speaker_id/session_id")
        speaker_id = None
        if sample_id:
            parts = sample_id.split('/')
            if len(parts) >= 2:
                speaker_id = parts[1]

        if speaker_id is None and 'portrait_face' in sample_id:
            speaker_id = sample_id.split('_portrait_face_')[0]

        if speaker_id is None:
            raise ValueError(f"Invalid sample ID format: {sample_id}")

        # Create a unique ID for the cut
        cut_id = f"{dataset_part}_{sample_id.replace('/', '_')}" if sample_id else f"{dataset_part}_{base_name}"
        
        adec = AudioDecoder(video_path)
        vdec = VideoDecoder(video_path)

        audio_samples = adec.get_all_samples()

        # Create Recording
        # Note: For video files, we use the video path as source
        # Lhotse can handle video files and extract audio
        recording = Recording(
            id=cut_id,
            sources=[
                AudioSource(
                    type='file',
                    channels=[0],
                    source=str(video_path),
                    video=VideoInfo(
                        fps=int(round(vdec.metadata.average_fps)),
                        num_frames=vdec.metadata.num_frames,
                        height=vdec.metadata.height,
                        width=vdec.metadata.width
                    ),
                )
            ],
            sampling_rate=adec.metadata.sample_rate,  # LRS2 typical sampling rate
            num_samples=audio_samples.data.shape[-1],  # Will be populated when audio is loaded
            duration=audio_samples.duration_seconds  # Will be populated when audio is loaded
        )

        # Create Supervision
        supervision = SupervisionSegment(
            id=cut_id,
            recording_id=cut_id,
            start=0.0,
            duration=audio_samples.duration_seconds,  # Will be populated when audio is loaded
            channel=0,
            text=transcript,
            language='en',
            speaker=speaker_id,
        )

        # Create MonoCut
        cut = MonoCut(
            id=cut_id,
            start=0.0,
            duration=audio_samples.duration_seconds,  # Will be populated when audio is loaded
            channel=0,
            supervisions=[supervision],
            recording=recording,
            custom={'dataset_part': dataset_part, 'sample_id': sample_id, 'per_spk_lip_crop_videos': {speaker_id: str(video_path)}}
        )

        return cut
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None


def process_single_video(video_path: Path, part_name: str) -> Optional[MonoCut]:
    """
    Process a single video file. This function is designed to be called in parallel.

    Args:
        video_path: Path to the video file
        part_name: Name of the dataset partition

    Returns:
        MonoCut object or None if processing fails
    """
    base_name = video_path.stem
    return create_cut_from_video(video_path, base_name, part_name)


def process_dataset_part(dataset_dir: Path, part_name: str, num_workers: int = 1) -> CutSet:
    """
    Process a single dataset partition and create a CutSet.

    Args:
        dataset_dir: Path to the dataset partition directory
        part_name: Name of the partition (e.g., 'train', 'val', 'test')
        num_workers: Number of parallel workers to use (default: 1 for sequential processing)

    Returns:
        CutSet containing all cuts for this partition
    """
    print(f"\nProcessing dataset part: {part_name}")

    # Find all video files
    video_files = find_video_files(dataset_dir)
    print(f"Found {len(video_files)} video files")

    # Create cuts
    cuts = []
    
    if num_workers == 1:
        # Sequential processing
        for video_path in tqdm(video_files, desc=f"Creating cuts for {part_name}"):
            cut = process_single_video(video_path, part_name)
            if cut is not None:
                cuts.append(cut)
    else:
        # Parallel processing
        print(f"Using {num_workers} worker processes")
        process_func = partial(process_single_video, part_name=part_name)
        
        with Pool(processes=num_workers) as pool:
            # Use imap_unordered for better memory efficiency with large datasets
            results = list(tqdm(
                pool.imap_unordered(process_func, video_files),
                total=len(video_files),
                desc=f"Creating cuts for {part_name}"
            ))
        
        # Filter out None values (failed processing)
        cuts = [cut for cut in results if cut is not None]

    print(f"Successfully processed {len(cuts)} out of {len(video_files)} video files")

    # Create CutSet
    cutset = CutSet.from_cuts(cuts)
    print(f"Created CutSet with {len(cutset)} cuts")

    return cutset


def main():
    """
    Main function to process LRS2 dataset and create Lhotse CutSets.
    """
    parser = argparse.ArgumentParser(
        description="Create Lhotse CutSets from LRS2 dataset directory structure"
    )
    parser.add_argument(
        '--lrs2_dir',
        type=Path,
        help='Path to the LRS2 root directory containing dataset partitions'
    )
    parser.add_argument(
        '--output_manifest_dir',
        type=Path,
        help='Directory where the output CutSet manifests will be stored'
    )
    parser.add_argument(
        '--process_parts',
        type=str,
        nargs='+',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel worker processes to use (default: 1 for sequential processing, '
             'use -1 to use all available CPU cores)'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip processing if the output manifest file already exists'
    )

    args = parser.parse_args()

    # Determine number of workers
    if args.num_workers == -1:
        args.num_workers = cpu_count()
        print(f"Using all available CPU cores: {args.num_workers}")
    elif args.num_workers < 1:
        raise ValueError(f"Number of workers must be >= 1 or -1 for all cores, got {args.num_workers}")
    elif args.num_workers > 1:
        print(f"Using {args.num_workers} worker processes")

    # Validate input directory
    if not args.lrs2_dir.exists():
        raise ValueError(f"LRS2 directory does not exist: {args.lrs2_dir}")

    if not args.lrs2_dir.is_dir():
        raise ValueError(f"LRS2 path is not a directory: {args.lrs2_dir}")

    # Create output directory if it doesn't exist
    args.output_manifest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output manifests will be saved to: {args.output_manifest_dir}")

    # Scan for subdirectories in lrs2_dir
    subdirs = [d for d in args.lrs2_dir.iterdir() if d.is_dir()]

    if not subdirs:
        raise ValueError(f"No subdirectories found in {args.lrs2_dir}")

    print(f"Found {len(subdirs)} dataset partitions: {[d.name for d in subdirs]}")

    # Process each dataset partition
    for subdir in subdirs:
        part_name = subdir.name

        # Check if manifest already exists and skip if requested
        output_filename = f"{part_name}_cuts.jsonl.gz"
        output_path = args.output_manifest_dir / output_filename

        if args.skip_existing and output_path.exists():
            print(f"\nSkipping {part_name}: Manifest already exists at {output_path}")
            continue

        # Process the partition with parallel processing
        cutset = process_dataset_part(subdir, part_name, num_workers=args.num_workers)

        # Save the CutSet
        print(f"Saving CutSet to: {output_path}")
        cutset.to_file(output_path)
        print(f"Successfully saved {part_name} CutSet")

    print("\n" + "="*50)
    print("All dataset partitions processed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
