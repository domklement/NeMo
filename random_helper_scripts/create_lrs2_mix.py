#!/usr/bin/env python3
"""
Create k-speaker mixed cuts from an LRS2 lhotse cutset.

This script takes a cutset where each cut has one supervision and creates
random k-speaker mixtures with different offsets. The script ensures that
at least k speakers are overlapped at the same time in each resulting MixedCut.
"""
import argparse
import random
from pathlib import Path
from typing import List

from typing import List, Optional, Tuple

from lhotse import CutSet, load_manifest
from lhotse.cut import Cut, MixedCut, MixTrack
from tqdm import tqdm


def create_mixed_cut_with_overlap(
    cuts: List[Cut],
    snr_range: Optional[Tuple[float, float]] = None,
    min_overlap: float = 1.0,
) -> MixedCut:
    """
    Create a MixedCut from k cuts ensuring they overlap at some point.
    
    Args:
        cuts: List of k cuts to mix together
        snr_range: Optional tuple of (min_snr, max_snr) in dB for random SNR sampling.
                   If None, no SNR adjustment is applied. The first track (reference) always has SNR=None.
        min_overlap: Minimum overlap duration in seconds (default: 1.0)
        
    Returns:
        A MixedCut with k tracks that have guaranteed overlap
    """
    # Sort cuts by duration to use the longest as reference
    sorted_cuts = sorted(cuts, key=lambda c: c.duration, reverse=True)
    
    # First track starts at offset 0 (reference track)
    reference_cut = sorted_cuts[0]
    tracks = [MixTrack(cut=reference_cut, offset=0.0)]
    
    # For remaining tracks, choose random offsets that ensure overlap
    # We want all k speakers to overlap, so we need to find a region where
    # all can be present simultaneously
    
    # Calculate the maximum possible overlap region
    # The overlap region is where all k cuts can be present at the same time
    min_cut_duration = min(c.duration for c in sorted_cuts)
    
    # For k speakers to overlap, we need the last speaker to start before
    # (reference_duration - min_cut_duration) to ensure all overlap
    max_start_time = max(0, reference_cut.duration - min_cut_duration)
    
    # Create tracks with offsets ensuring overlap
    for i, cut in enumerate(sorted_cuts[1:], start=1):
        # Choose offset such that this cut overlaps with the reference
        # and potentially with all other cuts
        if max_start_time > 0:
            # Random offset that ensures the cut fits within reference duration
            max_offset = min(max_start_time, reference_cut.duration - min_overlap)
            offset = random.uniform(0, max_offset)
        else:
            offset = 0.0
        
        # Sample random SNR if snr_range is provided
        if snr_range is not None:
            snr = random.uniform(snr_range[0], snr_range[1])
        else:
            snr = None
        
        tracks.append(MixTrack(cut=cut, offset=offset, snr=snr))
    
    # Create the MixedCut
    mixed_cut = MixedCut(
        id=f"mix_{reference_cut.id}_{len(cuts)}spk",
        tracks=tracks
    )
    
    # Add per_spk_lip_crop_videos custom field
    per_spk_lip_crop_videos = {}
    for cut in cuts:
        # # Get the video key from each cut's custom field
        # if hasattr(cut, 'custom') and cut.custom and 'per_spk_lip_crop_videos' in cut.custom:
        #     # Merge all speaker video paths from all cuts
        #     per_spk_lip_crop_videos.update(cut.custom['per_spk_lip_crop_videos'])
        per_spk_lip_crop_videos[cut.supervisions[0].speaker] = cut.recording.sources[0].source
    
    # Add to mixed_cut custom field
    if not hasattr(mixed_cut, 'custom') or mixed_cut.custom is None:
        mixed_cut.custom = {}
    mixed_cut.custom['per_spk_lip_crop_videos'] = per_spk_lip_crop_videos
    
    return mixed_cut


def create_k_speaker_mixtures(
    input_cutset_paths: List[str],
    output_path: str,
    k: int,
    num_mixtures: int = None,
    seed: int = 42,
    distinct_speakers: bool = True,
    snr_range: Optional[Tuple[float, float]] = None,
    min_duration: float = 5.0,
    min_overlap: float = 1.0,
):
    """
    Create k-speaker mixtures from a source cutset.
    
    Args:
        input_cutset_paths: List of paths to source cutset manifests (will be concatenated)
        output_path: Path where to save the output mixed cutset (including filename)
        k: Maximum number of overlapped speakers at one time
        num_mixtures: Number of mixtures to create (default: len(cutset) // k)
        seed: Random seed for reproducibility
        distinct_speakers: If True, ensure all k cuts have different speakers (default: True)
        snr_range: Optional tuple of (min_snr, max_snr) in dB. If provided, each non-reference
                   track will be mixed with a random SNR from this range.
        min_duration: Minimum duration in seconds for cuts to be considered (default: 5.0)
        min_overlap: Minimum overlap duration in seconds (default: 1.0)
    """
    # Load and concatenate multiple cutsets
    print(f"Loading {len(input_cutset_paths)} cutset(s)...")
    all_cuts = []
    for path in input_cutset_paths:
        print(f"  Loading: {path}")
        cuts = load_manifest(path)
        # if hasattr(cuts, 'to_eager'):
        #     cuts = cuts.to_eager()
        all_cuts.append(cuts)
    
    # Concatenate all cutsets
    if len(all_cuts) > 1:
        print("Concatenating cutsets...")
        cuts = all_cuts[0]
        for cutset in all_cuts[1:]:
            cuts = cuts + cutset
    else:
        cuts = all_cuts[0]
    
    # Filter by minimum duration
    print(f"Filtering cuts with duration >= {min_duration}s...")
    initial_count = len(cuts)
    cuts = cuts.filter(lambda c: c.duration >= min_duration).to_eager()
    filtered_count = len(cuts)
    print(f"  Kept {filtered_count}/{initial_count} cuts after duration filtering")
    
    # Convert to list for random sampling
    cuts_list = list(cuts)
    
    # Determine number of mixtures to create
    if num_mixtures is None:
        num_mixtures = len(cuts_list) // k
    
    print(f"Creating {num_mixtures} mixtures with {k} speakers each...")
    
    # Set random seed
    rng = random.Random(seed)
    
    # Build speaker-to-cuts mapping if distinct speakers required
    speaker_to_cuts = {}
    if distinct_speakers:
        for cut in cuts_list:
            # Assume the first supervision contains the speaker information
            if cut.supervisions:
                speaker_id = cut.supervisions[0].speaker
                if speaker_id not in speaker_to_cuts:
                    speaker_to_cuts[speaker_id] = []
                speaker_to_cuts[speaker_id].append(cut)
        
        if len(speaker_to_cuts) < k:
            print(f"Warning: Only {len(speaker_to_cuts)} distinct speakers found, but k={k} requested.")
            print(f"Setting distinct_speakers=False to allow same speaker mixtures.")
            distinct_speakers = False
    
    # Create mixtures
    mixed_cuts = []
    for i in tqdm(range(num_mixtures), desc="Creating mixtures", total=num_mixtures):
        if distinct_speakers:
            # Sample k different speakers first
            selected_speakers = rng.sample(list(speaker_to_cuts.keys()), k)
            # Then sample one cut from each speaker
            sampled_cuts = [rng.choice(speaker_to_cuts[spk]) for spk in selected_speakers]
        else:
            # Sample k random cuts (without replacement for this mixture)
            sampled_cuts = rng.sample(cuts_list, k)
        
        # Create mixed cut with overlap
        mixed_cut = create_mixed_cut_with_overlap(
            sampled_cuts, snr_range=snr_range, min_overlap=min_overlap
        )
        mixed_cuts.append(mixed_cut)
    
    # Create output CutSet
    output_cutset = CutSet.from_cuts(mixed_cuts)
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(mixed_cuts)} mixed cuts to {output_path}...")
    output_cutset.to_file(output_path)
    
    print("Done!")
    print(f"Output saved to: {output_path}")
    print(f"Total mixtures created: {len(mixed_cuts)}")
    print(f"Speakers per mixture: {k}")


def main():
    parser = argparse.ArgumentParser(
        description="Create k-speaker mixtures from LRS2 cutset with guaranteed overlap"
    )
    parser.add_argument(
        "--input_cutsets",
        type=str,
        nargs="+",
        help="Path(s) to the input cutset manifest file(s). Multiple paths will be concatenated."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path for the output mixed cutset manifest (including filename)"
    )
    parser.add_argument(
        "-k",
        type=int,
        help="Number of speakers to mix (max overlapped speakers at one time)"
    )
    parser.add_argument(
        "--num-mixtures",
        type=int,
        default=1000,
        help="Number of mixtures to create (default: len(cutset) // k)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--allow-same-speaker",
        action="store_true",
        help="Allow the same speaker to appear multiple times in a mixture (default: False)"
    )
    parser.add_argument(
        "--snr-range",
        type=float,
        nargs=2,
        default=[0, 15],
        metavar=("MIN_SNR", "MAX_SNR"),
        help="SNR range in dB for mixing non-reference tracks (default: 0 15)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=5.0,
        help="Minimum duration in seconds for cuts to be considered (default: 5.0)"
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=1.0,
        help="Minimum overlap duration in seconds between tracks (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Convert snr_range list to tuple if provided
    snr_range = tuple(args.snr_range) if args.snr_range is not None else None
    
    create_k_speaker_mixtures(
        input_cutset_paths=args.input_cutsets,
        output_path=args.output_path,
        k=args.k,
        num_mixtures=args.num_mixtures,
        seed=args.seed,
        distinct_speakers=not args.allow_same_speaker,
        snr_range=snr_range,
        min_duration=args.min_duration,
        min_overlap=args.min_overlap,
    )


if __name__ == "__main__":
    main()
