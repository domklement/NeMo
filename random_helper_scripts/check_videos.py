#!/usr/bin/env python3
"""Check Lhotse video cuts for frame/duration mismatch using TorchCodec.

This script iterates over cuts in a Lhotse CutSet, loads the video path from
`cut.recording.sources[0].source`, decodes frames using TorchCodec `VideoDecoder`,
and verifies that the number of decoded frames corresponds to the `cut.duration`
based on the frame rate. If the computed duration deviates from metadata beyond
the tolerance, it prints the cut ID and the source file path.

Usage:
  python random_helper_scripts/check_videos.py --cuts path/to/cuts.jsonl --num-workers 8

Requirements:
  - lhotse
  - torchcodec (https://github.com/pytorch/torchcodec)
"""

from __future__ import annotations

import argparse
import math
import os
from multiprocessing import Pool
from typing import Optional, Tuple, List

from torchcodec.decoders import VideoDecoder
from tqdm import tqdm


def _get_fps_from_decoder(decoder) -> float:
    return float(decoder.metadata.average_fps)


def _count_frames(decoder) -> int:
    """Iterate the decoder to count frames, trying common iteration APIs."""
    return len(decoder[:])


def _worker(args: Tuple[str, str, float, Optional[float], float]) -> Optional[Tuple[str, str]]:
    """Worker function: decode video, compare duration, return mismatch info.

    Args tuple: (cut_id, path, cut_duration, fps_meta, tolerance)
    Returns: (cut_id, path) if mismatch, else None
    """
    cut_id, path, cut_duration, fps_meta, tolerance = args

    # Initialize decoder
    decoder = VideoDecoder(path)

    # Count frames and determine fps
    frame_count = _count_frames(decoder)
    
    if frame_count == 0:
        return (cut_id, path)
    return None

    fps = fps_meta if fps_meta and fps_meta > 0 else _get_fps_from_decoder(decoder)

    if not fps or fps <= 0:
        # Without fps, we cannot compute duration precisely; treat as mismatch.
        return (cut_id, path)

    computed_duration = frame_count / fps

    # Compare with tolerance
    if math.fabs(computed_duration - cut_duration) > tolerance:
        return (cut_id, path)

    return None


def main():
    parser = argparse.ArgumentParser(description="Check Lhotse cuts vs decoded video frames.")
    parser.add_argument("--cuts", required=True, help="Path to Lhotse CutSet (json/jsonl).")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of multiprocessing workers.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Allowed absolute deviation (seconds) between metadata and computed duration.",
    )
    args = parser.parse_args()

    # Load cuts lazily to avoid big memory; from_file handles json/jsonl.
    try:
        from lhotse import CutSet
    except Exception as e:
        raise ImportError("lhotse is required to load cuts.") from e

    cuts = CutSet.from_file(args.cuts)

    # Prepare lightweight tuples for workers to avoid pickling heavy objects.
    tasks: List[Tuple[str, str, float, Optional[float], float]] = []
    for cut in cuts:
        # Expect recording and sources present per user spec.
        if not getattr(cut, "recording", None):
            # If recording is missing, skip (cannot resolve source path).
            continue
        sources = getattr(cut.recording, "sources", None)
        if not sources or len(sources) == 0:
            continue
        src_path = sources[0].source
        duration = float(cut.duration)
        # Try to read fps from metadata if present.
        fps_meta: Optional[float] = None
        try:
            video_meta = getattr(cut, "video", None)
            if video_meta is not None:
                fps_meta = getattr(video_meta, "frame_rate", None)
        except Exception:
            fps_meta = fps_meta
        if fps_meta is None:
            try:
                rec_video = getattr(cut.recording, "video", None)
                if rec_video is not None:
                    fps_meta = getattr(rec_video, "frame_rate", None)
            except Exception:
                pass

        tasks.append((cut.id, src_path, duration, fps_meta, args.tolerance))

    if not tasks:
        print("No decodable cuts found (missing recording/source).")
        return

    # Multiprocessing pool
    with Pool(processes=args.num_workers) as pool:
        itr = pool.imap_unordered(_worker, tasks, chunksize=8)
        for res in tqdm(itr, total=len(tasks), desc="Checking videos", unit="cut"):
            if res is not None:
                cut_id, path = res
                print(f"{cut_id}\t{path}")


if __name__ == "__main__":
    main()

