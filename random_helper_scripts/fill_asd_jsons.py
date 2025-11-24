#!/usr/bin/env python3
"""
Tool to fill gaps between speaker tracks in ASD JSONs.

For each session directory (contains central_video.mp4 and metadata.json) the
script creates per-speaker combined ASD JSONs. Each combined JSON has one
entry per frame index of the central video: frames outside track ranges are
filled with a constant value equal to the minimum log-likelihood observed in
any track ASD dictionary for that speaker.

Assumptions:
- Each track has a crop metadata file referenced in metadata.json
  (field 'crop_metadata').
- For each track video (e.g. '.../track_01.mp4') there is a corresponding
  ASD JSON file named 'track_01_asd.json' in the same directory.
  The ASD JSON contains a single dict mapping local frame indices
  (as strings "0", "1", ..., "N-1") to log-likelihood values (floats).
- The central video is used only to determine the total number of frames.

Usage example:
  python script/fill_tracks_asd.py --session_dir data-bin/dev/session_132 --output_root filled_out
"""

import argparse
import json
import glob
import multiprocessing
from pathlib import Path

try:
    from torchcodec.decoders import VideoDecoder
except Exception as e:
    VideoDecoder = None
    raise RuntimeError("torchcodec.VideoDecoder is required for this script")

FPS = 25


def load_crop_meta(session_dir: Path, track: dict):
    meta_path = session_dir / track.get('crop_metadata', '')
    if not meta_path.exists():
        raise FileNotFoundError(f"Crop metadata not found: {meta_path}")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta


def get_frame_range_from_meta(meta: dict):
    # Prefer explicit frame indices if present, otherwise use start_time/end_time
    if 'frame_start' in meta and 'frame_end' in meta:
        start = int(meta['frame_start'])
        end = int(meta['frame_end'])
    else:
        # Fallback to times * FPS
        start = int(round(float(meta.get('start_time', 0.0)) * FPS))
        end = int(round(float(meta.get('end_time', start / FPS)) * FPS))
    return start, end


def build_asd_path_for_track(session_dir: Path, track: dict):
    """
    Infer the ASD JSON path for a track.

    We assume that either 'video' or 'lip' field points to a track video and
    that the ASD JSON is named '<track_stem>_asd.json' in the same directory.

    Example:
      video: 'speakers/spk1/crops/track_01.mp4'
      ASD : 'speakers/spk1/crops/track_01_asd.json'
    """
    rel = track.get('video') or track.get('lip')
    if rel is None:
        return None
    vid_path = session_dir / rel
    asd_path = vid_path.parent / f"{vid_path.stem}_asd.json"
    return asd_path


def write_combined_asd(session_dir: Path, tracks: list, total_frames: int, out_path: Path):
    """
    Create a combined ASD JSON for the given tracks.

    - For each frame index f in [0, total_frames):
        * If covered by at least one track (first by start time):
            value = ASD[local_frame_idx] if present
                    else min over that track's ASD dict
        * If not covered by any track:
            value = global_min_asd over all tracks' ASD values

    The output is a dict with string keys "0".."total_frames-1".
    """

    track_descs = []
    global_min = None

    # Collect per-track ASD info
    for t in tracks:
        try:
            meta = load_crop_meta(session_dir, t)
        except FileNotFoundError as e:
            print(f"Warning: {e} (skipping track)")
            continue

        start, end = get_frame_range_from_meta(meta)
        asd_path = build_asd_path_for_track(session_dir, t)
        if asd_path is None:
            print(f"Warning: could not infer ASD path for track (no video/lip field): {t}")
            continue
        if not asd_path.exists():
            print(f"Warning: ASD JSON not found (skipping track): {asd_path}")
            continue

        with open(asd_path, 'r') as f:
            asd_dict = json.load(f)

        if not isinstance(asd_dict, dict) or len(asd_dict) == 0:
            print(f"Warning: ASD JSON empty or not a dict (skipping track): {asd_path}")
            continue

        # Values are log-likelihoods (floats), keys are "0".."N-1"
        try:
            values = list(asd_dict.values())
            local_min = min(values)
        except Exception:
            print(f"Warning: ASD JSON has non-numeric values (skipping track): {asd_path}")
            continue

        if global_min is None or local_min < global_min:
            global_min = local_min

        n_frames = len(asd_dict)

        track_descs.append({
            'start': start,
            'end': end,
            'asd': asd_dict,
            'local_min': local_min,
            'n_frames': n_frames,
        })

    if not track_descs:
        print(f"No valid ASD tracks found for {out_path}, skipping output.")
        return

    if global_min is None:
        print(f"Could not compute global minimum ASD value for {out_path}, skipping output.")
        return

    # Sort by start so we pick the first matching track when overlaps exist
    track_descs = sorted(track_descs, key=lambda x: x['start'])

    # Build combined ASD dict
    combined_asd = {}

    for fidx in range(total_frames):
        value = None

        for td in track_descs:
            if td['start'] <= fidx < td['end']:
                local_idx = fidx - td['start']
                if 0 <= local_idx < td['n_frames']:
                    key = str(local_idx)
                    if key in td['asd']:
                        value = td['asd'][key]
                    else:
                        # Local gap inside track ASD: use track-local minimum
                        value = td['local_min']
                else:
                    # Out of local range (shouldn't normally happen): use track-local minimum
                    value = td['local_min']
                break  # first covering track wins

        if value is None:
            # Not covered by any track: use global minimum
            value = global_min

        combined_asd[str(fidx)] = value

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(combined_asd, f)

    print(f"Wrote combined ASD JSON: {out_path}")


def process_session(session_dir: Path, output_root: Path):
    print(f"Processing session: {session_dir}")
    meta_file = session_dir / 'metadata.json'
    if not meta_file.exists():
        print(f"metadata.json not found in {session_dir}, skipping")
        return
    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    central_video = session_dir / 'central_video.mp4'
    if not central_video.exists():
        print(f"central_video.mp4 not found in {session_dir}, skipping")
        return

    # Use central video to determine total number of frames
    dec_central = VideoDecoder(
        str(central_video),
        device="cpu",
        seek_mode="exact",
        num_ffmpeg_threads=1,
        dimension_order="NHWC"
    )
    total_frames = len(dec_central)
    print(f" Central frames: {total_frames}")

    out_session = output_root / session_dir.name

    for speaker_name, spk in metadata.items():
        spk_out = out_session / 'speakers' / speaker_name

        # process central crops
        if 'central' in spk and 'crops' in spk['central']:
            tracks = spk['central']['crops']

            # Create combined ASD JSON for this speaker
            out_asd = spk_out / 'tracks_filled_asd.json'
            write_combined_asd(session_dir, tracks, total_frames, out_asd)
        else:
            print(f"No central crops found for speaker '{speaker_name}' in {session_dir}")


def process_session_worker(arg):
    """Top-level worker wrapper for multiprocessing.Pool.

    Accepts a single argument tuple `(session_dir_str, output_root_str)` so
    it can be pickled by multiprocessing on all platforms.
    """
    s, out_root_str = arg
    try:
        process_session(Path(s), Path(out_root_str))
    except Exception as e:
        print(f"Error processing session {s}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fill gaps between track ASD JSONs")
    parser.add_argument('--session_dir', type=str, required=True, help='Session dir or glob pattern')
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of worker processes for parallel session processing. 1 = no multiprocessing')
    opt = parser.parse_args()

    sess_pattern = opt.session_dir
    if '*' in sess_pattern:
        all_sessions = glob.glob(sess_pattern)
    else:
        all_sessions = [sess_pattern]

    out_root = Path(opt.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # If requested, run sessions in parallel across processes. Keep the
    # single-process behavior when num_workers == 1 so existing usage
    # remains unchanged.
    num_workers = max(1, int(opt.num_workers))

    if num_workers > 1 and len(all_sessions) > 1:
        print(f"Processing {len(all_sessions)} sessions with {num_workers} workers")

        args = [(s, str(out_root)) for s in all_sessions]
        with multiprocessing.Pool(num_workers) as pool:
            try:
                pool.map(process_session_worker, args)
            except KeyboardInterrupt:
                print("KeyboardInterrupt caught, terminating workers...")
                pool.terminate()
                pool.join()
                raise
    else:
        # Single-process mode (default)
        for s in all_sessions:
            process_session(Path(s), out_root)


if __name__ == '__main__':
    main()
