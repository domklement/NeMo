#!/usr/bin/env python3
"""
Minimal tool to fill gaps between speaker tracks using VideoDecoder + OpenCV.

For each session directory (contains central_video.mp4 and metadata.json) the
script creates per-speaker combined videos for both the main crop ('video') and
the lip crop ('lip'). Each combined video has the same number of frames as the
central video: frames outside track ranges are filled with black frames.

Constraints (per your request):
- Do NOT use ffmpeg. Only use torchcodec.VideoDecoder for reading videos and
  OpenCV (cv2) for writing and resizing frames.
- Do NOT attach audio yet.

Usage example:
  python script/fill_tracks.py --session_dir data-bin/dev/session_132 --output_root filled_out

This is minimal and focuses only on frame alignment and writing.
"""
import argparse
import json
import math
import os
from pathlib import Path
import glob
import time

import cv2
import numpy as np
import subprocess
import tempfile
from tqdm import tqdm

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
        start = int(round(float(meta.get('start_time', 0.0)) * FPS))
        end = int(round(float(meta.get('end_time', start / FPS)) * FPS))
    return start, end


def verify_output_video(path: Path, expected_frames: int, check_audio: bool = False, central_vid: Path = None,
                        fps: int = FPS):
    """Verify that `path` has `expected_frames` frames. Optionally verify audio exists.

    Raises RuntimeError when checks fail. Uses VideoDecoder for frame counting and ffprobe
    for audio presence (ffprobe must be available when check_audio=True).
    """
    if not path.exists():
        raise RuntimeError(f"Output file does not exist: {path}")

    # frame count check
    dec_out = VideoDecoder(str(path), device="cpu", seek_mode="exact", num_ffmpeg_threads=1, dimension_order="NHWC")
    n_out = len(dec_out)
    if n_out != expected_frames:
        raise RuntimeError(f"Frame count mismatch for {path}: expected {expected_frames}, got {n_out}")

    # audio check (if requested)
    if check_audio:
        try:
            # list audio streams; output empty if none
            ffprobe_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', str(path)
            ]
            out = subprocess.check_output(ffprobe_cmd, stderr=subprocess.STDOUT)
            has_audio = bool(out.strip())
        except FileNotFoundError:
            raise RuntimeError('ffprobe not found in PATH; cannot verify audio presence')
        except subprocess.CalledProcessError:
            # ffprobe failed -> treat as no audio
            has_audio = False

        if not has_audio:
            raise RuntimeError(f"Audio stream not found in {path}")

    print(f"Verification passed for {path}: {n_out} frames{', audio OK' if check_audio else ''}")


def write_combined_video(session_dir: Path, tracks: list, video_field: str, total_frames: int, out_path: Path, time_offset_seconds: float=0.0, fps: int = FPS):
    """Create a combined video for the given tracks and video_field ('video' or 'lip').
    Frames not covered by any track are written as black frames.
    """
    # Build track descriptors: (start_frame, end_frame, decoder)
    track_descs = []
    target_w = None
    target_h = None

    time_offset_frames = int(round(time_offset_seconds * fps))

    for t in tracks:
        rel = t.get(video_field) or t.get('video') or t.get('lip')
        if rel is None:
            # this track does not have the requested field
            continue
        vid_path = session_dir / rel
        if not vid_path.exists():
            print(f"Warning: track video not found (skipping): {vid_path}")
            continue
        meta = load_crop_meta(session_dir, t)
        start, end = get_frame_range_from_meta(meta)
        start += max(0, time_offset_frames)
        end += max(0, time_offset_frames)
        dec = VideoDecoder(str(vid_path), device="cpu", seek_mode="exact", num_ffmpeg_threads=1, dimension_order="NHWC")
        n_frames = len(dec)

        # determine video frame size from decoder first frame
        if n_frames > 0:
            frame0 = dec[0]
            h, w = int(frame0.shape[0]), int(frame0.shape[1])
            if target_w is None:
                target_w, target_h = w, h
            else:
                # keep max dims to avoid upscaling smaller crops unnecessarily
                target_w = max(target_w, w)
                target_h = max(target_h, h)

        track_descs.append({'start': start, 'end': end, 'dec': dec, 'n_frames': n_frames})

    if target_w is None or target_h is None:
        # nothing to write for this field
        print(f"No track videos found for field '{video_field}', skipping output {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # write first to a temporary no-audio file using ffmpeg pipe if available,
    # otherwise fall back to OpenCV VideoWriter
    out_noaudio = out_path.with_suffix('.noaudio.mp4')

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f"{target_w}x{target_h}",
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '0',
        '-pix_fmt', 'rgb24',
        '-hide_banner', '-loglevel', 'error',
        str(out_noaudio)
    ]

    use_ffmpeg_pipe = True
    pipe = None
    try:
        # capture stderr so we can diagnose ffmpeg failures
        pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        use_ffmpeg_pipe = False

    if not use_ffmpeg_pipe:
        # OpenCV writer (MJPG or mp4v may be available). Using mp4v for .mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_noaudio), fourcc, float(fps), (target_w, target_h))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open video writer for {out_noaudio}")

    # Sort track_descs by start so we pick first matching when overlaps exist
    track_descs = sorted(track_descs, key=lambda x: x['start'])

    # For each central frame index, choose frame from the first covering track or black
    for fid in tqdm(range(total_frames), desc=f"Writing {out_noaudio}", unit="frame"):
        fidx = fid - min(0, time_offset_frames)  # adjust for negative offset if any.
        chosen = None
        for td in track_descs:
            if td['start'] <= fidx < td['end']:
                local_idx = fidx - td['start']
                if 0 <= local_idx < td['n_frames']:
                    try:
                        frm = td['dec'][local_idx].numpy()  # HWC, uint8 (likely RGB)
                    except Exception:
                        frm = None
                else:
                    frm = None
                chosen = frm
                break

        if chosen is None:
            frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            frame = chosen

        # Ensure correct size
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if use_ffmpeg_pipe:
            # ffmpeg expects raw RGB24 bytes
            try:
                pipe.stdin.write(frame.tobytes())
            except Exception as e:
                pipe.stdin.close()
                pipe.wait()
                raise
        else:
            # OpenCV expects BGR
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    
    # time.sleep(1)

    if use_ffmpeg_pipe:
        pipe.stdin.close()
        rc = pipe.wait()
        if rc != 0:
            stderr_txt = ''
            try:
                stderr_txt = pipe.stderr.read().decode('utf-8', errors='replace')
            except Exception:
                pass
            raise RuntimeError(f"ffmpeg exited with code {rc} while writing {out_noaudio}: {stderr_txt}")
    else:
        writer.release()
    print(f"Wrote combined video (no audio): {out_noaudio}")

    # Verify that the produced no-audio video has the expected number of frames
    try:
        verify_output_video(out_noaudio, total_frames, check_audio=False)
    except Exception:
        # re-raise with context preserved
        raise

    # Attach central audio (if available) using ffmpeg. If ffmpeg is not present,
    # fall back to leaving the no-audio file under the original out_path name.
    central_vid = session_dir / 'central_video.mp4'
    audio_attached = False
    if central_vid.exists():
        try:
            # run ffmpeg to mux audio from central into the video
            cmd = [
                'ffmpeg', '-y', '-i', str(out_noaudio), '-i', str(central_vid),
                '-c:v', 'copy', '-c:a', 'aac', 
                '-hide_banner', '-loglevel', 'error',
                '-strict', 'experimental',
                '-map', '0:v:0', '-map', '1:a:0', str(out_path)
            ]
            subprocess.check_call(cmd)
            audio_attached = True
            print(f"Attached audio from {central_vid} -> {out_path}")
            # remove intermediate no-audio file
            try:
                os.remove(str(out_noaudio))
            except Exception:
                pass
        except FileNotFoundError:
            # ffmpeg not found
            print("ffmpeg not found in PATH; leaving output without audio")
            # rename no-audio to requested path
            try:
                os.replace(str(out_noaudio), str(out_path))
            except Exception:
                print(f"Failed to move {out_noaudio} to {out_path}")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed attaching audio: {e}; leaving no-audio file at {out_noaudio}")
            # attempt to keep a copy at out_path for consistency
            try:
                os.replace(str(out_noaudio), str(out_path))
            except Exception:
                pass
    else:
        # central audio not present; move no-audio to final location
        print(f"Central video not found ({central_vid}); leaving output without audio")
        try:
            os.replace(str(out_noaudio), str(out_path))
        except Exception:
            print(f"Failed to move {out_noaudio} to {out_path}")

    # Final verification: ensure final file has expected frames and (if attached) an audio stream
    try:
        verify_output_video(out_path, total_frames, check_audio=audio_attached, central_vid=central_vid if audio_attached else None)
    except Exception:
        # raise to let caller handle it (keeps behavior explicit rather than silent)
        raise


def process_session(session_dir: Path, output_root: Path, crop_type: str = 'central', fps: int = FPS):
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

    dec_central = VideoDecoder(str(central_video), device="cpu", seek_mode="exact", num_ffmpeg_threads=1, dimension_order="NHWC")
    total_frames = len(dec_central)
    print(f" Central frames: {total_frames}")

    out_session = output_root / session_dir.name

    for speaker_name, spk in metadata.items():
        spk_out = out_session / 'speakers' / speaker_name

        # Calculate time offset if processing ego crops
        time_offset = 0.0
        if crop_type == 'ego':
            assert 'ego' in spk

            ego_conv_start = spk['ego']['uem']['start']
            central_conv_start = spk['central']['uem']['start']

            # If this offset is positive, we need to add black frames at the start of ego crop.
            # If it's negative, ego crop starts before central crop. In such a case, we need to shift the frame idx by the offset amount.
            # The seconds scenario is handled in write_combined_video by adding time_offset_seconds to start/end frame indices.
            time_offset = central_conv_start - ego_conv_start
            # assert time_offset >= 0.0, "Ego crop starts after central crop, unexpected!"

        # process crops
        if crop_type in spk and 'crops' in spk[crop_type]:
            tracks = spk[crop_type]['crops']
            # create combined for main video field
            out_main = spk_out / 'tracks_filled.mp4'
            write_combined_video(session_dir, tracks, 'video', total_frames, out_main, fps=fps, time_offset_seconds=time_offset)
            # create combined for lip field
            out_lip = spk_out / 'tracks_filled_lip.mp4'
            write_combined_video(session_dir, tracks, 'lip', total_frames, out_lip, fps=fps, time_offset_seconds=time_offset)


def main():
    parser = argparse.ArgumentParser(description="Fill gaps between track videos using VideoDecoder + OpenCV")
    parser.add_argument('--session_dir', type=str, required=True, help='Session dir or glob pattern')
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--crop_type', type=str, default='central', choices=['central', 'ego'], help='Crop type to process (central or ego)')
    parser.add_argument('--fps', type=int, default=FPS)
    opt = parser.parse_args()

    sess_pattern = opt.session_dir
    if '*' in sess_pattern:
        all_sessions = glob.glob(sess_pattern)
    else:
        all_sessions = [sess_pattern]

    out_root = Path(opt.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for s in all_sessions:
        process_session(Path(s), out_root, crop_type=opt.crop_type, fps=opt.fps)


if __name__ == '__main__':
    main()
