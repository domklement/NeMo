#!/usr/bin/env python3
"""
Create Lhotse-style manifests for MCoRec dataset using filled-in crop videos.

Example repository layout (original):
  /home/jovyan/data/chime9/mcorec_data/dev/session_132

Filled-in crops mirror structure under:
  /home/jovyan/data/chime9/mcorec_data/filled_in_crops/dev/session_132

For each speaker the filled folder contains two videos:
  - tracks_filled.mp4      (face crop video; used as the recording source)
  - tracks_filled_lip.mp4  (lip crop video)

The script will:
 - walk sessions under an original root (to find where speakers and .vtt labels exist)
 - map to the filled_in_crops root to find the two videos for each speaker
 - use torchcodec.VideoDecoder to collect video metadata (fps, num_frames, duration, width, height)
 - produce a recordings manifest and a cuts-like manifest. Each cut will contain two custom fields:
     - per_spk_face_crop_videos: { speaker_id: absolute_path_to_tracks_filled.mp4, ... }
     - per_spk_lip_crop_videos:  { speaker_id: absolute_path_to_tracks_filled_lip.mp4, ... }

The script will try to build true Lhotse objects if `lhotse` is installed. If not, it will emit JSON files
in a lhotse-compatible-ish structure.

Usage:
  python create_mcorec_lhotse_recipes.py \
    --orig-root /home/jovyan/data/chime9/mcorec_data/dev \
    --filled-root /home/jovyan/data/chime9/mcorec_data/filled_in_crops/dev \
    --output-recordings recordings.json \
    --output-cuts cuts.json

"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import warnings

from webvtt import WebVTT
from torchcodec.decoders import AudioDecoder, VideoDecoder
import lhotse
from lhotse import Recording, RecordingSet, CutSet, MonoCut, SupervisionSegment
from lhotse.audio.utils import VideoInfo
from lhotse.audio.source import AudioSource

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_vtt_transcript(vtt_path: Path) -> str:
    captions = WebVTT().read(str(vtt_path))
    texts = []
    for caption in captions:
        # caption.text may contain newlines; normalize them to spaces
        t = getattr(caption, "text", "")
        if t:
            texts.append(" ".join(line.strip() for line in t.splitlines() if line.strip()))
    return " ".join(texts)


def time_to_seconds(t: str) -> float:
        # WebVTT times are like HH:MM:SS.mmm or MM:SS.mmm
        parts = t.split(":")
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            h, m, s = parts
            return h * 3600.0 + m * 60.0 + s
        elif len(parts) == 2:
            m, s = parts
            return m * 60.0 + s
        else:
            return float(parts[0])


def build_asd_candidate_paths(session_dir: Path, session_name: str, speaker_id: str, filled_asd_root: Path = None) -> List[Path]:
    """Build candidate paths for per-speaker ASD JSON files.
    
    If filled_asd_root is provided, tries:
      - filled_asd_root / session_name / 'speakers' / speaker_id / 'tracks_filled_asd.json'
    
    Always tries:
      - session_dir / 'speakers' / speaker_id / 'tracks_filled_asd.json'
      - session_dir.parent / session_name / 'speakers' / speaker_id / 'tracks_filled_asd.json'
    """
    candidates = []
    fname = 'tracks_filled_asd.json'
    if filled_asd_root is not None:
        candidates.append(Path(filled_asd_root) / session_name / 'speakers' / speaker_id / fname)
    # try in-session speakers folder
    candidates.append(session_dir / 'speakers' / speaker_id / fname)
    # try session_dir.parent / session_name / speakers / ...
    candidates.append(session_dir.parent / session_name / 'speakers' / speaker_id / fname)
    return candidates


def parse_vtt_segments(vtt_path: Path) -> List[Dict]:
    """Return a list of segments found in the VTT file.

    Each segment is a dict: {"start": float_seconds, "end": float_seconds, "text": str}
    """
    captions = WebVTT().read(str(vtt_path))
    segments: List[Dict] = []

    for caption in captions:
        start = getattr(caption, "start", None)
        end = getattr(caption, "end", None)
        text = getattr(caption, "text", "")
        if start is None or end is None:
            continue
        try:
            s = time_to_seconds(start)
            e = time_to_seconds(end)
        except Exception:
            # skip malformed times
            continue
        # normalize text same as parse_vtt_transcript
        norm_text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        segments.append({"start": s, "end": e, "text": norm_text})
    return segments


def get_video_info_torchcodec(video_path: str) -> Dict:
    """Use torchcodec.VideoDecoder if available to obtain video info.
    The function is defensive: it introspects available attributes or methods.
    Returns a dict with keys: fps, num_frames, duration, width, height, path
    """
    if VideoDecoder is None:
        raise RuntimeError("torchcodec.VideoDecoder is required but not available")

    # Instantiate decoder - different torchcodec versions may behave differently.
    dec = VideoDecoder(video_path)
    return {
        'num_frames': len(dec),
        'fps': int(round(dec.metadata.average_fps)),
        'height': dec.metadata.height,
        'width': dec.metadata.width,
    }

def get_audio_info_torchcodec(audio_path: str) -> Dict:
    """Use torchcodec.AudioDecoder if available to obtain audio info.
    The function is defensive: it introspects available attributes or methods.
    Returns a dict with keys: sampling_rate, num_samples, duration, path
    """

    # Instantiate decoder - different torchcodec versions may behave differently.
    adec = AudioDecoder(audio_path)
    decoded = adec.get_all_samples()
    return {
        'sampling_rate': adec.metadata.sample_rate,
        'channels': adec.metadata.num_channels,
        'num_samples': decoded.data.shape[-1],
        'duration': decoded.duration_seconds,
    }


def find_sessions(orig_root: Path) -> List[Path]:
    """Find session directories under the original root.
    A session is any directory directly under orig_root (or deeper) that contains a `speakers` directory.
    """
    sessions = []
    for root, dirs, files in os.walk(orig_root):
        if "speakers" in dirs:
            sessions.append(Path(root))
    return sessions


def _process_session(session_path_str: str, orig_root_str: str, filled_root_str: str, vis_features: Optional[Dict[str, str]] = None, filled_asd_root_str: Optional[str] = None, allow_missing_asd: bool = False, use_uem: bool = False) -> Dict:
    """Process a single session and return a serializable dict with result or error.

    This function is defined at module scope so it can be pickled by multiprocessing.Pool.
    """
    session = Path(session_path_str)
    orig_root_p = Path(orig_root_str)
    filled_root_p = Path(filled_root_str)
    rel_session = session.relative_to(orig_root_p)
    filled_session = filled_root_p.joinpath(rel_session)
    # Gather speakers in this session from original speakers folder
    speakers_dir = session.joinpath("speakers")
    if not speakers_dir.exists():
        return {"error": f"Speakers directory not found in session {session}", "session": str(session)}
    speaker_dirs = [d for d in speakers_dir.iterdir() if d.is_dir()]

    try:
        central_video_path = session / "central_video.mp4"
        central_video_info = get_video_info_torchcodec(str(central_video_path))
        central_audio_info = get_audio_info_torchcodec(str(central_video_path))
    except Exception as e:
        return {"error": f"Failed to read central video/audio for {session}: {e}", "session": str(session)}
    
    if use_uem:
        with open(session / "metadata.json", "r") as f:
            metadata = json.load(f)
            uem_start = -1
            uem_end = -1

            for spk in metadata:
                uem = metadata[spk]["central"]["uem"]
                if uem_start < 0 or uem_end < 0:
                    uem_start = float(uem["start"])
                    uem_end = float(uem["end"])
                else:
                    assert uem_start == float(uem["start"]) and uem_end == float(uem["end"]), "Inconsistent UEM across speakers"
    else:
        uem_start = 0.0
        uem_end = central_audio_info.get("duration")

    cut_id = f"{session.name}"
    per_spk_face: Dict[str, str] = {}
    per_spk_lip: Dict[str, str] = {}
    per_spk_asd: Dict[str, Optional[str]] = {}
    supervisions: List[Dict] = []
    
    per_spk_features: Dict[str, Dict[str, str]] = dict()
    if vis_features:
        for k, v in vis_features.items():
            per_spk_features[k] = dict()
            per_spk_features[k] = {spk_dir.name: str(Path(v).joinpath(session.name, spk_dir.name, "all_tracks.pt")) for spk_dir in speaker_dirs}

    filled_asd_root = Path(filled_asd_root_str) if filled_asd_root_str else None

    for spk_dir in speaker_dirs:
        spk_name = spk_dir.name
        transcript_path = session / f"labels/{spk_name}.vtt"
        if not transcript_path.exists():
            return {"error": f"Transcript file not found for speaker {spk_name} at {transcript_path}", "session": str(session)}

        # Map to filled path
        filled_spk_dir = filled_session.joinpath("speakers", spk_name)
        face_vid = filled_spk_dir.joinpath("tracks_filled.mp4")
        lip_vid = filled_spk_dir.joinpath("tracks_filled_lip.mp4")
        if face_vid.exists():
            per_spk_face[spk_name] = str(face_vid.resolve())
        else:
            return {"error": f"Missing face video for {spk_name} in {filled_spk_dir}", "session": str(session)}
        if lip_vid.exists():
            per_spk_lip[spk_name] = str(lip_vid.resolve())
        else:
            return {"error": f"Missing lip video for {spk_name} in {filled_spk_dir}", "session": str(session)}

        # Find per-speaker ASD file
        asd_candidates = build_asd_candidate_paths(session, session.name, spk_name, filled_asd_root)
        asd_path = None
        for candidate in asd_candidates:
            if candidate.exists():
                asd_path = str(candidate.resolve())
                break
        
        if asd_path is None and allow_missing_asd and asd_candidates:
            # set the first candidate as the expected path even if missing
            asd_path = str(asd_candidates[0])
        
        per_spk_asd[spk_name] = asd_path

        # read vtt segments for this speaker
        segments = parse_vtt_segments(transcript_path)
        for i, seg in enumerate(segments):
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])

            if seg_end < uem_start or seg_start > uem_end:
                warnings.warn(f"Segment [{seg_start}, {seg_end}] entirely out of UEM bounds [{uem_start}, {uem_end}] for speaker {spk_name} in session {session}. Skipping it.")
                continue

            if seg_start < uem_start or seg_end > uem_end:
                warnings.warn(f"Segment [{seg_start}, {seg_end}] out of UEM bounds [{uem_start}, {uem_end}] for speaker {spk_name} in session {session}. Cutting it to fit UEM.")
                seg_end = min(seg_end, uem_end+0.5)

            seg_dur = max(0.0, seg_end - seg_start)

            # UEM start is substracted when creating SupervisionSegment in build_manifests.
            supervisions.append({
                "id": f"{cut_id}_{i}",
                "recording_id": cut_id,
                "start": seg_start,
                "duration": seg_dur,
                "channel": list(range(central_audio_info.get("channels"))),
                "speaker": spk_name,
                "text": seg.get("text", ""),
                "language": 'en',
            })

    result = {
        "cut_id": cut_id,
        "cut_start": uem_start,
        "cut_duration": uem_end - uem_start,
        "central_video_path": str(central_video_path),
        "central_video_info": central_video_info,
        "central_audio_info": central_audio_info,
        "per_spk_face": per_spk_face,
        "per_spk_lip": per_spk_lip,
        "per_spk_asd": per_spk_asd,
        "supervisions": supervisions,
        "vis_features": per_spk_features,
    }
    return {"result": result}


def build_manifests(orig_root: Path, filled_root: Path, num_workers: Optional[int] = None, vis_features: Optional[Dict[str, str]] = None, filled_asd_root: Optional[Path] = None, allow_missing_asd: bool = False, use_uem: bool = False) -> CutSet:
    """Traverse sessions and build Lhotse RecordingSet and CutSet.

    For each speaker we create a Recording (face video) and a MonoCut that spans the
    full recording duration. SupervisionSegments are created per VTT caption segment
    (one supervision per VTT segment). The function returns a tuple: (RecordingSet, CutSet).
    """
    cuts: List[MonoCut] = []

    sessions = find_sessions(orig_root)
    logging.info(f"Found {len(sessions)} sessions under {orig_root}")

    # worker that processes a single session and returns a serializable dict
    # Use the module-level _process_ses§sion (defined above) which is picklable by multiprocessing.Pool

    # Run processing in parallel with a progress bar
    sessions_strs = [str(s) for s in sessions]
    results: List[Dict] = []
    worker = partial(_process_session, orig_root_str=str(orig_root), filled_root_str=str(filled_root), vis_features=vis_features, filled_asd_root_str=str(filled_asd_root) if filled_asd_root else None, allow_missing_asd=allow_missing_asd, use_uem=use_uem)
    # determine number of workers: use user-provided if given, otherwise use cpu_count()
    if num_workers is None:
        num_workers = min(cpu_count(), max(1, len(sessions_strs)))
    else:
        # ensure at least 1 and at most number of sessions
        num_workers = max(1, min(int(num_workers), len(sessions_strs)))
    with Pool(processes=num_workers) as p:
        for res in tqdm(p.imap_unordered(worker, sessions_strs), total=len(sessions_strs), desc="Processing sessions"):
            results.append(res)

    # convert results to MonoCut objects, collecting errors if any
    for r in results:
        if "error" in r:
            logging.error(f"Error processing session: {r.get('error')} (session={r.get('session')})")
            continue
        payload = r.get("result")
        if not payload:
            continue

        cut_id = payload["cut_id"]
        central_video_path = Path(payload["central_video_path"])
        central_video_info = payload["central_video_info"]
        central_audio_info = payload["central_audio_info"]

        if use_uem:
            uem_start = payload["cut_start"]
            uem_duration = payload["cut_duration"]
        else:
            uem_start = 0.0
            uem_duration = central_audio_info.get("duration")

        recording = Recording(
            id=cut_id,
            channel_ids=list(range(central_audio_info.get("channels"))),
            duration=central_audio_info.get("duration"),
            num_samples=central_audio_info.get("num_samples"),
            sampling_rate=central_audio_info.get("sampling_rate"),
            sources=[
                AudioSource(
                    channels=list(range(central_audio_info.get("channels"))),
                    source=str(central_video_path),
                    type='file',
                    video=VideoInfo(
                        fps=int(central_video_info.get("fps")),
                        height=central_video_info.get("height"),
                        num_frames=central_video_info.get("num_frames"),
                        width=central_video_info.get("width"),
                    ),
                )
            ],
        )

        supervisions: List[SupervisionSegment] = []
        for sup in payload["supervisions"]:
            s = SupervisionSegment(
                id=sup["id"],
                recording_id=sup["recording_id"],
                start=sup["start"] - uem_start,
                duration=sup["duration"],
                channel=sup["channel"],
                speaker=sup["speaker"],
                text=sup.get("text", ""),
                language=sup.get("language", "en"),
            )
            supervisions.append(s)
        
        custom_fields = {
            "per_spk_face_crop_videos": payload["per_spk_face"],
            "per_spk_lip_crop_videos": payload["per_spk_lip"],
            "per_spk_asd": payload["per_spk_asd"],
            **(payload['vis_features'] if 'vis_features' in payload else {})
        }
        
        cut = MonoCut(
            id=f"{cut_id}_cut0",
            start=uem_start,
            duration=uem_duration,
            channel=recording.channel_ids,
            recording=recording,
            supervisions=supervisions,
            custom=custom_fields,
        )
        cuts.append(cut)

    return CutSet.from_cuts(cuts)


def main():
    parser = argparse.ArgumentParser(description="Create Lhotse manifests for MCoRec filled-in crops")
    parser.add_argument("--orig-root", required=True, type=Path, help="Original MCoRec root (e.g., /.../mcorec_data/dev)")
    parser.add_argument("--filled-root", required=True, type=Path, help="Filled-in crops root that mirrors orig root")
    parser.add_argument("--output-cuts", required=True, type=Path, help="Output cuts manifest path (json)")
    parser.add_argument('--visual-feature-keys', type=str, nargs='+', default=None, help="List of visual feature keys to include in the cuts' custom fields")
    parser.add_argument('--visual-feature-dirs', type=str, nargs='+', default=None, help="List of directories containing visual features corresponding to the keys")
    parser.add_argument('--filled-asd-root', type=Path, default=None, help="Root where filled ASD JSON outputs live (optional)")
    parser.add_argument('--allow-missing-asd', action='store_true', help="Allow missing ASD files and still write expected path")
    parser.add_argument('--use-uem', action='store_true', help="Use UEM files for segmenting the recordings")
    parser.add_argument('--num-workers', type=int, default=None, help="Number of parallel workers to use")

    args = parser.parse_args()
    assert len(args.visual_feature_keys or []) == len(args.visual_feature_dirs or []), "Number of visual feature keys must match number of directories"

    if VideoDecoder is None:
        logging.error("torchcodec.VideoDecoder is not importable. Please install torchcodec to proceed.")

    cuts_set = build_manifests(args.orig_root, args.filled_root, num_workers=args.num_workers, vis_features=dict(list(zip(args.visual_feature_keys or [], args.visual_feature_dirs or []))), filled_asd_root=args.filled_asd_root, allow_missing_asd=args.allow_missing_asd, use_uem=args.use_uem)

    logging.info(f"Writing cuts to {args.output_cuts}")
    cuts_set.to_file(str(args.output_cuts))


if __name__ == "__main__":
    main()
