#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from lhotse import (
    AudioSource,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
)
from lhotse.cut import MonoCut, CutSet


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(
        description="Convert JSONL (one cut per line with per-speaker segments) to a Lhotse CutSet."
    )
    ap.add_argument("input_jsonl", type=Path, help="Path to input JSONL.")
    ap.add_argument(
        "--out-cuts",
        type=Path,
        default=Path("cuts.jsonl.gz"),
        help="Output CutSet path (jsonl[.gz]).",
    )
    args = ap.parse_args()

    recordings = {}
    cuts = []

    with args.input_jsonl.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Line {ln} is not valid JSON: {e}")

            audio_path = str(ex["audio_filepath"])
            total_dur = float(ex["duration"])
            offset = float(ex.get("offset", 0.0))

            # Build (or reuse) Recording
            rec_id = sha1(audio_path)
            if rec_id not in recordings:
                # if args.probe_audio:
                #     # Safer: inspect file to get exact duration/rate/channel info
                #     rec = Recording.from_file(audio_path, recording_id=rec_id)
                # else:
                #     # Faster: trust provided duration; samplerate unknown here
                #     # Lhotse allows minimal Recording construction via AudioSource.
                src = AudioSource(type="file", channels=[0], source=audio_path)
                rec = Recording(
                    id=rec_id,
                    sources=[src],
                    sampling_rate=None,  # unknown when not probing
                    num_samples=None,
                    duration=total_dur,
                )
                recordings[rec_id] = rec
            rec = recordings[rec_id]

            # Cut id: stable & unique per line
            cut_id = ex.get("id") or f"cut-{sha1(f'{audio_path}|{offset}|{total_dur}|{ln}')}"
            # Supervisions: from the per-speaker segments
            sups = []
            for i, seg in enumerate(ex.get("text", [])):
                start = float(seg["start"])  # absolute wrt file (as in your example)
                dur = float(seg["duration"])
                # Keep only chunks that overlap the cut time span:
                seg_end = start + dur
                cut_start = offset
                cut_end = offset + total_dur
                if seg_end <= cut_start or start >= cut_end:
                    continue
                # Clip to the cut window:
                adj_start = max(start, cut_start)
                adj_end = min(seg_end, cut_end)
                adj_dur = max(0.0, adj_end - adj_start)
                if adj_dur <= 0:
                    continue

                sup = SupervisionSegment(
                    id=f"{cut_id}-sup-{i}",
                    recording_id=rec.id,
                    start=adj_start,     # absolute time in recording
                    duration=adj_dur,
                    channel=0,
                    text=seg.get("text", None),
                    speaker=seg.get("speaker", None),
                    language=seg.get("language", None),
                )
                sups.append(sup)

            # Create the MonoCut spanning the requested portion of the recording.
            cut = MonoCut(
                id=cut_id,
                start=offset,
                duration=total_dur,
                channel=0,
                recording=rec,
                supervisions=sups,
                custom={
                    # 'video_features_path': 
                    'per_spk_feature_files': ex.get("per_spk_feature_files", {}),  # {speaker: path to features}
                }
            )
            cuts.append(cut)

    # Dump manifests
    CutSet.from_cuts(cuts).to_file(args.out_cuts)
    # RecordingSet.from_recordings(list(recordings.values())).to_file(args.out_recs)

    print(f"Wrote {len(cuts)} cuts -> {args.out_cuts}")
    # print(f"Wrote {len(recordings)} recordings -> {args.out_recs}")
    print("Tip: validate with `lhotse validate --recordings recordings.jsonl.gz cuts.jsonl.gz`")


if __name__ == "__main__":
    main()