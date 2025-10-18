#!/usr/bin/env python3
"""Add spk_to_vid_path mapping to each JSON object in a JSONL manifest.

Usage:
  mcorec_add_vid_paths.py --manifest_path PATH --sessions_root PATH [--inplace]

For each line (JSON) the script extracts the session name from the "audio_filepath"
field (assumes the session is a path component like .../session_42/...) and collects
unique speaker ids from the "text" list (expects each element to be a dict with a
"speaker" key). It then adds a "spk_to_vid_path" dict mapping each speaker id to
"{session}/speakers/{spk}/all_tracks.pt".

The modified lines are written to a new file with suffix ".with_vid_paths.jsonl"
unless --inplace is specified.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {lineno} of {path}: {e}")


def extract_session_from_audio_path(audio_path: str) -> str:
    # Assume session is the directory name immediately containing the audio file
    # Example: /.../session_42/central_audio.wav -> session_42
    parts = audio_path.replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot extract session from audio path: '{audio_path}'")
    # session is the parent directory of the file
    return parts[-2]


def build_spk_to_vid(sessions_root: str, session, speakers: Iterable[str]) -> dict:
    return {spk: os.path.join(sessions_root, session, str(spk), "all_tracks.pt") for spk in speakers}


def process_manifest(manifest_path: str, sessions_root: str, out_path) -> int:
    total = 0
    updated = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for obj in iter_jsonl(manifest_path):
            total += 1
            audio_fp = obj.get("audio_filepath")
            if not audio_fp:
                print(f"Warning: no audio_filepath for entry #{total}, skipping")
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            try:
                session = extract_session_from_audio_path(audio_fp)
            except Exception as e:
                print(f"Warning: failed to extract session for entry #{total}: {e}")
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            text = obj.get("text")
            speakers = []
            if isinstance(text, list):
                for elt in text:
                    if not isinstance(elt, dict):
                        continue
                    spk = elt.get("speaker")
                    if spk is None:
                        continue
                    speakers.append(spk)
            else:
                print(f"Warning: 'text' field not a list for entry #{total} (session={session})")

            unique_speakers = list(dict.fromkeys(str(s) for s in speakers))
            spk_map = build_spk_to_vid(sessions_root, session, unique_speakers)
            obj["per_spk_feature_files"] = spk_map
            updated += 1
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({updated}/{total} entries updated)")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Add speaker -> video path mapping to JSONL manifest")
    parser.add_argument("--manifest_path", required=True, help="Path to JSONL manifest (one json per line)")
    parser.add_argument("--sessions_root", required=True, help="Path to sessions root (not used for content, kept for validation)")
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()

    if not os.path.isfile(args.manifest_path):
        raise SystemExit(f"Manifest path does not exist or is not a file: {args.manifest_path}")

    if not os.path.isdir(args.sessions_root):
        print(f"Warning: sessions_root '{args.sessions_root}' does not exist or is not a directory. The script will still proceed.")

    process_manifest(args.manifest_path, args.sessions_root, args.output_path)


if __name__ == "__main__":
    main()
