#!/usr/bin/env python3
"""
Add per-speaker ASD JSON paths to cuts manifest.

For each cut in a Lhotse cuts manifest (JSONL or JSON array), this script
adds/updates `cut['custom']['per_spk_asd']` with a mapping
speaker_id -> path_to_filled_asd.json.

Path inference follows the layout used by `fill_asd_jsons.py`:
- If `--filled_root` is provided, the filled file is expected at
  `filled_root / session_name / 'speakers' / speaker_id / 'tracks_filled_asd.json'`.
- Otherwise, the script tries these candidates (in order):
  1) session_dir / 'speakers' / speaker_id / 'tracks_filled_asd.json'
  2) session_dir.parent / session_name / 'speakers' / speaker_id / 'tracks_filled_asd.json'

The session_dir is inferred from `cut['recording']['sources'][0]['source']`,
which is expected to point to `.../session_dir/session_name/central_video.mp4`.

Example:
  python random_helper_scripts/add_per_spk_asd_to_cuts.py \
      --manifest cuts.jsonl --out cuts.with_asd.jsonl --filled_root /filled_out
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any

try:
    from lhotse import CutSet
except Exception as e:
    raise RuntimeError("lhotse is required for this script. Install with `pip install lhotse`")


def infer_session_dir_from_source(source: str) -> Path:
    # source expected like: /.../session_dir/session_name/central_video.mp4
    p = Path(source)
    # parent is session dir
    return p.parent


def get_speaker_id_from_supervision(sup) -> str:
    # `sup` is a Lhotse SupervisionSegment
    if hasattr(sup, 'speaker') and sup.speaker is not None:
        return str(sup.speaker)
    if hasattr(sup, 'speaker_id') and sup.speaker_id is not None:
        return str(sup.speaker_id)
    if hasattr(sup, 'recording_id') and sup.recording_id is not None:
        ch = getattr(sup, 'channel', None)
        return f"{sup.recording_id}_ch{ch}" if ch is not None else str(sup.recording_id)
    raise KeyError('Could not determine speaker id from supervision')


def build_candidate_paths(session_dir: Path, session_name: str, speaker_id: str, filled_root: Path = None) -> List[Path]:
    candidates = []
    fname = 'tracks_filled_asd.json'
    if filled_root is not None:
        candidates.append(Path(filled_root) / session_name / 'speakers' / speaker_id / fname)
    # try in-session speakers folder
    candidates.append(session_dir / 'speakers' / speaker_id / fname)
    # try session_dir.parent / session_name / speakers / ... (matches fill_asd_jsons.py out_root/session_name/...)
    candidates.append(session_dir.parent / session_name / 'speakers' / speaker_id / fname)
    return candidates


def process_cuts(cutset: CutSet, filled_root: Path = None, allow_missing: bool = False) -> int:
    new_cuts = []
    missing = 0
    for cut in cutset:
        # ensure custom exists
        if cut.custom is None:
            cut.custom = {}

        # Get recording source
        try:
            src_path = cut.recording.sources[0].source
        except Exception:
            print(f"Warning: cut {cut.id} has no recording sources; skipping")
            continue

        session_dir = infer_session_dir_from_source(src_path)
        session_name = session_dir.name

        per_spk = {}
        # speakers = sorted(CutSet.from_cuts([cut]).speakers)

        supervisions = cut.supervisions or []
        for sup in supervisions:
            try:
                spk = get_speaker_id_from_supervision(sup)
            except KeyError:
                print(f"Warning: could not get speaker id for supervision in cut {cut.id}; skipping supervision")
                continue

            candidates = build_candidate_paths(session_dir, session_name, spk, filled_root)
            chosen = None
            for p in candidates:
                if p.exists():
                    chosen = str(p)
                    break

            if chosen is None:
                missing += 1
                if allow_missing and candidates:
                    # set the first candidate as the expected path even if missing
                    chosen = str(candidates[0])
                else:
                    chosen = None

            per_spk[spk] = chosen

        cut.custom['per_spk_asd'] = per_spk
        new_cuts.append(cut)

    return CutSet.from_cuts(new_cuts), missing
# sed -i 's|http://157.230.104.137|http://dashboard.mybikecounter.com|g' /home/nvidia/car_counter/camera-object-counter/config.json; echo 'nvidia' | sudo -S docker restart ml_container; echo 'nvidia' | sudo docker logs --tail 0 -f ml_container


def main():
    parser = argparse.ArgumentParser(description='Add per-speaker ASD JSON paths to Lhotse CutSet manifest')
    parser.add_argument('--manifest', '-m', required=True, help='Input Lhotse CutSet manifest (JSONL/JSON/.jsonl.gz)')
    parser.add_argument('--out', '-o', required=False, help='Output manifest path (defaults to overwrite input)')
    parser.add_argument('--filled_root', '-f', required=False, help='Root where filled ASD outputs live (optional)')
    parser.add_argument('--allow_missing', action='store_true', help='Allow missing files and still write expected path')
    args = parser.parse_args()

    in_path = Path(args.manifest)
    if not in_path.exists():
        raise SystemExit(f"Manifest not found: {in_path}")

    out_path = Path(args.out) if args.out else in_path

    filled_root = Path(args.filled_root) if args.filled_root else None
    # Load CutSet using lhotse
    cutset = CutSet.from_file(str(in_path))
    print(f"Loaded CutSet with {len(cutset)} cuts")

    new_cutset, missing = process_cuts(cutset, filled_root=filled_root, allow_missing=args.allow_missing)

    # Save modified CutSet
    new_cutset.to_file(str(out_path))
    print(f"Wrote updated CutSet to {out_path}")
    if missing:
        print(f"Note: {missing} speaker ASD paths were missing (not found on disk). Use --allow_missing to still write expected paths.)")


if __name__ == '__main__':
    main()
