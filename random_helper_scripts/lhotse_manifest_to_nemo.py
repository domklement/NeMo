#!/usr/bin/env python3
"""Convert a Lhotse CutSet/manifest to a NeMo-style JSONL manifest.

Usage examples:
  python lhotse_manifest_to_nemo.py --input cuts.jsonl --output nemo.jsonl

This script attempts to use `lhotse` when available. If not, it falls back
to parsing the input as JSON/JSONL where each record already contains fields
similar to Lhotse's Cut dict representation.

The per-speaker feature files are read from `cut.custom[feature_key]` by default
the key is `av_hubert_lip_features` (configurable via CLI).
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def try_import_lhotse() -> bool:
    try:
        # Import lazily so script still works if lhotse is not installed
        global CutSet
        from lhotse import CutSet  # type: ignore

        return True
    except Exception:
        return False


def iter_cuts_with_lhotse(path: str):
    # yields tuples (cut, as_dict)
    from lhotse import CutSet  # type: ignore

    # CutSet has from_jsonl/from_json methods depending on version; try both
    try:
        cuts = CutSet.from_jsonl(path)
    except Exception:
        try:
            cuts = CutSet.from_json(path)
        except Exception:
            # last fallback: use load_manifest
            from lhotse import load_manifest  # type: ignore

            cuts = load_manifest(path)

    for cut in cuts:
        yield cut, cut.to_dict()


def iter_cuts_from_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield None, json.loads(line)


def extract_audio_filepath_from_recording(rec: Any) -> str | None:
    # Try common attribute names used in different Lhotse versions
    if rec is None:
        return None
    # if recording is a dict
    if isinstance(rec, dict):
        # common keys
        for k in ("sources", "source", "wav", "audio", "audio_filepath", "uri", "path", "wav_path"):
            if k in rec:
                v = rec[k]
                if isinstance(v, list) and v:
                    # sources is often a list of dicts with 'source'
                    if isinstance(v[0], dict) and "source" in v[0]:
                        return v[0]["source"]
                    return v[0]
                return v
        # maybe nested
        if "sources" in rec and isinstance(rec["sources"], list) and rec["sources"]:
            src = rec["sources"][0]
            if isinstance(src, dict) and "source" in src:
                return src["source"]
    else:
        # try lhotse Recording attributes
        for attr in ("sources", "source", "wav", "audio", "wav_path", "uri"):
            if hasattr(rec, attr):
                v = getattr(rec, attr)
                if v is None:
                    continue
                if isinstance(v, (list, tuple)) and v:
                    first = v[0]
                    if isinstance(first, dict) and "source" in first:
                        return first["source"]
                    return first
                return v
    return None


def build_nemo_record(cut_dict: Dict[str, Any], feature_key: str) -> Dict[str, Any]:
    # Determine audio filepath
    audio_filepath = None
    # Lhotse cut dict may have 'recording' or 'recordings'
    if "recording" in cut_dict:
        audio_filepath = extract_audio_filepath_from_recording(cut_dict.get("recording"))
    elif "recordings" in cut_dict:
        audio_filepath = extract_audio_filepath_from_recording(cut_dict.get("recordings"))
    # fallback common key
    audio_filepath = audio_filepath or cut_dict.get("audio_filepath") or cut_dict.get("wav") or cut_dict.get("uri")

    # offset: start time of cut relative to recording
    offset = float(cut_dict.get("start", 0.0) or 0.0)

    # duration: prefer explicit cut duration, else compute from supervisions
    duration = cut_dict.get("duration")
    if duration is None:
        # try computing from supervisions
        sups = cut_dict.get("supervisions") or cut_dict.get("annotations") or []
        if sups:
            # compute end of last supervision
            max_end = 0.0
            for s in sups:
                s_start = float(s.get("start", 0.0) or 0.0)
                s_dur = float(s.get("duration", 0.0) or 0.0)
                max_end = max(max_end, s_start + s_dur)
            duration = max_end - offset
        else:
            duration = None
    else:
        duration = float(duration)

    # Build text list from supervisions
    text_items: List[Dict[str, Any]] = []
    for s in cut_dict.get("supervisions", []) or cut_dict.get("annotations", []) or []:
        t = s.get("text") or s.get("transcript") or s.get("label") or ""
        text_items.append(
            {
                "start": float(s.get("start", 0.0) or 0.0),
                "duration": float(s.get("duration", 0.0) or 0.0),
                "speaker": s.get("speaker") or s.get("channel") or s.get("speaker_id"),
                "text": t,
            }
        )

    # per-spk feature files from custom field
    per_spk = {}
    custom = cut_dict.get("custom") or {}
    if isinstance(custom, dict) and feature_key in custom:
        per_spk_val = custom[feature_key]
        if isinstance(per_spk_val, dict):
            per_spk = per_spk_val
        elif isinstance(per_spk_val, str):
            # single path -> attach to a generic key
            per_spk = {"all": per_spk_val}

    out: Dict[str, Any] = {"audio_filepath": audio_filepath, "duration": duration, "offset": offset, "text": text_items}
    if per_spk:
        out["per_spk_feature_files"] = per_spk

    return out


def convert(input_path: str, output_path: str, feature_key: str, use_lhotse: bool | None = None) -> int:
    """Convert and return number of records written."""
    if use_lhotse is None:
        use_lhotse = try_import_lhotse()

    if use_lhotse:
        logger.info("Using lhotse to load cuts (if available).")
        iterator = iter_cuts_with_lhotse(input_path)
    else:
        logger.info("lhotse not available; treating input as JSON/JSONL.")
        iterator = iter_cuts_from_jsonl(input_path)

    written = 0
    with open(output_path, "w", encoding="utf-8") as out_fh:
        for cut_obj, cut_dict in iterator:
            # If we used lhotse we already got cut_dict via cut.to_dict(); else it's raw
            if cut_dict is None:
                continue
            nemo_rec = build_nemo_record(cut_dict, feature_key=feature_key)
            out_fh.write(json.dumps(nemo_rec, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d records to %s", written, output_path)
    return written


def main():
    parser = argparse.ArgumentParser(description="Convert a Lhotse JSON/JSONL (CutSet) to NeMo JSONL manifest.")
    parser.add_argument("--input", "-i", required=True, help="Path to Lhotse cuts JSON/JSONL")
    parser.add_argument("--output", "-o", required=True, help="Path to output NeMo JSONL manifest")
    parser.add_argument("--feature-key", default="av_hubert_lip_features", help="custom key in cut.custom storing per-spk features")
    parser.add_argument("--no-lhotse", dest="use_lhotse", action="store_false", help="Do not try to import/use lhotse even if installed")
    args = parser.parse_args()

    try:
        convert(args.input, args.output, args.feature_key, use_lhotse=args.use_lhotse)
    except Exception as e:
        logger.exception("Conversion failed: %s", e)
        raise


if __name__ == "__main__":
    main()
