#!/usr/bin/env python3
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Optional, Tuple

import sox
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Compute audio durations in parallel.")
    p.add_argument("tsv_path", help="Input TSV/space-delimited file with paths in the first column.")
    p.add_argument("output_path", help="Where to write JSONL with {audio_path, duration}.")
    p.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)."
    )
    p.add_argument(
        "-c", "--chunksize",
        type=int,
        default=64,
        help="Chunk size for scheduling tasks to workers (default: 64)."
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit on first error instead of skipping problematic files."
    )
    return p.parse_args()


def _duration_for_line(line: str) -> Tuple[str, Optional[float], Optional[str]]:
    """
    Worker function: given one input line, return (path, duration, error_msg).
    We return an error message instead of raising so the main process can decide what to do.
    """
    line = line.strip()
    if not line:
        return ("", None, "empty line")

    # First token (split on any whitespace) is treated as path
    path = line.split()[0]
    try:
        dur = sox.file_info.duration(path)
        if dur is None:
            return (path, None, "duration returned None")
        return (path, float(dur), None)
    except Exception as e:
        return (path, None, f"{type(e).__name__}: {e}")


def main():
    args = parse_args()

    # Read all lines; if very large, you can switch to a streaming queue approach,
    # but for most metadata lists this is fine and lets us show a total in tqdm.
    with open(args.tsv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    if total == 0:
        print("Input file is empty; nothing to do.", file=sys.stderr)
        return 0

    # Launch workers
    # Use executor.map with chunksize for efficiency; iterate results as they arrive.
    # We keep writing in the main process to avoid file-write contention.
    errors = 0
    processed = 0

    with open(args.output_path, "w", encoding="utf-8") as fout:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            # map preserves order; if you prefer faster first-complete semantics, use ex.submit/as_completed
            for path, duration, err in tqdm(
                ex.map(_duration_for_line, lines, chunksize=args.chunksize),
                total=total,
                desc="Processing",
                unit="file",
            ):
                processed += 1
                if err is not None:
                    errors += 1
                    if args.strict:
                        # Fail fast in strict mode
                        raise RuntimeError(f"Error processing {path!r}: {err}")
                    # Otherwise, just skip this entry and continue
                    continue

                fout.write(json.dumps({"audio_filepath": path, "duration": duration}) + "\n")

    if errors:
        print(f"Done with {processed} lines processed, {errors} errors skipped.", file=sys.stderr)
    else:
        print(f"Done. {processed} lines processed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
