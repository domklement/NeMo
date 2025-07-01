import json
import os
from pathlib import Path
from lhotse import CutSet
from collections import defaultdict
from tqdm import tqdm

def build_nemo_diarization_manifest(cutset: CutSet, rttm_output_dir: str):
    os.makedirs(rttm_output_dir, exist_ok=True)
    nemo_manifest = []

    for cut in tqdm(cutset, desc="Processing Cuts"):
        if not cut.supervisions:
            continue

        audio_path = cut.recording.sources[0].source if cut.recording else cut.audio_path
        recording_id = cut.id
        duration = cut.duration
        supervisions = cut.supervisions

        speakers = set(sup.speaker for sup in supervisions if sup.speaker is not None)
        rttm_filename = f"{recording_id}.rttm"
        rttm_path = os.path.join(rttm_output_dir, rttm_filename)

        with open(rttm_path, 'w') as rttm_file:
            for sup in supervisions:
                if sup.speaker is None:
                    continue
                rttm_line = (
                    f"SPEAKER {recording_id} 1 {sup.start:.3f} "
                    f"{sup.duration:.3f} <NA> <NA> {sup.speaker} <NA> <NA>\n"
                )
                rttm_file.write(rttm_line)

        nemo_entry = {
            "audio_filepath": str(audio_path),
            "offset": 0.0,
            "duration": round(duration, 3),
            "label": "infer",
            "text": "-",
            "num_speakers": len(speakers),
            "rttm_filepath": rttm_path
        }
        nemo_manifest.append(nemo_entry)

    return nemo_manifest

def save_nemo_manifest(nemo_manifest, output_path):
    with open(output_path, 'w') as f:
        for entry in nemo_manifest:
            json.dump(entry, f)
            f.write('\n')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Lhotse CutSet to NeMo diarization manifest.")
    parser.add_argument("--cutset", required=True, help="Path to Lhotse CutSet JSON file.")
    parser.add_argument("--rttm_output_dir", required=True, help="Directory to save RTTM files.")
    parser.add_argument("--nemo_manifest_output", required=True, help="Output NeMo manifest JSONL file.")

    args = parser.parse_args()

    cutset = CutSet.from_file(args.cutset)
    nemo_manifest = build_nemo_diarization_manifest(cutset, args.rttm_output_dir)
    save_nemo_manifest(nemo_manifest, args.nemo_manifest_output)

    print(f"✅ Processed {len(nemo_manifest)} cuts and saved NeMo manifest to: {args.nemo_manifest_output}")
