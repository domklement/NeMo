#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from lhotse import CutSet
from tqdm import tqdm

def process_cut(cut):
    """Process a single cut and create a JSON file with the required format."""
    # Create the base dictionary with audio information
    cut_dict = {
        "audio_filepath": str(cut.recording.sources[0].source),
        "duration": float(cut.duration),
        "offset": float(cut.start),
        "text": []
    }
    
    # Add supervision information
    for supervision in cut.supervisions:
        text_dict = {
            "start": float(supervision.start),
            "duration": float(supervision.duration),
            "speaker": supervision.speaker,
            "text": supervision.text
        }
        cut_dict["text"].append(text_dict)

    return cut_dict

def main():
    parser = argparse.ArgumentParser(description='Convert Lhotse CutSet to individual JSON files')
    parser.add_argument('input_manifest', type=str, help='Path to input Lhotse manifest file')
    parser.add_argument('output_manifest', type=str, help='Path to output JSON file')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_manifest = Path(args.output_manifest)
    
    # Load the CutSet
    cuts = CutSet.from_file(args.input_manifest)
    
    # Process each cut
    with open(output_manifest, 'w', encoding='utf-8') as f:
        for cut in tqdm(cuts):
            f.write(json.dumps(process_cut(cut)) + "\n")

if __name__ == "__main__":
    main()
