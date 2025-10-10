#!/usr/bin/env python3
"""
Script to split NeMo manifest JSON file into n-second segments.

This script reads a NeMo manifest file where each line contains a JSON object with audio metadata,
and creates a new manifest file where each original segment is split into smaller n-second segments.
The audio_filepath and rttm_filepath remain the same, but offset and duration are updated for each segment.
The num_speakers field is updated to reflect the actual number of speakers present in each segment
by parsing the corresponding RTTM file.

Example input manifest line:
{"audio_filepath": "/tmp/diar_data/compound_jiangyu_dataset/wavs/train/zyffh.wav", "offset": 0, "duration": 252.072, "label": "infer", "text": "-", "num_speakers": 3, "rttm_filepath": "/tmp/diar_data/compound_jiangyu_dataset/data_ssd/train/per_utt_rttms/zyffh.rttm", "uem_filepath": null, "ctm_filepath": null}

Example output for 30-second segments:
{"audio_filepath": "/tmp/diar_data/compound_jiangyu_dataset/wavs/train/zyffh.wav", "offset": 0, "duration": 30.0, "label": "infer", "text": "-", "num_speakers": 2, "rttm_filepath": "/tmp/diar_data/compound_jiangyu_dataset/data_ssd/train/per_utt_rttms/zyffh.rttm", "uem_filepath": null, "ctm_filepath": null}
{"audio_filepath": "/tmp/diar_data/compound_jiangyu_dataset/wavs/train/zyffh.wav", "offset": 30.0, "duration": 30.0, "label": "infer", "text": "-", "num_speakers": 3, "rttm_filepath": "/tmp/diar_data/compound_jiangyu_dataset/data_ssd/train/per_utt_rttms/zyffh.rttm", "uem_filepath": null, "ctm_filepath": null}
...
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Set


def convert_rttm_line(rttm_line: str, round_digits: int = 3) -> tuple:
    """
    Convert a line in RTTM file to speaker label, start and end timestamps.
    
    Args:
        rttm_line: A line in RTTM formatted file containing offset and duration of each segment.
        round_digits: Number of digits to be rounded.
        
    Returns:
        start: Start timestamp in floating point number.
        end: End timestamp in floating point number.
        speaker: Speaker string in RTTM lines.
    """
    rttm = rttm_line.strip().split()
    start = round(float(rttm[3]), round_digits)
    end = round(float(rttm[4]), round_digits) + round(float(rttm[3]), round_digits)
    speaker = rttm[7]
    return start, end, speaker


def count_speakers_in_segment(rttm_filepath: str, segment_offset: float, segment_duration: float) -> int:
    """
    Count the number of unique speakers present in a given segment by parsing the RTTM file.
    
    Args:
        rttm_filepath: Path to the RTTM file
        segment_offset: Start time of the segment
        segment_duration: Duration of the segment
        
    Returns:
        Number of unique speakers in the segment
    """
    if not rttm_filepath or not os.path.exists(rttm_filepath):
        return 0
    
    segment_start = segment_offset
    segment_end = segment_offset + segment_duration
    speakers_in_segment: Set[str] = set()
    
    try:
        with open(rttm_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    start, end, speaker = convert_rttm_line(line)
                    
                    # Skip invalid RTTM lines where the start time is greater than the end time
                    if start > end:
                        continue
                    
                    # Check if the RTTM segment overlaps with the specified segment
                    if (end > segment_start and start < segment_end) or (start < segment_end and end > segment_start):
                        speakers_in_segment.add(speaker)
                        
                except (IndexError, ValueError) as e:
                    print(f"Warning: Invalid RTTM line format: {line} - {e}", file=sys.stderr)
                    continue
                    
    except Exception as e:
        print(f"Warning: Could not read RTTM file {rttm_filepath}: {e}", file=sys.stderr)
        return 0
    
    return len(speakers_in_segment)


def split_manifest_segment(manifest_line: Dict[str, Any], segment_duration: float, remove_silent_segments: bool = False, min_duration: float = None, max_speakers: int = 4) -> List[Dict[str, Any]]:
    """
    Split a single manifest line into multiple segments of specified duration.
    
    Args:
        manifest_line: Dictionary containing manifest entry
        segment_duration: Duration of each segment in seconds
        
    Returns:
        List of manifest entries, one for each segment
    """
    original_offset = manifest_line.get('offset', 0)
    original_duration = manifest_line['duration']
    rttm_filepath = manifest_line.get('rttm_filepath')
    
    # Calculate how many segments we can create
    start_time = original_offset
    end_time = original_offset + original_duration
    
    segments = []
    current_offset = start_time
    
    while current_offset < end_time:
        # Calculate duration for this segment
        remaining_duration = end_time - current_offset
        segment_dur = min(segment_duration, remaining_duration)
        
        # Count speakers in this segment
        num_speakers = count_speakers_in_segment(rttm_filepath, current_offset, segment_dur)

        if min_duration and segment_dur < min_duration:
            current_offset += segment_duration
            continue

        if max_speakers and num_speakers > max_speakers:
            current_offset += segment_duration
            continue

        if remove_silent_segments and num_speakers == 0:
            current_offset += segment_duration
            continue
        
        # Create new manifest entry for this segment
        segment_entry = manifest_line.copy()
        segment_entry['offset'] = current_offset
        segment_entry['duration'] = segment_dur
        segment_entry['num_speakers'] = num_speakers
        
        segments.append(segment_entry)
        
        # Move to next segment
        current_offset += segment_duration
    
    return segments


def process_manifest_file(input_manifest: str, output_manifest: str, segment_duration: float, remove_silent_segments: bool = False, min_duration: float = None, max_speakers: int = 4) -> None:
    """
    Process the entire manifest file and create segmented version.
    
    Args:
        input_manifest: Path to input manifest file
        output_manifest: Path to output manifest file
        segment_duration: Duration of each segment in seconds
    """
    if not os.path.exists(input_manifest):
        raise FileNotFoundError(f"Input manifest file not found: {input_manifest}")
    
    total_segments = 0
    total_original_entries = 0
    
    with open(input_manifest, 'r', encoding='utf-8') as infile, \
         open(output_manifest, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse JSON line
                manifest_entry = json.loads(line)
                total_original_entries += 1
                
                # Split this entry into segments
                segments = split_manifest_segment(manifest_entry, segment_duration, remove_silent_segments, min_duration, max_speakers)
                
                # Write each segment to output file
                for segment in segments:
                    json.dump(segment, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    total_segments += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
                continue
            except KeyError as e:
                print(f"Warning: Missing required field on line {line_num}: {e}", file=sys.stderr)
                continue
    
    print(f"Processing complete!")
    print(f"Original entries: {total_original_entries}")
    print(f"Total segments created: {total_segments}")
    print(f"Output saved to: {output_manifest}")


def main():
    parser = argparse.ArgumentParser(
        description="Split NeMo manifest JSON file into n-second segments with actual speaker counts from RTTM files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split manifest into 30-second segments
  python create_segments.py --input manifest.json --output manifest_30s.json --segment_duration 30
  
  # Split manifest into 1-minute segments
  python create_segments.py --input manifest.json --output manifest_60s.json --segment_duration 60
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input manifest file"
    )
    
    parser.add_argument(
        "--output", "-o", 
        required=True,
        help="Path to output manifest file"
    )
    
    parser.add_argument(
        "--segment_duration", "-d",
        type=float,
        required=True,
        help="Duration of each segment in seconds"
    )

    parser.add_argument(
        "--min_duration", "-m",
        type=float,
        required=False,
        default=1.0,
        help="Minimum duration of each segment in seconds"
    )

    # max speakers
    parser.add_argument(
        "--max_speakers", "-s",
        type=int,
        required=False,
        default=100,
        help="Maximum number of speakers in each segment"
    )

    parser.add_argument('--remove_silent_segments', action='store_true', help='Remove segments with no speakers')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.segment_duration <= 0:
        print("Error: segment_duration must be positive", file=sys.stderr)
        sys.exit(1)
    
    try:
        process_manifest_file(args.input, args.output, args.segment_duration, args.remove_silent_segments, args.min_duration, args.max_speakers)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
