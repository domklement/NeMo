import os
from pathlib import Path
from typing import Dict, List, Tuple
import lhotse
from lhotse import CutSet, Recording, SupervisionSet, load_manifest

def find_matching_pairs(directory: str) -> List[Tuple[str, str]]:
    """
    Find pairs of recording and supervision files in the given directory.
    Returns a list of tuples containing (recording_path, supervision_path).
    """
    files = os.listdir(directory)
    pairs = []
    
    # Group files by their base name (excluding 'recording' or 'supervision')
    file_groups: Dict[str, Dict[str, str]] = {}
    
    for file in files:
        if 'recording' in file or 'supervision' in file:
            # Get the base name by removing 'recording' or 'supervision'
            base_name = file.replace('recording', '').replace('supervision', '')
            if base_name not in file_groups:
                file_groups[base_name] = {}
            
            if 'recording' in file:
                file_groups[base_name]['recording'] = os.path.join(directory, file)
            elif 'supervision' in file:
                file_groups[base_name]['supervision'] = os.path.join(directory, file)
    
    # Create pairs from complete groups
    for base_name, files in file_groups.items():
        if 'recording' in files and 'supervision' in files:
            pairs.append((files['recording'], files['supervision']))
    
    return pairs

def create_cutsets(directory: str, output_dir: str) -> None:
    """
    Create cutsets from recording and supervision pairs in the given directory.
    Saves the cutsets to the output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find matching pairs
    pairs = find_matching_pairs(directory)
    
    if not pairs:
        print(f"No matching recording-supervision pairs found in {directory}")
        return
    
    print(f"Found {len(pairs)} matching pairs")
    
    # Process each pair
    for recording_path, supervision_path in pairs:
        try:
            # Load recording and supervision
            recording = load_manifest(recording_path)
            supervision = load_manifest(supervision_path)
            
            # Create cutset
            cuts = CutSet.from_manifests(
                recordings=recording,
                supervisions=supervision
            )
            
            # Generate output filename
            base_name = os.path.basename(recording_path).replace('recordings', 'cuts')
            output_path = os.path.join(output_dir, base_name)
            
            # Save cutset
            cuts.to_file(output_path)
            print(f"Created cutset: {output_path}")
            
        except Exception as e:
            print(f"Error processing pair {recording_path} - {supervision_path}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Lhotse cutsets from recording-supervision pairs")
    parser.add_argument("input_dir", help="Directory containing recording and supervision files")
    parser.add_argument("output_dir", help="Directory to save the created cutsets")
    
    args = parser.parse_args()
    
    create_cutsets(args.input_dir, args.output_dir) 