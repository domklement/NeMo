import argparse

def count_unique_speakers(rttm_path):
    speakers = set()

    with open(rttm_path, 'r') as f:
        for line in f:
            if not line.startswith("SPEAKER"):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                speaker_id = parts[7]
                speakers.add(speaker_id)

    return speakers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count unique speakers in an RTTM file.")
    parser.add_argument("rttm_file", help="Path to the RTTM file.")

    args = parser.parse_args()

    speakers = count_unique_speakers(args.rttm_file)
    # for spk in sorted(speakers):
    #     print(f" - {spk}")
    print(f"✅ Found {len(speakers)} unique speaker(s):")
