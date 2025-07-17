import os
import argparse

def split_rttm_by_utterance(input_rttm_path, output_dir, output_scp_path):
    """
    Splits an RTTM file into individual RTTM files by utterance ID,
    and creates an rttm.scp file listing all output file paths.

    Parameters:
    - input_rttm_path: Path to the input RTTM file.
    - output_dir: Directory to save split RTTM files.
    - output_scp_path: Path to save the rttm.scp file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    utterance_map = {}

    with open(input_rttm_path, 'r') as infile:
        for line in infile:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # skip malformed lines

            utt_id = parts[1]  # RTTM format: <type> <file-id> ...
            utt_id = utt_id.replace('/', '__')
            if utt_id not in utterance_map:
                utterance_map[utt_id] = []
            utterance_map[utt_id].append(line)

    scp_lines = []

    for utt_id, lines in utterance_map.items():
        output_path = os.path.join(output_dir, f"{utt_id}.rttm")
        with open(output_path, 'w') as outfile:
            outfile.writelines(lines)
        scp_lines.append(output_path)

    # Write the rttm.scp file
    with open(output_scp_path, 'w') as scpfile:
        for path in sorted(scp_lines):
            scpfile.write(f"{path}\n")

    print(f"✅ Split completed: {len(scp_lines)} RTTM files written.")
    print(f"📄 RTTM SCP file saved to: {output_scp_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split an RTTM file by utterance ID and generate an rttm.scp list."
    )
    parser.add_argument("input_rttm", help="Path to the input RTTM file")
    parser.add_argument("output_dir", help="Directory to save split RTTM files")
    parser.add_argument("rttm_scp", help="Path to output rttm.scp file")

    args = parser.parse_args()
    split_rttm_by_utterance(args.input_rttm, args.output_dir, args.rttm_scp)


if __name__ == "__main__":
    main()
