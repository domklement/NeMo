import os
import argparse
import json
import sox
from tqdm import tqdm

def split_rttm_by_utterance(input_rttm_path, input_wav_dir, output_dir, output_nemo_manifest):
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
    audio_map = {}
    output_manifest = []
    spk_map = {}
    # {"audio_filepath": "/scratch.ssd/dklement/job_11306482.pbs-m1/diar_data/compound_jiangyu_dataset/wavs/dev/00004.wav", "offset": 0, "duration": 27.538875, "label": "infer", "text": "-", "num_speakers": 6, "rttm_filepath": "/scratch.ssd/dklement/job_11306482.pbs-m1/diar_data/compound_jiangyu_dataset/data_ssd/dev/per_utt_rttms/00004.rttm", "uem_filepath": null, "ctm_filepath": null}

    with open(input_rttm_path, 'r') as infile:
        for line in tqdm(infile, desc="Processing RTTM file"):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # skip malformed lines

            utt_id = parts[1]  # RTTM format: <type> <file-id> ...
            audio_path = os.path.join(input_wav_dir, f"{utt_id}.wav" if not utt_id.endswith('.wav') else utt_id)
            utt_id = utt_id.replace('/', '__')
            audio_map[utt_id] = audio_path
            if utt_id not in utterance_map:
                utterance_map[utt_id] = []
            if utt_id not in spk_map:
                spk_map[utt_id] = set()
            utterance_map[utt_id].append(line)
            spk_map[utt_id].add(parts[-3])

    scp_lines = []

    for utt_id, lines in tqdm(utterance_map.items(), desc="Processing utterances"):
        output_path = os.path.join(output_dir, f"{utt_id}.rttm")
        output_manifest.append({
            "audio_filepath": audio_map[utt_id],
            "offset": 0,
            "duration": sox.file_info.duration(audio_map[utt_id]),
            "label": "infer",
            "text": "-",
            "num_speakers": len(spk_map[utt_id]),
            "rttm_filepath": output_path,
        })

        with open(output_path, 'w') as outfile:
            outfile.writelines(lines)
        scp_lines.append(output_path)

    with open(output_nemo_manifest, 'w') as outfile:
        for line in output_manifest:
            outfile.write(json.dumps(line) + "\n")

    print(f"✅ Split completed: {len(scp_lines)} RTTM files written.")
    print(f"📄 Nemo manifest file saved to: {output_nemo_manifest}")


def main():
    parser = argparse.ArgumentParser(
        description="Split an RTTM file by utterance ID and generate a nemo manifest file."
    )
    parser.add_argument("--input_rttm", required=True, help="Path to the input RTTM file")
    parser.add_argument("--input_wav_dir", required=True, help="Path to the input WAV file")
    parser.add_argument("--output_rttm_dir", required=True, help="Directory to save split RTTM files")
    parser.add_argument("--output_nemo_manifest", required=True, help="Path to output nemo manifest file")

    args = parser.parse_args()
    split_rttm_by_utterance(args.input_rttm, args.input_wav_dir, args.output_rttm_dir, args.output_nemo_manifest)


if __name__ == "__main__":
    main()
