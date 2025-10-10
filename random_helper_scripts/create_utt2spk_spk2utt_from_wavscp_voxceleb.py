# Load wav.scp, and create utt2spk and spk2utt
from collections import defaultdict
from tqdm import tqdm
import sys

wav_scp_path = sys.argv[1]
utt2spk_path = wav_scp_path.replace('wav.scp', 'utt2spk')
spk2utt_path = wav_scp_path.replace('wav.scp', 'spk2utt')

utt2spk = []
spk2utt_dict = defaultdict(list)

with open(wav_scp_path, 'r') as f:
    for line in tqdm(f):
        utt_id, path = line.strip().split()
        parts = path.split('/')
        if len(parts) < 8:
            continue  # skip malformed paths
        speaker = parts[-3]           # e.g., id03750
        youtube_id = parts[-2]        # e.g., NdfdGn3pW8M
        filename = parts[-1].split('.')[0]  # e.g., 00210

        utt2spk.append(f"{utt_id} {speaker}")
        spk2utt_dict[speaker].append(utt_id)

# Write utt2spk
with open(utt2spk_path, 'w') as f:
    for line in utt2spk:
        f.write(line + '\n')

# Write spk2utt
with open(spk2utt_path, 'w') as f:
    for spk, utts in spk2utt_dict.items():
        f.write(f"{spk} {' '.join(utts)}\n")
