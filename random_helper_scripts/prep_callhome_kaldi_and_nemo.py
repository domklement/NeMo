import os
import sys

import sox

rec_list_path = sys.argv[1]
out_dir_path = os.path.dirname(rec_list_path)
rttm_path = sys.argv[2]
wav_path = sys.argv[2]

with open(rec_list_path, 'r') as f:
    rec_list = f.readlines()

for rec in rec_list:
    rec = rec.strip()
    rttm_file = os.path.join(rttm_path, f"{rec}.rttm")
    wav_file = os.path.join(wav_path, f"{rec}.wav")
    