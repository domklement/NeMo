import sys
import sox
from tqdm import tqdm

tsv_path = sys.argv[1]
output_path = sys.argv[2]

with open(tsv_path, "r") as f:
    with open(output_path, "w") as f_out:
        for line in tqdm(f):
            path = line.strip().split()[0]
            duration = sox.file_info.duration(path)
            f_out.write(f'{"audio_path": "{path}", "duration": {duration}}\n')
