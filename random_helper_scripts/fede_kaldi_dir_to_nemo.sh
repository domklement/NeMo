#!/bin/bash

dset_dir=$1
old_wav_dir=$2
new_wav_dir=$3

cat $dset_dir/wav.scp | sed -e "s|${old_wav_dir}|${new_wav_dir}|g" > $dset_dir/wav.scp.fixed
mv $dset_dir/wav.scp.fixed $dset_dir/wav.scp

python /home/jovyan/NeMo/random_helper_scripts/rttm_to_nemo_manifest.py --input_rttm $dset_dir/rttm --input_wav_dir $new_wav_dir --output_rttm_dir $dset_dir/per_utt_rttms --output_nemo_manifest $dset_dir/nemo_manifest.jsonl
