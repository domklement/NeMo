#!/bin/bash

num_workers=50
NUM_MEETINGS=100000
NUM_MEETINGS_PER_WORKER=$((NUM_MEETINGS / num_workers))
MAX_DURATION=600

# TARGET_DIR="/scratch.ssd/dklement/job_11306482.pbs-m1/diar_data/sim_emilia_100k_meetings_600s"
# SRC_MANIFEST="/scratch.ssd/dklement/job_11306482.pbs-m1/emilia_data/emilia_subset_20utt_50kspks_en_only/manifests/emilia-train-ihm-cutset_aligned_fixed_sources_min_dnsmos_3.3_2.jsonl"

TARGET_DIR="/scratch.ssd/dklement/job_11306482.pbs-m1/diar_data/sim_ls_100k_meetings_600s"
SRC_MANIFEST="/scratch.ssd/dklement/job_11306482.pbs-m1/data/librispeech/librispeech_manifests/librispeech_cuts_train-960h.jsonl.gz"

mkdir -p $TARGET_DIR

if [ ! -f $TARGET_DIR/manifests/all_cuts_splitted.jsonl.gz ]; then
    mkdir -p $TARGET_DIR/manifests
    python /storage/brno12-cerit/home/dklement/speech/ASR/NeMo/random_helper_scripts/sim_data_split_aligs.py $SRC_MANIFEST $TARGET_DIR/manifests/all_cuts_splitted.jsonl.gz
fi

echo "Simulating $NUM_MEETINGS meetings with max duration $MAX_DURATION seconds"

for i in $(seq 0 $((num_workers-1))); do
    # Recompute num_from and num_to based on i and NUM_MEETINGS_PER_WORKER
    num_from=$((i * NUM_MEETINGS_PER_WORKER))
    num_to=$(((i + 1) * NUM_MEETINGS_PER_WORKER))
    python /storage/brno12-cerit/home/dklement/speech/ASR/NeMo/random_helper_scripts/sim_data_samuele_general.py $num_from $num_to $TARGET_DIR $MAX_DURATION &
done

wait
