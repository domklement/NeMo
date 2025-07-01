#!/bin/bash

num_workers=50
NUM_MEETINGS=100000
NUM_MEETINGS_PER_WORKER=$((NUM_MEETINGS / num_workers))

for i in $(seq 0 $((num_workers-1))); do
    # Recompute num_from and num_to based on i and NUM_MEETINGS_PER_WORKER
    num_from=$((i * NUM_MEETINGS_PER_WORKER))
    num_to=$(((i + 1) * NUM_MEETINGS_PER_WORKER))
    python /storage/brno12-cerit/home/dklement/speech/ASR/NeMo/random_helper_scripts/sim_data_samuele.py $num_from $num_to &
done

wait
