#!/bin/bash 

lhotse workflows simulate-meetings  \
    --method conversational \
    --num-speakers-per-meeting 2 \
    --num-meetings 1000 \
    --num-jobs 8 \
    --seed 42 \
    --allow-3fold-overlap \
    /tmp/librispeech/manifests/librispeech_cuts_train-clean-100.jsonl.gz \
    /tmp/librispeech/ls_sim_conv/sim_convs_2spk_ls_clean_100_1k_meetings.jsonl.gz