#!/bin/bash

mkdir -p /tmp/librispeech

python scripts/dataset_processing/get_librispeech_data.py --data_root /tmp/librispeech --data_sets mini --num_workers 10
