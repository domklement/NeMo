#!/bin/bash

# mkdir -p /tmp/librispeech

# python scripts/dataset_processing/get_librispeech_data.py --data_root /tmp/librispeech --data_sets mini --num_workers 10

conda install -c conda-forge gxx gcc -y
conda install cuda-toolkit -y
pip install meeteval
pip install -e '.[all]'
pip install cuda-python
pip install nvitop
