#!/bin/bash

# Activate your venv!

root=$(dirname $(dirname $0))
echo "Detected workflow_root: $root"

# ========== CONFIG

config=$root"/data_generation_configs/shd_easy.yaml"

# ========== SCRIPT

export PYTHONPATH=$root:$PYTHONPATH

python3 -B $root/sel_bench/read_dataset.py --config $config
