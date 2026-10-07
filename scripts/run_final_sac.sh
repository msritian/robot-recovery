#!/bin/bash
# Final SAC runs (local, Apple M2): 5 paired seeds in parallel, 1 CPU thread each, with --capture-video.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export OMP_NUM_THREADS=1
for seed in 1 2 3 4 5; do
    nohup uv run python algos/sac_continuous_action.py --env-id AntRecoveryLowCost-v0 --seed $seed \
        --total-timesteps 1000000 --save-model --capture-video --exp-name final_sac > logs/final_sac_$seed.log 2>&1 &
done
wait
