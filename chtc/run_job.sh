#!/bin/bash
# One training run inside the container: run_job.sh <tag> <algo> <env_id> <seed> <timesteps> [extra CleanRL args...]
# Writes <tag>.tar.gz containing runs/ (TensorBoard + model) and videos/.
set -euo pipefail
tag=$1; algo=$2; env_id=$3; seed=$4; steps=$5; shift 5

tar -xzf code.tar.gz
export PYTHONPATH=$PWD:${PYTHONPATH:-}
python algos/${algo}_continuous_action.py \
    --env-id "$env_id" --seed "$seed" --total-timesteps "$steps" \
    --exp-name "$tag" --save-model --capture-video "$@"

tar -czf "${tag}__${seed}.tar.gz" runs videos
