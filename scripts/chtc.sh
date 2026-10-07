#!/bin/bash
# Launch and collect CHTC sweeps from the laptop (needs the shared `ssh chtc` connection to be open).
#
#   scripts/chtc.sh submit <batch> <jobs.txt>   ship the committed code (git HEAD) + job list, submit one job per line
#   scripts/chtc.sh status                      queue summary
#   scripts/chtc.sh fetch <batch>               copy finished results to results/<batch>/ and unpack them
#
# jobs.txt lines: <tag> <algo: ppo|sac> <env_id> <seed> <timesteps> [extra CleanRL args...]
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=robot-recovery
cmd=${1:-}; shift || true

case "$cmd" in
submit)
    batch=$1; jobs=$2
    if [ -n "$(git status --porcelain -- robot_recovery algos)" ]; then
        echo "commit robot_recovery/ and algos/ first, so the batch is reproducible from its commit" >&2; exit 1
    fi
    tmp=$(mktemp -d)
    git archive --format=tar.gz -o "$tmp/code.tar.gz" HEAD robot_recovery algos
    ssh chtc "mkdir -p $REMOTE/chtc/$batch/logs $REMOTE/chtc/$batch/results"
    scp -q "$tmp/code.tar.gz" chtc:$REMOTE/chtc/$batch/code.tar.gz
    scp -q "$jobs" chtc:$REMOTE/chtc/$batch/jobs.txt
    git rev-parse HEAD | ssh chtc "cat > $REMOTE/chtc/$batch/COMMIT"
    ssh chtc "cd $REMOTE/chtc && condor_submit sweep.sub batch=$batch jobs_file=$batch/jobs.txt -batch-name $batch"
    rm -rf "$tmp"
    ;;
status)
    ssh chtc "condor_q -batch"
    ;;
fetch)
    batch=$1
    mkdir -p "results/$batch"
    rsync -a "chtc:$REMOTE/chtc/$batch/results/" "chtc:$REMOTE/chtc/$batch/COMMIT" "chtc:$REMOTE/chtc/$batch/jobs.txt" "results/$batch/"
    rsync -a "chtc:$REMOTE/chtc/$batch/logs/" "results/$batch/logs/"
    for f in "results/$batch"/*.tar.gz; do
        [ -e "$f" ] || continue
        tar -xzf "$f" -C "results/$batch" && rm "$f"
    done
    echo "results/$batch: $(ls "results/$batch/runs" 2>/dev/null | wc -l | tr -d ' ') runs"
    ;;
*)
    sed -n '2,9p' "$0"; exit 1
    ;;
esac
