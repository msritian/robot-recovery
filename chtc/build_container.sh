#!/bin/bash
# Runs on a CHTC build node (see build.sub). Builds the dependency image WITHOUT fakeroot
# (definition-file builds fail on CHTC build nodes in batch jobs):
#   1. pull python:3.10-slim into a user-owned sandbox  2. pip install pinned deps into it  3. pack a .sif
# Only dependencies go in the image; repo code ships with each job (chtc/run_job.sh).
# No OSMesa (needs apt as root), so jobs train headless; videos are recorded locally.
set -euo pipefail
OUT=${1:?output .sif name}
SB="$_CONDOR_SCRATCH_DIR/sb"
export APPTAINER_TMPDIR="$_CONDOR_SCRATCH_DIR/apptmp" APPTAINER_CACHEDIR="$_CONDOR_SCRATCH_DIR/aptcache"
export HOME="$_CONDOR_SCRATCH_DIR/home"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$HOME"
echo "[build] host=$(hostname) start=$(date -Is) apptainer=$(apptainer --version)"

apptainer build --sandbox "$SB" docker://python:3.10-slim

# versions match pyproject.toml; CPU-only torch keeps the image small
apptainer exec --writable --no-home "$SB" pip install --no-cache-dir \
    torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
apptainer exec --writable --no-home "$SB" pip install --no-cache-dir \
    gymnasium==0.29.1 mujoco==2.3.3 numpy==1.24.4 "tensorboard>=2.10,<2.16" tyro==0.5.10 \
    moviepy==1.0.3 "imageio>=2.14.1,<3" "pandas>=2.0,<2.3" "matplotlib>=3.7,<3.10" "scipy>=1.10,<1.14"
cat > "$SB/.singularity.d/env/90-robot-recovery.sh" <<'EOF'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
EOF
apptainer exec --no-home "$SB" python -c "import torch, mujoco, gymnasium; print('[build] ok', torch.__version__, mujoco.__version__, gymnasium.__version__)"
apptainer exec --no-home "$SB" pip freeze > pip-freeze.txt

apptainer build "$OUT" "$SB"
ls -la "$OUT"
