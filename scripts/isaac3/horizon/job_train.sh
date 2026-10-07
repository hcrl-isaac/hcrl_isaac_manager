#!/bin/bash
# Inside the bookworm container on the compute node: build the venv when it is missing or older than the Isaac Lab lock,
# then run train.py on GPUS GPUs (torchrun --distributed for more than one).
# usage: job_train.sh ROOT GPUS [train.py args...]
set -euo pipefail
ROOT=$1; GPUS=$2; shift 2
D=$ROOT/resources/isaac3/bench
STAMP=$ROOT/venv/.setup-done
(
    flock 9  # one venv build at a time when several jobs start together
    if [ ! -f "$STAMP" ] || [ "$ROOT/resources/IsaacLab/uv.lock" -nt "$STAMP" ]; then
        bash "$D/remote_setup.sh" "$ROOT" && touch "$STAMP"
    fi
) 9>"$ROOT/venv.lock"
RUN=$ROOT/tmp/job-${SLURM_JOB_ID:-local}
mkdir -p "$RUN" "$ROOT/train"
export TMPDIR=$RUN XDG_CACHE_HOME=$RUN/cache WARP_CACHE_PATH=$RUN/warp
set -a; [ -f "$ROOT/.env.wandb" ] && source "$ROOT/.env.wandb"; set +a
cd "$ROOT/train"
TRAIN=$ROOT/resources/hcrl_isaaclab/scripts/train.py
if [ "$GPUS" -gt 1 ]; then
    exec "$ROOT/venv/bin/torchrun" --standalone --nproc_per_node "$GPUS" "$TRAIN" --distributed "$@"
fi
exec "$ROOT/venv/bin/python" "$TRAIN" "$@"
