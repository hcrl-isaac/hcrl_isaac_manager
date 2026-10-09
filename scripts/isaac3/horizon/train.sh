#!/bin/bash
# Train on Isaac Lab 3.0 (Kit-less) on a SLURM cluster's GPUs, built for TACC Horizon's GB200s (aarch64, Newton only:
# ovphysx ships no aarch64 native library). Ships this checkout's 3.0 trees with deploy.sh, then submits one job that
# builds the venv in a python:3.12-bookworm container on its first run and trains under torchrun.
# usage: train.sh [--cluster horizon] [--gpus 4] [--time 02:00:00] [--force] [--dry-run] -- <train.py args>
#   --gpus: one node's GPUs (4 on a Horizon GB200 node; multi-node is not supported)
#   --force: deploy although an isaac3-train job is pending or running (it imports the trees this replaces)
#   e.g. train.sh --gpus 4 -- --task hcrl/T1-Velocity-v0 physics=newton_mjwarp --num_envs 32768 --logger wandb
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MANAGER=$(cd "$HERE/../../.." && pwd)
CLUSTER=horizon GPUS=4 TIME=02:00:00 DRY=0 FORCE=0
NODE_GPUS=4  # GB200 nodes
while [ $# -gt 0 ]; do
    case $1 in
        --cluster) CLUSTER=$2; shift 2 ;;
        --gpus) GPUS=$2; shift 2 ;;
        --time) TIME=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --force) FORCE=1; shift ;;
        --) shift; break ;;
        *) echo "unknown option $1 (train.py args go after --)" >&2; exit 2 ;;
    esac
done
[ "$GPUS" -ge 1 ] && [ "$GPUS" -le "$NODE_GPUS" ] 2>/dev/null ||
    { echo "--gpus must be 1-$NODE_GPUS (one node; multi-node is not supported)" >&2; exit 2; }
[ $# -gt 0 ] || { echo "no train.py args (e.g. -- --task hcrl/T1-Velocity-v0 physics=newton_mjwarp)" >&2; exit 2; }
# deploy.sh ships this checkout's resources/, which must hold the 3.0 trees (feature/isaac-3.0 everywhere)
grep -q isaaclab_newton "$MANAGER/resources/IsaacLab/pyproject.toml" 2>/dev/null ||
    { echo "$MANAGER/resources/IsaacLab is not Isaac Lab 3.0: run this from a feature/isaac-3.0 checkout" >&2; exit 1; }
CFG=$MANAGER/scripts/cluster/config/$CLUSTER
[ -f "$CFG/.env.cluster" ] || { echo "no cluster profile $CFG (just cluster add $CLUSTER)" >&2; exit 1; }
source "$CFG/.env.cluster"
# the 3.0 root sits beside the profile's 2.x workspace, e.g. /scratch/.../horizon/isaac3
ROOT=$(dirname "$CLUSTER_ISAACLAB_DIR")/isaac3
FLAGS=""
for flag in -p -A -q --reservation; do
    value=$(python3 "$MANAGER/scripts/cluster/tools/merge_profile.py" get "$CFG/submit_job_slurm.sh" "$flag")
    [ -n "$value" ] && FLAGS+=" $flag $value"
done
SSH=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm/%C" -o ControlPersist=48h -o ConnectTimeout=60)
ARGS=$(printf '%q ' "$@")
JOB="#!/bin/bash
#SBATCH -J isaac3-train
#SBATCH -N 1
#SBATCH -t $TIME
#SBATCH -o $ROOT/train/slurm-%j.log
${CLUSTER_MODULE_LOAD:+module load $CLUSTER_MODULE_LOAD}
bash $ROOT/resources/isaac3/bench/container_run.sh $ROOT $ROOT/resources/isaac3/horizon/job_train.sh $ROOT $GPUS $ARGS"
if [ "$DRY" = 1 ]; then
    echo "[isaac3] would deploy to $CLUSTER_LOGIN:$ROOT and submit (sbatch$FLAGS):"
    echo "$JOB"
    exit 0
fi
# one deploy root, installed editable into the venv: deploying under a queued, starting, running or suspended job swaps
# its code. Only a COMPLETING job, which no longer imports the trees, is left out.
busy=$(ssh "${SSH[@]}" "$CLUSTER_LOGIN" "squeue -h -u \$USER -n isaac3-train -t PENDING,CONFIGURING,RUNNING,SUSPENDED,REQUEUED -o '%i %T'")
if [ -n "$busy" ] && [ "$FORCE" = 0 ]; then
    echo "[isaac3] not deploying: isaac3-train jobs import the trees at $ROOT and would get this checkout's code:" >&2
    echo "$busy" | sed 's/^/  /' >&2
    echo "  wait for them, or pass --force to replace their code anyway" >&2
    exit 1
fi
SSH_OPTS="${SSH[*]}" RSYNC_RSH="ssh ${SSH[*]}" bash "$MANAGER/scripts/isaac3/bench/deploy.sh" "$CLUSTER_LOGIN" "$ROOT"
# W&B credentials travel as a file, never in argv
[ -f "$MANAGER/scripts/.env.wandb" ] && rsync -q --chmod=F600 -e "ssh ${SSH[*]}" "$MANAGER/scripts/.env.wandb" \
    "$CLUSTER_LOGIN:$ROOT/.env.wandb"
ssh "${SSH[@]}" "$CLUSTER_LOGIN" "mkdir -p $ROOT/train; [ ! -f $ROOT/.env.wandb ] || chmod 600 $ROOT/.env.wandb"
id=$(printf '%s\n' "$JOB" | ssh "${SSH[@]}" "$CLUSTER_LOGIN" "sbatch --parsable$FLAGS" | tail -1)
echo "[isaac3] submitted job $id on $CLUSTER ($GPUS GPU)"
iters=$(printf '%s\n' "$@" | sed -n '/^--max_iterations$/{n;p}' | head -1)
echo "  log:      ssh $CLUSTER_LOGIN tail -f $ROOT/train/slurm-$id.log"
echo "  watchdog: scripts/tools/watch_run.sh isaac3-$id $CLUSTER_LOGIN:$ROOT/train/slurm-$id.log ${iters:-<target-iters>}"
