#!/usr/bin/env bash
# Launch a single-node multi-GPU torchrun training job on a LARG box, under nohup.
#
# A100s lack RT cores, so we pass --server -> train.py defers video rendering to
# the async logger (run scripts/larg/video_logger.sh on an RT-capable box). The
# run is W&B-logged; other flags pass via `--`.
#
# Usage:
#   scripts/larg/train.sh <host> <task> <run_name> [run_group] [num_envs] [-- extra train.py args]
# Poll:
#   scripts/larg/train.sh --log <host> <task>

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

ILAB_REL="resources/IsaacLab"
TRAIN="source/hcrl_isaaclab/scripts/train.py"
NPROC="${LARG_NPROC:-4}"

if [ "${1:-}" = "--log" ]; then
  shift; host="$1"; task="$2"
  larg_ssh "$host" "ls -t \$HOME/larg_train_${task}_*.log 2>/dev/null | head -1 | xargs -r tail -n 40; echo '--- proc ---'; pgrep -af 'torch.distributed.run|train.py' | head || echo '(no train proc)'; echo '--- gpu ---'; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
  exit 0
fi

host="$1"; task="$2"; run_name="$3"; run_group="${4:-larg}"; num_envs="${5:-}"
shift $(( $# < 5 ? $# : 5 )) || true
extra=()
if [ "${1:-}" = "--" ]; then shift; extra=("$@"); fi
[ -n "${host:-}" ] && [ -n "${task:-}" ] && [ -n "${run_name:-}" ] || {
  echo "usage: $0 <host> <task> <run_name> [run_group] [num_envs] [-- extra]"; exit 1; }

envs_arg=""
[ -n "$num_envs" ] && envs_arg="--num_envs $num_envs"

ts="\$(date +%Y%m%d-%H%M%S)"
log="\$HOME/larg_train_${task}_${ts}.log"
remote_cmd="cd \$HOME/$LARG_REMOTE_DIR/$ILAB_REL && \
  export PATH=\$HOME/.local/bin:\$PATH ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES && \
  set -a; source \$HOME/$LARG_REMOTE_DIR/scripts/.env.wandb 2>/dev/null; set +a; \
  setsid ./ilab/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NPROC \
    $TRAIN --distributed --server --task $task \
    --run_name $run_name --run_group $run_group $envs_arg ${extra[*]} \
    > $log 2>&1 < /dev/null & \
  echo started pid \$!; echo log: $log"

echo "=== train $task ($run_name) on $host: ${NPROC}xGPU ==="
larg_ssh "$host" "bash -lc '$remote_cmd'"
