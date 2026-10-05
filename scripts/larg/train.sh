#!/usr/bin/env bash
# Launch a training run on a LARG box from its synced workspace, detached (setsid nohup).
#
# Passes --video async: the A100s lack RT cores, so videos render elsewhere. Extra train.py flags pass via `--`.
# One GPU runs train.py directly; LARG_NPROC > 1 runs it under torchrun with --distributed.
#
# Usage:
#   scripts/larg/train.sh <host> <task> <run_name> [run_group] [num_envs] [-- extra train.py args]
# Poll:
#   scripts/larg/train.sh --log <host> <task>
# Env:
#   LARG_NPROC            GPUs for the run (default 1)
#   CUDA_VISIBLE_DEVICES  physical GPUs to pin the run to (e.g. 2 or 0,1); also tags the run dir
#   LARG_SCRATCH          per-box scratch for run logs and Kit caches (default /var/local/$LARG_USER)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

TRAIN="resources/hcrl_isaaclab/scripts/train.py"
NPROC="${LARG_NPROC:-1}"
RUNS="${LARG_SCRATCH:-/var/local/$LARG_USER}/larg-runs"

if [ "${1:-}" = "--log" ]; then
  shift; host="$1"; task="$2"
  larg_ssh "$host" "ls -td $RUNS/*${task//\//_}* 2>/dev/null | head -1 | xargs -r -I{} tail -n 40 {}/train.log; \
    echo '--- proc ---'; pgrep -af '[t]orch.distributed.run|[t]rain.py' | head || echo '(no train proc)'; \
    echo '--- gpu ---'; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
  exit 0
fi

host="${1:-}"; task="${2:-}"; run_name="${3:-}"; run_group="${4:-larg}"; num_envs="${5:-}"
shift $(( $# < 5 ? $# : 5 )) || true
extra=()
if [ "${1:-}" = "--" ]; then shift; extra=("$@"); fi
[ -n "$host" ] && [ -n "$task" ] && [ -n "$run_name" ] || {
  echo "usage: $0 <host> <task> <run_name> [run_group] [num_envs] [-- extra]"; exit 1; }

args=(--video async --task "$task" --run_name "$run_name" --run_group "$run_group")
[ -n "$num_envs" ] && args+=(--num_envs "$num_envs")
args+=("${extra[@]}")
if [ "$NPROC" -gt 1 ]; then
  launch=(./ilab/bin/python -m torch.distributed.run --standalone --nnodes=1 "--nproc_per_node=$NPROC" "$TRAIN" --distributed)
else
  launch=(./ilab/bin/python "$TRAIN")
fi

gpus="${CUDA_VISIBLE_DEVICES:-}"
tag="${task//\//_}_$(date +%Y%m%d-%H%M%S)${gpus:+_gpu${gpus//,/-}}"
run_dir="$RUNS/$tag"
# per run: TMPDIR and the log; per GPU set: Kit caches, so concurrent runs on one box never share one
cache="${RUNS%/*}/kit-cache/gpu${gpus:-all}"
q() { printf '%q ' "$@"; }
remote="set -u
mkdir -p $(q "$run_dir/tmp" "$cache/xdg" "$cache/omni") || exit 1
cd $(larg_remote_path) || { echo 'no workspace at $(larg_remote_path)'; exit 1; }
export PATH=\$HOME/.local/bin:\$PATH ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export TMPDIR=$(q "$run_dir/tmp") XDG_CACHE_HOME=$(q "$cache/xdg") OMNI_CACHE_DIR=$(q "$cache/omni")
${gpus:+export CUDA_VISIBLE_DEVICES=$(q "$gpus")}
gomp=\"\$(ls ilab/lib/python3.11/site-packages/torch/lib/libgomp-*.so.1 2>/dev/null | head -1)\"
[ -n \"\$gomp\" ] && export LD_PRELOAD=\"\$gomp\${LD_PRELOAD:+:\$LD_PRELOAD}\"
set -a; source scripts/.env.wandb 2>/dev/null; set +a
setsid nohup $(q "${launch[@]}" "${args[@]}") > $(q "$run_dir/train.log") 2>&1 < /dev/null &
echo started pid \$!; echo log: $(q "$run_dir/train.log")"

echo "=== train $task ($run_name) on $host: ${NPROC} GPU(s)${gpus:+ [$gpus]} ==="
larg_ssh "$host" "bash -lc $(printf %q "$remote")"
