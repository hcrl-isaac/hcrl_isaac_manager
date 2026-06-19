#!/usr/bin/env bash
# Run the single-GPU num_envs FPS sweep (bench.py) for a task on a LARG box,
# under nohup. The sweep's best per-GPU env count feeds the train launch.
#
# Usage: scripts/larg/bench.sh <host> <task> [-- extra bench.py args]
# Poll:  scripts/larg/bench.sh --log <host> <task>

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

ILAB_REL="resources/IsaacLab"
PY="$ILAB_REL/ilab/bin/python"
BENCH="source/hcrl_isaaclab/scripts/bench.py"

if [ "${1:-}" = "--log" ]; then
  shift; host="$1"; task="$2"
  larg_ssh "$host" "tail -n 30 \$HOME/larg_bench_${task}.log 2>/dev/null; echo '--- proc ---'; pgrep -af bench.py || echo '(no bench.py)'"
  exit 0
fi

host="$1"; task="$2"; shift 2 || true
extra=()
if [ "${1:-}" = "--" ]; then shift; extra=("$@"); fi
[ -n "${host:-}" ] && [ -n "${task:-}" ] || { echo "usage: $0 <host> <task> [-- extra]"; exit 1; }

log="\$HOME/larg_bench_${task}.log"
remote_cmd="cd \$HOME/$LARG_REMOTE_DIR/$ILAB_REL && \
  export PATH=\$HOME/.local/bin:\$PATH ACCEPT_EULA=Y && \
  set -a; source \$HOME/$LARG_REMOTE_DIR/scripts/.env.wandb 2>/dev/null; set +a; \
  nohup ./ilab/bin/python $BENCH --task $task ${extra[*]} > $log 2>&1 & \
  echo started pid \$!; echo log: $log"

echo "=== bench $task on $host ==="
larg_ssh "$host" "bash -lc '$remote_cmd'"
