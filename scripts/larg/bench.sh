#!/usr/bin/env bash
# Run the single-GPU num_envs FPS sweep (bench.py) for a task on a LARG box,
# under nohup. The sweep's best per-GPU env count feeds the train launch.
#
# Usage: scripts/larg/bench.sh <host> <task> [-- extra bench.py args]
# Poll:  scripts/larg/bench.sh --log <host> <task>

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

BENCH="resources/hcrl_isaaclab/scripts/bench.py"
RUNS="${LARG_SCRATCH:-/var/local/$LARG_USER}/larg-runs"

if [ "${1:-}" = "--log" ]; then
  shift; host="$1"; task="$2"
  larg_ssh "$host" "tail -n 30 $RUNS/bench_${task//\//_}.log 2>/dev/null; echo '--- proc ---'; pgrep -af '[b]ench.py' || echo '(no bench.py)'"
  exit 0
fi

host="$1"; task="$2"; shift 2 || true
extra=()
if [ "${1:-}" = "--" ]; then shift; extra=("$@"); fi
[ -n "${host:-}" ] && [ -n "${task:-}" ] || { echo "usage: $0 <host> <task> [-- extra]"; exit 1; }

log="$RUNS/bench_${task//\//_}.log"
q() { printf '%q ' "$@"; }
remote="mkdir -p $(q "$RUNS/tmp") && cd $(larg_remote_path) || exit 1
export PATH=\$HOME/.local/bin:\$PATH ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES TMPDIR=$(q "$RUNS/tmp")
set -a; source scripts/.env.wandb 2>/dev/null; set +a
setsid nohup ./ilab/bin/python $BENCH $(q --task "$task" "${extra[@]}") > $(q "$log") 2>&1 < /dev/null &
echo started pid \$!; echo log: $(q "$log")"

echo "=== bench $task on $host ==="
larg_ssh "$host" "bash -lc $(printf %q "$remote")"
