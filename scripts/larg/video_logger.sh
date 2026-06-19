#!/usr/bin/env bash
# Run the async video logger on a local RT-core capable box for
# LARG A100 runs. The A100 training jobs run with --server: they log rollout
# states to W&B and tag the run for async video. This pulls those states,
# renders the video locally, and uploads it back to the W&B run.
#
# Usage:
#   scripts/larg/video_logger.sh <task> [wandb_project]            # one pass
#   scripts/larg/video_logger.sh --loop [secs] <task> [project]    # repeat every <secs> (default 1800)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

MANAGER_DIR="${LARG_LOCAL_DIR}"
SCRIPTS_DIR="$MANAGER_DIR/resources/IsaacLab/source/hcrl_isaaclab/scripts"
VENV_PY="$MANAGER_DIR/resources/IsaacLab/ilab/bin/python"
ENV_WANDB="$MANAGER_DIR/scripts/.env.wandb"
DEFAULT_PROJECT="sturman-university-of-texas-at-austin/G1_Meta_LocoManipulation"

loop=0; interval=1800
if [ "${1:-}" = "--loop" ]; then loop=1; shift; case "${1:-}" in ''|*[!0-9]*) ;; *) interval="$1"; shift;; esac; fi

task="$1"; project="${2:-$DEFAULT_PROJECT}"
[ -n "${task:-}" ] || { echo "usage: $0 [--loop [secs]] <task> [wandb_project]"; exit 1; }

# Call video_logger.py directly with absolute paths. Skip a pass if the
# local render GPU is busy (>50%), mirroring the wrapper's guard.
run_once() {
  local util; util=$(nvidia-smi --query-gpu=utilization.gpu --id=0 --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  if [ -n "$util" ] && [ "$util" -gt 50 ]; then echo "[video] local GPU ${util}% busy; skipping pass"; return 0; fi
  # --max_runs_per_sweep 1: record ONE source run per process then exit (this loop restarts it)
  ( set -a; . "$ENV_WANDB"; set +a
    cd "$SCRIPTS_DIR" && "$VENV_PY" -u video_logger.py --mode async --task "$task" \
      --wandb_project "$project" --max_runs_per_sweep 1 )
}

if [ "$loop" = "1" ]; then
  echo "[video] looping every ${interval}s for task=$task project=$project"
  while true; do run_once || echo "[video] pass failed (GPU busy?), retrying next cycle"; sleep "$interval"; done
else
  run_once
fi
