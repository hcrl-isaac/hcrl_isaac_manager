#!/usr/bin/env bash
# Shared config + helpers for deploying to the UT LARG GPU boxes.
#
# These are bare-metal workstations (no SLURM, no containers) reachable directly
# over SSH from this device. We run training in a per-box `ilab` uv venv that
# mirrors the local one (Isaac Lab installed via the justfile `setup` flow).

set -euo pipefail

LARG_USER="${LARG_USER:-sturman}"
LARG_DOMAIN="${LARG_DOMAIN:-cs.utexas.edu}"

# Remote path (relative to remote $HOME) where the manager tree is synced.
LARG_REMOTE_DIR="${LARG_REMOTE_DIR:-hcrl_isaac_manager}"

# Local manager root (this repo's parent-of-scripts).
LARG_LOCAL_DIR="${LARG_LOCAL_DIR:-$HOME/hcrl_isaac_manager}"

# A100 80GB boxes (4 GPUs each) — primary targets.
LARG_A100_HOSTS=(mckennie hazard debruyne aaronson)
# A40 boxes (4 GPUs each) — fallback.
LARG_A40_HOSTS=(pepi pulisic salah pogba)

# Resolve a short host name (mckennie) -> full ssh target (sturman@mckennie.cs.utexas.edu).
larg_target() {
  local h="$1"
  case "$h" in
    *@*) echo "$h" ;;
    *.*) echo "${LARG_USER}@${h}" ;;
    *)   echo "${LARG_USER}@${h}.${LARG_DOMAIN}" ;;
  esac
}

larg_ssh() {
  local host="$1"; shift
  ssh -o ConnectTimeout=10 "$(larg_target "$host")" "$@"
}
