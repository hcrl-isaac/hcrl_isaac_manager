#!/usr/bin/env bash
# Shared config + helpers for the UT LARG GPU boxes: bare-metal workstations reached over SSH, with
# training in a per-box `ilab` uv venv.

set -euo pipefail

LARG_USER="${LARG_USER:-sturman}"
LARG_DOMAIN="${LARG_DOMAIN:-cs.utexas.edu}"

# Remote path (relative to remote $HOME) where the manager tree is synced.
LARG_REMOTE_DIR="${LARG_REMOTE_DIR:-hcrl_isaac_manager}"

# Remote manager path: an absolute LARG_REMOTE_DIR as-is, a bare name relative to the remote $HOME.
larg_remote_path() {
  case "$LARG_REMOTE_DIR" in
    /*) printf '%s' "$LARG_REMOTE_DIR" ;;
    *)  printf '$HOME/%s' "$LARG_REMOTE_DIR" ;;
  esac
}

# Local manager root (this repo's parent-of-scripts).
LARG_LOCAL_DIR="${LARG_LOCAL_DIR:-$HOME/hcrl_isaac_manager}"

# A100 80GB boxes (4 GPUs each) -- primary targets.
LARG_A100_HOSTS=(mckennie hazard debruyne aaronson)
# A40 boxes (4 GPUs each) -- fallback.
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
