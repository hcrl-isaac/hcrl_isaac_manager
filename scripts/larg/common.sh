#!/usr/bin/env bash
# Shared config + helpers for ssh GPU boxes: bare-metal workstations reached over SSH, with training in a per-box
# `ilab` uv venv. The boxes, their domain and your login come from the ssh pools in
# scripts/cluster/res/compute.local.toml (`pls setup --add <pool> --kind ssh ...`).

set -euo pipefail

LARG_INVENTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../cluster/res" && pwd)/inventory.py"
larg_inventory() { python3 "$LARG_INVENTORY" "$@"; }

# Your login on the boxes: $LARG_USER, else the one login the ssh pools name (needed only for the default paths).
LARG_USER="${LARG_USER:-$(larg_inventory ssh-user 2>/dev/null || true)}"
larg_need_user() {
  [ -n "$LARG_USER" ] && return 0
  larg_inventory ssh-user >/dev/null || true  # prints why
  echo "[larg] $1 defaults to /var/local/<login>: set LARG_USER, or $1 itself" >&2
  exit 1
}

# The manager tree on each box: per-box local disk, off the quota'd NFS home (absolute, or a name under $HOME).
if [ -z "${LARG_REMOTE_DIR:-}" ]; then
  larg_need_user LARG_REMOTE_DIR
  LARG_REMOTE_DIR="/var/local/${LARG_USER}/hcrl_isaac_manager"
fi

# Remote manager path: an absolute LARG_REMOTE_DIR as-is, a bare name relative to the remote $HOME.
larg_remote_path() {
  case "$LARG_REMOTE_DIR" in
    /*) printf '%s' "$LARG_REMOTE_DIR" ;;
    *)  printf '$HOME/%s' "$LARG_REMOTE_DIR" ;;
  esac
}

# Per-box scratch for run dirs and Kit caches.
larg_scratch() {
  if [ -z "${LARG_SCRATCH:-}" ]; then
    larg_need_user LARG_SCRATCH
    LARG_SCRATCH="/var/local/${LARG_USER}"
  fi
  printf '%s' "$LARG_SCRATCH"
}

# Local manager root (this repo's parent-of-scripts).
LARG_LOCAL_DIR="${LARG_LOCAL_DIR:-$HOME/hcrl_isaac_manager}"

# A host (a short name of an ssh pool, a full name, or user@host) -> its ssh target, from the pool's user and domain.
larg_target() {
  case "$1" in
    *@*) echo "$1" ;;
    *) larg_inventory ssh-target "$1" ;;
  esac
}

larg_ssh() {
  local host="$1" target; shift
  target="$(larg_target "$host")"
  ssh -o ConnectTimeout=10 "$target" "$@"
}
