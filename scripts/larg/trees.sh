#!/usr/bin/env bash
# Branch-pinned code trees on a LARG box, as `develop stage` makes them on a cluster (scripts/cluster/cluster_dev/trees.sh):
# named repos at named refs or local worktrees, every other repo a link to the box's synced workspace. A tree lives on the
# box's local disk at <workspace>/trees/<name>-<fingerprint>/, and `train.sh --tree <name>` runs from it.
#
# Usage:
#   scripts/larg/trees.sh stage <host> [--no-space-check] <name> <repo>=<ref|/path/to/worktree> ...
#   scripts/larg/trees.sh list <host>
#   scripts/larg/trees.sh rm <host> <name>-<fingerprint> [--force] | --partials

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

cmd="${1:-}"; host="${2:-}"
[ -n "$cmd" ] && [ -n "$host" ] || { sed -n '7,9p' "$0" | sed 's/^# //'; exit 1; }
shift 2

# what trees.sh expects of its host (cluster_dev.sh), for a LARG box
CLUSTER_LOGIN="$(larg_target "$host")"
SSH_OPTS=(-o ConnectTimeout=10)
REMOTE_ISAACLAB_DIR="$(larg_remote_path)"
LOCAL_ISAACLAB_DIR="$LARG_LOCAL_DIR"
SCRIPT_DIR="$HERE/../cluster/cluster_dev"  # trees.sh carries node_exec.sh and scripts/.env.* from beside it
TREE_RUN_HINT="scripts/larg/train.sh --tree <id> $host <task> <run_name> [run_group] [num_envs] [-- train.py args]"
ensure_master() { :; }  # plain ssh, no control master
log() { echo -e "[larg] $*"; }
err() { echo -e "\033[31m[larg] ERROR: $*\033[0m" >&2; }
on_login() { larg_ssh "$host" "bash -c $(printf %q "$1")"; }
check_space() {  # check_space DIR WHAT: the trees share the box's local disk with run logs and Kit caches
  local free min="${LARG_MIN_FREE_GB:-10}"
  free="$(on_login "mkdir -p '$1' && df -BG --output=avail '$1' | tail -1 | tr -dc 0-9")"
  if [ -n "$free" ] && [ "$free" -lt "$min" ]; then
    err "only ${free} GB free for $2 at $1 on ${host} (LARG_MIN_FREE_GB=${min}); free space or pass --no-space-check"
    exit 1
  fi
}
# trees.sh reads `set -u` defaults off cluster_dev.sh's environment
set +u
source "$SCRIPT_DIR/trees.sh"
set -u

case "$cmd" in
  stage)
    cmd_stage "$@"
    # a staged hcrl_isaaclab carries an empty .artifacts (node_exec binds the shared one over it on a cluster):
    # on a LARG box it links to the workspace's artifact root instead
    on_login "for t in \$(ls -1dt '${TREES_DIR}'/*/ | grep -v '\\.partial\\.'); do a=\"\$t/resources/hcrl_isaaclab/.artifacts\"; \
      [ -d \"\$a\" ] && [ ! -L \"\$a\" ] && rmdir \"\$a\" 2>/dev/null && ln -s '${REMOTE_ISAACLAB_DIR}/resources/hcrl_isaaclab/.artifacts' \"\$a\"; \
      done; true"
    ;;
  list) cmd_trees ;;
  rm) cmd_trees rm "$@" ;;
  *) err "unknown command ${cmd} (stage | list | rm)"; exit 1 ;;
esac
