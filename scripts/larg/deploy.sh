#!/usr/bin/env bash
# One-shot deploy to a LARG box: rsync code, then run remote_setup.sh in the
# background (nohup) so the long isaacsim install survives the SSH session.
#
# Usage: scripts/larg/deploy.sh <host> [<host> ...]
# Then poll the printed log with: scripts/larg/deploy.sh --log <host>

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

REMOTE_LOG="\$HOME/larg_setup.log"

if [ "${1:-}" = "--log" ]; then
  shift
  for host in "$@"; do
    echo "=== $host: tail $REMOTE_LOG ==="
    larg_ssh "$host" "tail -n 40 $REMOTE_LOG 2>/dev/null; echo; echo '--- setup proc ---'; pgrep -af remote_setup.sh || echo '(no remote_setup.sh running)'"
  done
  exit 0
fi

[ $# -ge 1 ] || { echo "usage: $0 <host> [<host> ...]  |  $0 --log <host>"; exit 1; }

for host in "$@"; do
  echo "########## DEPLOY $host ##########"
  "$HERE/sync.sh" "$host"
  echo "=== launching remote_setup.sh (nohup) on $host ==="
  larg_ssh "$host" "cd \$HOME/$LARG_REMOTE_DIR && nohup bash scripts/larg/remote_setup.sh > $REMOTE_LOG 2>&1 & echo started pid \$! ; echo log: $REMOTE_LOG"
done
echo
echo "Poll with: $0 --log $*"
