#!/usr/bin/env bash
# The sentinel starts a spooled step itself (inside the job): a request dropped in its spool runs once, writing to the
# log path beside it, and the spool goes when the sentinel is cancelled.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
T="$(mktemp -d)"
trap 'kill "$pid" 2>/dev/null || true; rm -rf "$T"' EXIT
export HOME="$T/home" SLURM_JOB_ID=42
mkdir -p "$HOME"
spool="$HOME/.cluster_dev/spool/42"
bash "$REPO/scripts/cluster/cluster_dev/sentinel.sbatch" > "$T/sentinel.out" 2>&1 &
pid=$!
for _ in $(seq 50); do [ -e "$spool/.ready" ] && break; sleep 0.1; done
[ -e "$spool/.ready" ] || { echo "FAIL: the sentinel made no spool"; exit 1; }

# a request is written whole, then renamed into place, as `exec --detach` does
echo 'echo "step ran in job $SLURM_JOB_ID"' > "$spool/a.tmp"
printf %s "$T/a.log" > "$spool/a.log"
mv "$spool/a.tmp" "$spool/a.req"
for _ in $(seq 100); do grep -q "step ran" "$T/a.log" 2>/dev/null && break; sleep 0.1; done
grep -q "step ran in job 42" "$T/a.log" || { echo "FAIL: the spooled step did not run"; cat "$T/sentinel.out"; exit 1; }
[ ! -e "$spool/a.req" ] && [ -e "$spool/a.started" ] || { echo "FAIL: the request was not taken once"; exit 1; }
echo "PASS a spooled step runs once, logging to its own path"

kill -TERM "$pid"
for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
kill -0 "$pid" 2>/dev/null && { echo "FAIL: the sentinel ignored TERM"; exit 1; }
[ ! -e "$spool" ] || { echo "FAIL: a cancelled sentinel left its spool"; exit 1; }
echo "PASS a cancelled sentinel removes its spool"
