#!/usr/bin/env bash
# Launch a detached cluster_dev step and record which step id it became. Shared sentinels give no
# step-to-owner mapping, so the id is the set difference of step ids before and after; two new ids = AMBIGUOUS.
#
# Usage: dev_exec_track.sh <cluster> <job id> <command...>
set -uo pipefail
CLUSTER=$1; JOB=$2; shift 2
LEDGER="${HOME}/.cluster_dev/${CLUSTER}/my_steps.tsv"
mkdir -p "$(dirname "$LEDGER")"
MGR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# query the login node over the existing control master; there is no `develop exec-login`, and a
# steps() that silently returns nothing makes every launch look AMBIGUOUS
LOGIN=$(grep -m1 '^CLUSTER_LOGIN=' "$MGR/scripts/cluster/config/$CLUSTER/.env.cluster" | cut -d= -f2-)
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm/%C" -o ControlPersist=48h \
          -o ConnectTimeout=60 -o BatchMode=yes)
steps() { timeout 60 ssh "${SSH_OPTS[@]}" "$LOGIN" "squeue -s -j $JOB -h -o %i" 2>/dev/null \
            | grep -oE "${JOB}\.[0-9]+" | sort -u; }

before=$(steps)
out=$(DEV_JOBID="$JOB" just -f "$MGR/justfile" cluster "$CLUSTER" develop exec --detach -- "$@" 2>&1)
log=$(printf '%s\n' "$out" | grep -oE '\$HOME/cluster_dev_run_[0-9-]+\.log' | head -1)
printf '%s\n' "$out"
# staging can sit at "Checking available allocation" for minutes before the step appears, so poll
new=""
for _ in $(seq 1 24); do
    sleep 10
    new=$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$(steps)"))
    [ -n "$(printf '%s' "$new" | tr -d '[:space:]')" ] && break
done
count=$(printf '%s\n' "$new" | grep -c . || true)
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [ "$count" = "1" ]; then
    printf '%s\t%s\t%s\tMINE\n' "$ts" "$new" "$log" >> "$LEDGER"
    echo "[track] recorded $new -> $LEDGER"
else
    printf '%s\t%s\t%s\tAMBIGUOUS(%s)\n' "$ts" "$(printf '%s' "$new" | tr '\n' ',')" "$log" "$count" >> "$LEDGER"
    echo "[track] AMBIGUOUS: $count new steps ($(printf '%s' "$new" | tr '\n' ' ')) -- do not cancel any of these on a guess"
fi
