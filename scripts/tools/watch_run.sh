#!/usr/bin/env bash
# Watchdog for a training run. Every exit is FINISHED, STALLED, DEAD or FAILED.
#
#   watch_run.sh <label> <log> <target-iters> [stall_min] [report_every] [metric-grep]
#
# <log> is a local path, or user@host:/path when the log lands on a different host from the process.
set -u

LABEL=$1; LOG_SPEC=$2; TARGET=$3; STALL_MIN=${4:-20}; EVERY=${5:-100}; MET=${6:-}
case $LOG_SPEC in
  *:*) HOST=${LOG_SPEC%%:*}; LOG=${LOG_SPEC#*:} ;;
  *)   HOST=""; LOG=$LOG_SPEC ;;
esac
[ "${EVERY:-0}" -gt 0 ] 2>/dev/null || EVERY=0   # 0 disables the periodic metric line

ERRPAT='Traceback|error running python|Error executing|CUDA out of memory|Could not override|No contact sensors|Segmentation fault|Killed|srun: error|Disk quota exceeded|Worker exited -11'
# benign noise: ranks starting together, srun failing to load its unused http_parser plugin, and srun answering an
# internet scanner that probes its port (TACC amd-rtx) with version-mismatch errors while the step runs on
BENIGNPAT='omni/kit/pipapi|pip3-envs|no current CUDA context|ignore_import_check|_process_ext_pipapi_config|http_parser'
BENIGNPAT+='|Incompatible versions of client and server code|protocol_version [0-9]+ not supported|destroy_forward: no init'

# A Python traceback is judged whole (header, frames, exception, chained "During handling ..." tracebacks): noise when
# any of its lines is benign (Kit's pipapi pip-env race at boot, on every rank), else a failure. Lines outside a
# traceback count when they match ERRPAT and not BENIGNPAT. mode=count prints the count, mode=lines the offending lines.
# Rank prefixes such as "[default3]:" are ignored. (No single quotes: the program is passed in quotes.)
ERRAWK='
function strip(s) { sub(/^\[[^]]*\]:[ ]?/, "", s); return s }
function close_tb() { if (intb && !ok_tb) { bad++; if (mode == "lines") print tb } intb = 0; after = 0 }
{
    s = strip($0)
    if (s ~ /Traceback \(most recent call last\)/) {
        if (!intb) { intb = 1; ok_tb = 0; tb = $0 }
        after = 0; next
    }
    if (intb) {
        if (s ~ benign) ok_tb = 1
        if (s ~ /^[[:space:]]/ || s == "" || s ~ /During handling of the above exception|direct cause of the following exception/) {
            if (s ~ /^[[:space:]]/ && after) close_tb(); else next
        } else if (!after) { after = 1; tb = tb " / " $0; next }
        else close_tb()
    }
    if (!intb && s ~ err && s !~ benign) { bad++; if (mode == "lines") print $0 }
}
END { close_tb(); if (mode == "count") print bad + 0 }'

CM=${WATCH_RUN_CONTROL_PATH:-$HOME/.ssh/cm/%C}
mkdir -p "$(dirname "${CM/\%C/x}")" 2>/dev/null || true
run_remote() {  # stdin: the probe script
    if [ -n "$HOST" ]; then
        ssh -o ControlMaster=auto -o ControlPath="$CM" -o ConnectTimeout=25 "$HOST" bash -s 2>/dev/null
    else
        bash -s 2>/dev/null
    fi
}

probe() {
    run_remote <<EOF
f="$LOG"
if [ ! -e "\$f" ]; then echo "missing"; exit 0; fi
it=\$(grep -aoE 'Learning iteration [0-9]+/' "\$f" | tail -1 | grep -oE '[0-9]+')
age=\$(( \$(date +%s) - \$(stat -c %Y "\$f") ))
ln=\$(grep -anE 'Learning iteration [0-9]+/' "\$f" | tail -1 | cut -d: -f1)
# errors count only after the last progress line
err=\$(tail -n +\${ln:-1} "\$f" | awk -v mode=count -v err='$ERRPAT' -v benign='$BENIGNPAT' '$ERRAWK')
echo "it=\${it:-none} age=\$age err=\$err"
EOF
}

started=$(date +%s); last_it=-1; last_change=$started; lastc=-1; booted=0
while true; do
    st=$(probe)
    now=$(date +%s)
    if [ -z "$st" ]; then sleep 120; continue; fi          # a transient ssh failure is not evidence
    if [ "$st" = missing ]; then
        # a log that does not exist yet has not started
        if [ $(( (now - started) / 60 )) -ge "$STALL_MIN" ]; then
            echo "[$LABEL] DEAD: no log after $STALL_MIN min at $LOG_SPEC"; exit 1
        fi
        echo "[$LABEL] not started yet"; sleep 120; continue
    fi
    it=${st#*it=}; it=${it%% *}; age=${st#*age=}; age=${age%% *}; err=${st##*err=}
    [ "$it" = none ] && it=""

    if [ "${err:-0}" != "0" ]; then
        echo "[$LABEL] FAILED:"
        printf 'awk -v mode=lines -v err=%q -v benign=%q %q "%s" | tail -3 | cut -c1-200\n' \
            "$ERRPAT" "$BENIGNPAT" "$ERRAWK" "$LOG" | run_remote
        exit 1
    fi
    [ -n "$it" ] && [ "$it" != "$last_it" ] && { last_it=$it; last_change=$now; }
    [ $booted -eq 0 ] && [ -n "$it" ] && { booted=1; echo "[$LABEL] training (iter $it)"; }
    # a run may stop a few short of the nominal target
    if [ -n "$it" ] && [ "$it" -ge $((TARGET - 5)) ]; then
        echo "[$LABEL] finished at iter $it (target $TARGET)"; exit 0
    fi
    if [ "${age:-0}" -ge $((STALL_MIN * 60)) ]; then
        echo "[$LABEL] DEAD or hung: log untouched for $((age / 60)) min, last iter ${last_it}"
        printf 'tail -c 600 "%s" | tr "\\r" "\\n" | tail -3 | cut -c1-160\n' "$LOG" | run_remote
        exit 1
    fi
    if [ $booted -eq 1 ] && [ $(( (now - last_change) / 60 )) -ge "$STALL_MIN" ]; then
        echo "[$LABEL] STALLED: iter stuck at ${last_it} for ${STALL_MIN}+ min (log still growing)"
        last_change=$now
    fi
    if [ -n "$it" ] && [ "$EVERY" -gt 0 ] && [ -n "$MET" ]; then
        c=$((it / EVERY))
        if [ "$c" != "$lastc" ]; then
            lastc=$c
            echo "[$LABEL] iter $it $(printf 'grep -aE %q "%s" | tail -3 | tr -s " " | paste -sd" " | cut -c1-200\n' "$MET" "$LOG" | run_remote)"
        fi
    fi
    sleep 120
done
