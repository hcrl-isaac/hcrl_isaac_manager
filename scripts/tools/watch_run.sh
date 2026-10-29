#!/usr/bin/env bash
# Watchdog for a training run. Every exit is FINISHED, STALLED, DEAD, FAILED or BLIND (no probe answered for stall_min).
#
#   watch_run.sh [--wandb <entity/project/run_id>] <label> <log> <target-iters> [stall_min] [report_every] [metric-grep]
#
# <log> is a local path, user@host:/path when the log lands on a different host from the process, or ray:<job_id>
# for a Ray job (read through `pls jobs --on ray logs`; a job Ray reports FAILED or STOPPED is FAILED here). A run that
# logs every N iterations finishes when it closes (W&B's sync lines after its last iteration) within N of the target.
# With --wandb, a log that cannot be read (its login node down) is replaced by the run's W&B state, iteration and last
# log time, so a login outage is not BLIND while the run still logs.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WANDB_RUN=""
if [ "${1:-}" = --wandb ]; then WANDB_RUN=$2; shift 2; fi
LABEL=$1; LOG_SPEC=$2; TARGET=$3; STALL_MIN=${4:-20}; EVERY=${5:-100}; MET=${6:-}
RAY_JOB=""
case $LOG_SPEC in
  ray:*) RAY_JOB=${LOG_SPEC#ray:}; HOST=""; LOG="${TMPDIR:-/tmp}/watch_run_ray_${RAY_JOB}_$$.log"
         trap 'rm -f "$LOG" "$LOG.new" "$LOG.status"' EXIT ;;
  *:*) HOST=${LOG_SPEC%%:*}; LOG=${LOG_SPEC#*:} ;;
  *)   HOST=""; LOG=$LOG_SPEC ;;
esac
PLS=${WATCH_RUN_PLS:-pls}  # the pls a Ray watch asks; tests stub it
[ "${EVERY:-0}" -gt 0 ] 2>/dev/null || EVERY=0   # 0 disables the periodic metric line

ERRPAT='Traceback|error running python|Error executing|CUDA out of memory|Could not override|No contact sensors|Segmentation fault|Killed|srun: error|Disk quota exceeded|Worker exited -11'
# benign noise: ranks starting together, srun failing to load its unused http_parser plugin, and srun answering an
# internet scanner that probes its port (TACC amd-rtx) with version-mismatch errors while the step runs on
BENIGNPAT='omni/kit/pipapi|pip3-envs|no current CUDA context|ignore_import_check|_process_ext_pipapi_config|http_parser'
BENIGNPAT+='|Incompatible versions of client and server code|protocol_version [0-9]+ not supported|destroy_forward: no init'

# A Python traceback is judged per segment (each header with its frames and exception line; chained "During handling
# ..." tracebacks are further segments): noise only when every segment's exception line is benign (Kit's pipapi
# pip-env race at boot, on every rank), so a benign segment chained into a real error still fails. A segment that
# ends without an exception line counts as a failure. Lines outside a traceback count when they match ERRPAT and not
# BENIGNPAT. mode=count prints the count, mode=lines the offending lines. Rank prefixes such as "[default3]:" are
# ignored. (No single quotes: the program is passed in quotes.)
ERRAWK='
function strip(s) { sub(/^\[[^]]*\]:[ ]?/, "", s); return s }
function close_tb() {
    if (intb && !after) all_ok = 0
    if (intb && !all_ok) { bad++; if (mode == "lines") print tb }
    intb = 0; after = 0
}
{
    s = strip($0)
    if (s ~ /Traceback \(most recent call last\)/) {
        if (!intb) { intb = 1; all_ok = 1; tb = $0 }
        else if (!after) all_ok = 0
        after = 0; next
    }
    if (intb) {
        if (s ~ /^[[:space:]]/ || s == "" || s ~ /During handling of the above exception|direct cause of the following exception/) {
            if (s ~ /^[[:space:]]/ && after) close_tb(); else next
        } else if (!after) {
            after = 1; tb = tb " / " $0
            if (s !~ benign) all_ok = 0
            next
        } else close_tb()
    }
    if (!intb && s ~ err && s !~ benign) { bad++; if (mode == "lines") print $0 }
}
END { close_tb(); if (mode == "count") print bad + 0 }'

CM=${WATCH_RUN_CONTROL_PATH:-$HOME/.ssh/cm/%C}
mkdir -p "$(dirname "${CM/\%C/x}")" 2>/dev/null || true
PROBE_S=90  # wall-clock limit per probe: a lost ControlMaster must fail, never wait on a 2FA prompt
run_remote() {  # stdin: the probe script
    if [ -n "$HOST" ]; then
        # only what the script prints counts: a login notice (e.g. TACC's filesystem banner) comes before the marker,
        # and a read that never reaches it is a failed probe, retried until stall_min
        { echo 'echo __WATCH_RUN_BEGIN__'; cat; } |
            timeout "$PROBE_S" ssh -o BatchMode=yes -o ControlMaster=auto -o ControlPath="$CM" -o ConnectTimeout=25 \
                -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$HOST" bash -s 2>/dev/null |
            awk 'seen; $0 == "__WATCH_RUN_BEGIN__" { seen = 1 }'
    else
        timeout "$PROBE_S" bash -s 2>/dev/null
    fi
}

RAY_STATUS=""
ray_refresh() {  # a Ray job's log into $LOG, replaced only when it changed, so its mtime is the last new output
    timeout "$PROBE_S" "$PLS" jobs --on ray logs "$RAY_JOB" > "$LOG.new" 2>/dev/null || { rm -f "$LOG.new"; return 1; }
    if cmp -s "$LOG.new" "$LOG"; then rm -f "$LOG.new"; else mv "$LOG.new" "$LOG"; fi
    # probe runs in a subshell, so the state goes through a file the loop reads
    timeout "$PROBE_S" "$PLS" jobs --on ray list --all_statuses --status_of "$RAY_JOB" 2>/dev/null | tail -1 > "$LOG.status"
}

wandb_probe() {  # `state=.. it=.. age=..` for $WANDB_RUN, or nothing (WATCH_RUN_WANDB_PROBE stands in for tests)
    if [ -n "${WATCH_RUN_WANDB_PROBE:-}" ]; then timeout "$PROBE_S" "$WATCH_RUN_WANDB_PROBE" "$WANDB_RUN" 2>/dev/null; return; fi
    local py common
    common=$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
    py="${common%/.git}/ilab/bin/python"; [ -x "$py" ] || py=python3
    timeout "$PROBE_S" "$py" "$HERE/wandb_probe.py" "$WANDB_RUN" 2>/dev/null
}

probe() {
    if [ -n "$RAY_JOB" ]; then ray_refresh || return 0; fi  # no output: a failed probe
    run_remote <<EOF
f="$LOG"
if [ ! -e "\$f" ]; then echo "missing"; exit 0; fi
it=\$(grep -aoE 'Learning iteration [0-9]+/' "\$f" | tail -1 | grep -oE '[0-9]+')
age=\$(( \$(date +%s) - \$(stat -c %Y "\$f") ))
prev=\$(grep -aoE 'Learning iteration [0-9]+/' "\$f" | tail -2 | head -1 | grep -oE '[0-9]+')
ln=\$(grep -anE 'Learning iteration [0-9]+/' "\$f" | tail -1 | cut -d: -f1)
# errors and the run's closing lines count only after the last progress line (W&B prints "View run" at start too)
err=\$(tail -n +\${ln:-1} "\$f" | awk -v mode=count -v err='$ERRPAT' -v benign='$BENIGNPAT' '$ERRAWK')
closed=\$(tail -n +\${ln:-1} "\$f" | grep -acE 'Synced [0-9]+ W&B file|View run .* at:')
echo "it=\${it:-none} prev=\${prev:-none} age=\$age err=\$err closed=\$closed"
EOF
}

POLL_S=120  # seconds between probes
started=$(date +%s); last_it=-1; last_change=$started; lastc=-1; booted=0; last_ok=$started; wb_note=0
while true; do
    st=$(probe)
    now=$(date +%s)
    [ -n "$RAY_JOB" ] && RAY_STATUS=$(cat "$LOG.status" 2>/dev/null)
    wb=""
    [ -z "$st" ] && [ -n "$WANDB_RUN" ] && wb=$(wandb_probe)
    if [ -n "$wb" ]; then  # the log is out of reach, W&B answers: its state, iteration and last log time stand in
        last_ok=$now
        [ $wb_note -eq 0 ] && { echo "[$LABEL] $LOG_SPEC unreachable: following W&B run $WANDB_RUN"; wb_note=1; }
        wfield() { local v=${wb#* $1=}; [ "$v" = "$wb" ] && v=${wb#$1=}; echo "${v%% *}"; }
        wstate=$(wfield state); w_iter=$(wfield it); wage=$(wfield age)
        [ "$w_iter" = none ] && w_iter=""
        case $wstate in
            crashed|failed|killed)
                echo "[$LABEL] FAILED: W&B run $WANDB_RUN is $wstate at iter ${w_iter:-none} (target $TARGET)"; exit 1 ;;
            finished)
                if [ -n "$w_iter" ] && [ "$w_iter" -ge $((TARGET - 5)) ]; then
                    echo "[$LABEL] finished: W&B run $WANDB_RUN finished at iter $w_iter (target $TARGET)"; exit 0
                fi
                echo "[$LABEL] FAILED: W&B run $WANDB_RUN finished at iter ${w_iter:-none}, short of target $TARGET"; exit 1 ;;
        esac
        if [ "${wage:--1}" -ge $((STALL_MIN * 60)) ]; then
            echo "[$LABEL] DEAD or hung: W&B run $WANDB_RUN logged nothing for $((wage / 60)) min, last iter ${w_iter:-none}"
            exit 1
        fi
        if [ -n "$w_iter" ]; then
            [ "$w_iter" != "$last_it" ] && { last_it=$w_iter; last_change=$now; }
            [ $booted -eq 0 ] && { booted=1; echo "[$LABEL] training (iter $w_iter, from W&B)"; }
            if [ "$w_iter" -ge $((TARGET - 5)) ]; then echo "[$LABEL] finished at iter $w_iter (target $TARGET, from W&B)"; exit 0; fi
            if [ "$EVERY" -gt 0 ] && [ $((w_iter / EVERY)) != "$lastc" ]; then
                lastc=$((w_iter / EVERY)); echo "[$LABEL] iter $w_iter (W&B)"
            fi
        fi
        if [ $booted -eq 1 ] && [ $(( (now - last_change) / 60 )) -ge "$STALL_MIN" ]; then
            echo "[$LABEL] STALLED: iter stuck at ${last_it} for ${STALL_MIN}+ min (W&B still logging)"
            last_change=$now
        fi
        sleep "${POLL_S:-120}"; continue
    fi
    if [ -z "$st" ]; then  # one failed probe is not evidence, stall_min of them is
        if [ $(( (now - last_ok) / 60 )) -ge "$STALL_MIN" ]; then
            echo "[$LABEL] BLIND: no probe of $LOG_SPEC answered for $(( (now - last_ok) / 60 )) min, last iter ${last_it}"
            exit 1
        fi
        sleep "${POLL_S:-120}"; continue
    fi
    last_ok=$now
    if [ "$st" = missing ]; then
        # a log that does not exist yet has not started
        if [ $(( (now - started) / 60 )) -ge "$STALL_MIN" ]; then
            echo "[$LABEL] DEAD: no log after $STALL_MIN min at $LOG_SPEC"; exit 1
        fi
        echo "[$LABEL] not started yet"; sleep 120; continue
    fi
    field() { local v=${st#* $1=}; [ "$v" = "$st" ] && v=${st#$1=}; echo "${v%% *}"; }
    it=$(field it); prev=$(field prev); age=$(field age); err=$(field err); closed=$(field closed)
    [ "$it" = none ] && it=""
    [ "$prev" = none ] && prev=$it

    if [ "${err:-0}" != "0" ]; then
        echo "[$LABEL] FAILED:"
        printf 'awk -v mode=lines -v err=%q -v benign=%q %q "%s" | tail -3 | cut -c1-200\n' \
            "$ERRPAT" "$BENIGNPAT" "$ERRAWK" "$LOG" | run_remote
        exit 1
    fi
    [ -n "$it" ] && [ "$it" != "$last_it" ] && { last_it=$it; last_change=$now; }
    if [ $booted -eq 0 ] && [ -n "$it" ]; then
        booted=1; echo "[$LABEL] training (iter $it)"
        [ "$EVERY" -gt 0 ] && lastc=$((it / EVERY))
    fi
    # a run that logs every N iterations closes up to N past its last progress line
    if [ -n "$it" ] && [ "${closed:-0}" != "0" ]; then
        gap=$((it - prev)); [ "$gap" -gt 0 ] || gap=0
        if [ $((it + gap)) -ge $((TARGET - 5)) ]; then
            echo "[$LABEL] finished: the run closed after iter $it (logs every ${gap:-1}, target $TARGET)"; exit 0
        fi
        echo "[$LABEL] FAILED: the run closed at iter $it, short of target $TARGET"; exit 1
    fi
    # a run may stop a few short of the nominal target
    if [ -n "$it" ] && [ "$it" -ge $((TARGET - 5)) ]; then
        echo "[$LABEL] finished at iter $it (target $TARGET)"; exit 0
    fi
    case $RAY_STATUS in  # Ray knows the job ended before its log can say so
        FAILED|STOPPED)
            echo "[$LABEL] FAILED: Ray job $RAY_JOB is $RAY_STATUS at iter ${it:-none} (target $TARGET)"
            tail -c 600 "$LOG" | tr "\r" "\n" | tail -3 | cut -c1-160
            exit 1 ;;
        SUCCEEDED)
            echo "[$LABEL] FAILED: Ray job $RAY_JOB SUCCEEDED at iter ${it:-none}, short of target $TARGET"; exit 1 ;;
    esac
    if [ "${age:-0}" -ge $((STALL_MIN * 60)) ]; then
        echo "[$LABEL] DEAD or hung: log untouched for $((age / 60)) min, last iter ${last_it}"
        printf 'tail -c 600 "%s" | tr "\\r" "\\n" | tail -3 | cut -c1-160\n' "$LOG" | run_remote
        exit 1
    fi
    if [ $booted -eq 1 ] && [ $(( (now - last_change) / 60 )) -ge "$STALL_MIN" ]; then
        echo "[$LABEL] STALLED: iter stuck at ${last_it} for ${STALL_MIN}+ min (log still growing)"
        last_change=$now
    fi
    if [ -n "$it" ] && [ "$EVERY" -gt 0 ]; then  # a progress line every report_every iterations
        c=$((it / EVERY))
        if [ "$c" != "$lastc" ]; then
            lastc=$c
            line="[$LABEL] iter $it"
            [ -n "$MET" ] && line="$line $(printf 'grep -aE %q "%s" | tail -3 | tr -s " " | paste -sd" " | cut -c1-200\n' "$MET" "$LOG" | run_remote)"
            echo "$line"
        fi
    fi
    sleep "${POLL_S:-120}"
done
