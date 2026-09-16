#!/usr/bin/env bash
# LARG variant of watch_run.sh: the log AND the process are on the same box, so check both.
#   watch_larg.sh <label> <host> <log> <proc-grep> <target-iter> [stall_min] [report_every] [metric-grep]
set -u
LABEL=$1; HOST=$2; LOG=$3; PROC=$4; TARGET=$5; STALL_MIN=${6:-20}; EVERY=${7:-100}; MET=${8:-}
ERRPAT='Traceback|error running python|Error executing|Could not override|CUDA out of memory|No contact sensors|Segmentation fault|Killed|ValueError|RuntimeError'
SSHO="ssh -o ConnectTimeout=20 -o BatchMode=yes"
last_it=-1; last_change=$(date +%s); lastc=-1; booted=0
while true; do
  st=$(timeout 45 $SSHO $HOST "it=\$(grep -aoE 'Learning iteration [0-9]+/' $LOG 2>/dev/null | tail -1 | grep -oE '[0-9]+'); n=\$(ps -eo args | grep -c '$PROC'); err=\$(grep -acE '$ERRPAT' $LOG 2>/dev/null); echo \"it=\${it:-none} n=\$n err=\$err\"" 2>/dev/null | tail -1)
  [ -z "$st" ] && { sleep 120; continue; }
  it=$(echo "$st" | grep -oE 'it=[0-9]+' | cut -d= -f2); n=$(echo "$st" | grep -oE 'n=[0-9]+' | cut -d= -f2); err=$(echo "$st" | grep -oE 'err=[0-9]+' | cut -d= -f2); now=$(date +%s)
  if [ "${err:-0}" != "0" ]; then echo "[$LABEL] FAILED:"; timeout 45 $SSHO $HOST "grep -aE '$ERRPAT' $LOG | tail -3 | cut -c1-200" 2>/dev/null | grep -v ros/noetic; exit 1; fi
  if [ -n "$it" ] && [ "$it" != "$last_it" ]; then last_it=$it; last_change=$now; fi
  if [ $booted -eq 0 ] && [ -n "$it" ]; then booted=1; echo "[$LABEL] training (iter $it)"; fi
  if [ -n "$it" ] && [ "$it" -ge $((TARGET - 5)) ]; then echo "[$LABEL] finished at iter $it"; exit 0; fi
  if [ "${n:-1}" = "0" ]; then echo "[$LABEL] DEAD: no process, last iter $last_it"; timeout 45 $SSHO $HOST "tail -c 600 $LOG | tr '\r' '\n' | tail -3 | cut -c1-160" 2>/dev/null | grep -v ros/noetic; exit 1; fi
  if [ $booted -eq 1 ] && [ $(( (now - last_change) / 60 )) -ge "$STALL_MIN" ]; then echo "[$LABEL] STALLED at iter $last_it for ${STALL_MIN}+ min"; last_change=$now; fi
  if [ -n "$it" ]; then c=$((it / EVERY)); if [ "$c" != "$lastc" ]; then lastc=$c; echo "[$LABEL] iter $it $(timeout 45 $SSHO $HOST "grep -aE '$MET' $LOG | tail -3 | tr -s ' ' | paste -sd' ' | cut -c1-220" 2>/dev/null | grep -v ros/noetic)"; fi; fi
  sleep 120
done
