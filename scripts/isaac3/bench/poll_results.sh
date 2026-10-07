#!/bin/bash
# usage: poll_results.sh USER@HOST:ROOT ... -- print each new bench result line from every box; exit once every box reports done or FAILED.
declare -A SEEN
CM="-o BatchMode=yes -o ControlMaster=no -o ControlPath=/home/ecs3239/.ssh/cm/%C"
hosts=("$@")  # user@host:/root ...
while true; do
  finished=0
  for h in "${hosts[@]}"; do
    host=${h%%:*}; root=${h#*:}
    out=$(timeout 30 ssh $CM $host "cat $root/bench/results.txt 2>/dev/null" 2>/dev/null)
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      key="$host|$line"
      [ -n "${SEEN[$key]}" ] && continue
      SEEN[$key]=1
      echo "$line"
    done <<< "$out"
    echo "$out" | grep -qE " done$|FAILED" && finished=$((finished + 1))
  done
  [ $finished -ge ${#hosts[@]} ] && { echo "ALL BENCHES FINISHED"; exit 0; }
  sleep 60
done
