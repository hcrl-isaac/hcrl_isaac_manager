#!/usr/bin/env bash
# ssh, with each session over a shared ControlMaster holding one of a few local slots, so many watchers and commands
# sharing one master never open more sessions at once than sshd's MaxSessions; a session the server still refuses is
# retried with backoff. Arguments are ssh's own. Calls with no ControlPath, -O control commands and -t sessions pass
# straight through.
set -u
SLOTS=4           # under sshd's default MaxSessions (10) with room for the master's own and interactive sessions
SLOT_WAIT_S=60    # a session that waits longer runs unslotted: a long transfer must not block every other call
RETRIES=5
REFUSED='Session open refused by peer|mux_client_request_session: session request failed'

cp="" interactive=0 control=0 prev=""
for a in "$@"; do
    case "$a" in
        ControlPath=*) [ "$prev" = "-o" ] && cp=${a#ControlPath=} ;;
        -oControlPath=*) cp=${a#-oControlPath=} ;;
        -t | -tt) interactive=1 ;;
        -O) control=1 ;;
    esac
    prev=$a
done
if [ -z "$cp" ] || [ "$cp" = none ] || [ $interactive = 1 ] || [ $control = 1 ]; then
    exec ssh "$@"
fi

dir="$HOME/.ssh/cm/slots"
mkdir -p "$dir" 2>/dev/null
# one set of slots per master: the ControlPath expanded for this host (a %C template names every host's)
real=$(ssh -G "$@" 2>/dev/null | awk '$1 == "controlpath" { print $2; exit }')
key=$(printf '%s' "${real:-$cp}" | cksum | cut -d' ' -f1)
deadline=$(( $(date +%s) + SLOT_WAIT_S ))
got=""
while [ -z "$got" ] && [ "$(date +%s)" -lt "$deadline" ]; do
    for i in $(seq 0 $((SLOTS - 1))); do
        exec {fd}>>"$dir/$key.$i"
        if flock -n "$fd"; then got=$i; break; fi
        exec {fd}>&-
    done
    [ -z "$got" ] && sleep "0.$(( RANDOM % 9 + 1 ))"
done

err=$(mktemp "${TMPDIR:-/tmp}/ssh_slot.XXXXXX")
trap 'rm -f "$err"' EXIT
delay=1
for attempt in $(seq 1 $RETRIES); do
    ssh "$@" 2> >(tee "$err" >&2)
    rc=$?
    [ -n "${!:-}" ] && wait "$!" 2>/dev/null  # the tee, so $err is complete
    # a refused session never started, so its stdin was not read and the call can be made again
    if [ $rc -eq 255 ] && grep -qE "$REFUSED" "$err" && [ "$attempt" -lt $RETRIES ]; then
        sleep $(( delay + RANDOM % 2 )); delay=$(( delay * 2 ))
        continue
    fi
    exit $rc
done
