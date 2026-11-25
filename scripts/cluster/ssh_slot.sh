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

cp="" dest="" interactive=0 control=0 opt=""
for a in "$@"; do
    if [ -n "$opt" ]; then  # the value of the option before it, never the destination
        [ "$opt" = -o ] && case "$a" in ControlPath=*) cp=${a#ControlPath=} ;; esac
        [ "$opt" = -O ] && control=1
        opt=""
        continue
    fi
    case "$a" in
        -oControlPath=*) cp=${a#-oControlPath=} ;;
        -[BbcDEeFIiJLlmOoPpQRSWw]) opt=$a ;;
        -O?*) control=1 ;;
        -t | -tt) interactive=1 ;;
        -*) ;;
        *) dest=$a; break ;;  # the rest is the remote command
    esac
done
if [ -z "$cp" ] || [ "$cp" = none ] || [ $interactive = 1 ] || [ $control = 1 ]; then
    exec ssh "$@"
fi

dir="$HOME/.ssh/cm/slots"
mkdir -p "$dir" 2>/dev/null
# one set of slots per master: its ControlPath template with the destination (a %C template names every host's)
key=$(printf '%s %s' "$cp" "$dest" | cksum | cut -d' ' -f1)
deadline=$(( $(date +%s) + SLOT_WAIT_S ))
got=""
while [ -z "$got" ] && [ "$(date +%s)" -lt "$deadline" ]; do
    for i in $(seq 0 $((SLOTS - 1))); do
        exec 9>>"$dir/$key.$i"  # the lock is this shell's fd 9; nothing it starts keeps it
        if flock -n 9; then got=$i; break; fi
        exec 9>&-
    done
    [ -z "$got" ] && sleep "0.$(( RANDOM % 9 + 1 ))"
done

err=$(mktemp "${TMPDIR:-/tmp}/ssh_slot.XXXXXX")
trap 'rm -f "$err"' EXIT
delay=1
for attempt in $(seq 1 $RETRIES); do
    # a ControlPersist master ssh daemonizes would otherwise hold the slot for its whole life
    ssh "$@" 9>&- 2> >(exec tee "$err" >&2 9>&-)
    rc=$?
    [ -n "${!:-}" ] && wait "$!" 2>/dev/null  # the tee, so $err is complete
    # a refused session never started, so its stdin was not read and the call can be made again
    if [ $rc -eq 255 ] && grep -qE "$REFUSED" "$err" && [ "$attempt" -lt $RETRIES ]; then
        sleep $(( delay + RANDOM % 2 )); delay=$(( delay * 2 ))
        continue
    fi
    exit $rc
done
