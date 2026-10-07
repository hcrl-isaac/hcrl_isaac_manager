#!/usr/bin/env bash
# watch_run.sh reports a run that died on a render-worker segfault or a full filesystem as FAILED, rather
# than waiting out the stall timer. Local logs only, no ssh.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

watch() {  # watch <log-file> ; prints the verdict, exits with watch_run's code
    bash "$REPO/scripts/tools/watch_run.sh" probe "$1" 4000 20 0
}

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

# the renderer dies on driver 595 and the trainer leaves no traceback of its own, so this line is the only tell
printf 'Learning iteration 5/4000\n[recorder] Worker exited -11; retrying.\n' > "$T/segv.log"
out="$(watch "$T/segv.log")"; rc=$?
check "render-worker segfault is FAILED" '[ "$rc" = 1 ] && case $out in *FAILED*) true ;; *) false ;; esac'

printf 'Learning iteration 5/4000\nOSError: [Errno 122] Disk quota exceeded\n' > "$T/quota.log"
out="$(watch "$T/quota.log")"; rc=$?
check "disk quota is FAILED" '[ "$rc" = 1 ] && case $out in *FAILED*) true ;; *) false ;; esac'

# srun answering an internet scanner on its port while the step trains on (Stampede3 amd-rtx, 2026-10-06)
scanner='srun: error: unpack_header: protocol_version 65363 not supported
srun: error: destroy_forward: no init
srun: error: slurm_unpack_received_msg: [63.146.94.167.censys-scanner.com:13610] Incompatible versions of client and server code
srun: error: wrap_on_data: [63.146.94.167.censys-scanner.com:13610] on_data returned rc: Incompatible versions of client and server code'
printf 'Learning iteration 927/4000\n%s\n' "$scanner" > "$T/scanner.log"
out="$(timeout 5 bash "$REPO/scripts/tools/watch_run.sh" probe "$T/scanner.log" 4000 20 0 2>&1)"; rc=$?
check "scanner noise on srun's port is not FAILED" '[ "$rc" = 124 ] && case $out in *FAILED*) false ;; *) true ;; esac'

printf 'Learning iteration 927/4000\n%s\nsrun: error: Node failure on c571-003\n' "$scanner" > "$T/scanner_real.log"
out="$(watch "$T/scanner_real.log")"; rc=$?
check "a real srun error next to scanner noise is FAILED" '[ "$rc" = 1 ] && case $out in *FAILED*) true ;; *) false ;; esac'

# the added patterns must not swallow the normal completion path
printf 'Learning iteration 4000/4000\n' > "$T/done.log"
out="$(watch "$T/done.log")"; rc=$?
check "reaching the target is FINISHED" '[ "$rc" = 0 ] && case $out in *finished*) true ;; *) false ;; esac'

[ "$fails" -eq 0 ] || exit 1
echo "all watch_run exit tests passed"
