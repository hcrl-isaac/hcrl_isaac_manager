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

# srun answering an internet scanner on its port while the step trains on (Stampede3 amd-rtx)
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

# Kit's pip-env race at boot prints a chained traceback on every rank of a healthy run (amd-rtx BFM-Zero retrain)
pipapi='[default3]:Traceback (most recent call last):
[default3]:  File "/isaac-sim/extscache/omni.kit.pipapi-0.0.0/omni/kit/pipapi/pipapi.py", line 412, in _ensure_env
[default3]:    os.makedirs(env_dir)
[default3]:FileNotFoundError: [Errno 2] No such file or directory: '"'"'/isaac-sim/kit/data/Kit/Isaac-Sim/5.1/pip3-envs/default'"'"'
[default3]:
[default3]:During handling of the above exception, another exception occurred:
[default3]:
[default3]:Traceback (most recent call last):
[default3]:  File "/isaac-sim/extscache/omni.kit.pipapi-0.0.0/omni/kit/pipapi/pipapi.py", line 418, in _ensure_env
[default3]:    os.mkdir(env_dir)
[default3]:FileExistsError: [Errno 17] File exists: '"'"'/isaac-sim/kit/data/Kit/Isaac-Sim/5.1/pip3-envs/default'"'"'
[default3]:Successfully loaded 862 motions'
printf '%s\n' "$pipapi" > "$T/pipapi_boot.log"
out="$(timeout 5 bash "$REPO/scripts/tools/watch_run.sh" probe "$T/pipapi_boot.log" 4000 20 0 2>&1)"; rc=$?
check "Kit's pipapi traceback at boot is not FAILED" '[ "$rc" = 124 ] && case $out in *FAILED*) false ;; *) true ;; esac'
printf '%s\nLearning iteration 12/4000\n' "$pipapi" > "$T/pipapi_training.log"
out="$(timeout 5 bash "$REPO/scripts/tools/watch_run.sh" probe "$T/pipapi_training.log" 4000 20 0 2>&1)"; rc=$?
check "and the run it precedes trains on" '[ "$rc" = 124 ] && case $out in *training*) true ;; *) false ;; esac'

real='[default1]:Traceback (most recent call last):
[default1]:  File "/workspace/ext/hcrl_isaaclab/scripts/train.py", line 210, in main
[default1]:    runner.learn()
[default1]:RuntimeError: CUDA error: an illegal memory access was encountered'
printf '%s\n%s\n' "$pipapi" "$real" > "$T/real_tb.log"
out="$(watch "$T/real_tb.log")"; rc=$?
check "a real traceback beside the pipapi one is FAILED" '[ "$rc" = 1 ] && case $out in *FAILED*) true ;; *) false ;; esac'
check "and the report shows the real one, not the boot noise" 'case $out in *illegal\ memory*) true ;; *) false ;; esac && case $out in *pip3-envs*) false ;; *) true ;; esac'
# a benign segment chained into a real failure is still a failure: each segment is judged by its exception line
chained='Traceback (most recent call last):
  File "/isaac-sim/extscache/omni.kit.pipapi-0.0.0/omni/kit/pipapi/pipapi.py", line 412, in _ensure_env
FileExistsError: [Errno 17] File exists: '"'"'/isaac-sim/kit/data/Kit/Isaac-Sim/5.1/pip3-envs/default'"'"'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/workspace/ext/hcrl_isaaclab/scripts/train.py", line 88, in main
RuntimeError: CUDA error: device-side assert triggered'
printf 'Learning iteration 40/4000\n%s\n' "$chained" > "$T/chained.log"
out="$(watch "$T/chained.log")"; rc=$?
check "a pipapi segment chained into a real error is FAILED" '[ "$rc" = 1 ] && case $out in *device-side*) true ;; *) false ;; esac'
printf 'Learning iteration 40/4000\nTraceback (most recent call last):\n  File "x.py", line 3, in f  # no current CUDA context\nValueError: bad shape\n' > "$T/benign_frame.log"
out="$(watch "$T/benign_frame.log")"; rc=$?
check "a benign phrase in a frame does not hide a real exception" '[ "$rc" = 1 ] && case $out in *ValueError*) true ;; *) false ;; esac'
printf 'Learning iteration 40/4000\nTraceback (most recent call last):\n  File "train.py", line 3, in <module>\nKeyError: '"'"'x'"'"'\n' > "$T/plain_tb.log"
out="$(watch "$T/plain_tb.log")"; rc=$?
check "an unprefixed traceback after progress is FAILED" '[ "$rc" = 1 ] && case $out in *KeyError*) true ;; *) false ;; esac'

# the added patterns must not swallow the normal completion path
printf 'Learning iteration 4000/4000\n' > "$T/done.log"
out="$(watch "$T/done.log")"; rc=$?
check "reaching the target is FINISHED" '[ "$rc" = 0 ] && case $out in *finished*) true ;; *) false ;; esac'

[ "$fails" -eq 0 ] || exit 1
echo "all watch_run exit tests passed"
