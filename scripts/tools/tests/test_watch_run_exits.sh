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

# FB-CPR logs every 50 iterations: the last line is 28400/28444, then it saves, syncs W&B and exits
printf 'Learning iteration 28350/28444\nLearning iteration 28400/28444\nsaved model_28443.pt\nwandb: Synced 6 W&B file(s), 0 media file(s)\n' > "$T/closed.log"
out="$(bash "$REPO/scripts/tools/watch_run.sh" probe "$T/closed.log" 28443 20 0)"; rc=$?
check "a run that closes within one log interval of the target is FINISHED" '[ "$rc" = 0 ] && case $out in *finished*) true ;; *) false ;; esac'
printf 'Learning iteration 100/4000\nLearning iteration 150/4000\nwandb: \xf0\x9f\x9a\x80 View run r at: https://wandb.ai/x\n' > "$T/short.log"
out="$(watch "$T/short.log")"; rc=$?
check "a run that closes far short of the target is FAILED" '[ "$rc" = 1 ] && case $out in *short\ of\ target*) true ;; *) false ;; esac'
printf 'wandb: \xf0\x9f\x9a\x80 View run r at: https://wandb.ai/x\nLearning iteration 10/4000\n' > "$T/starting.log"
out="$(timeout 5 bash "$REPO/scripts/tools/watch_run.sh" probe "$T/starting.log" 4000 20 0 2>&1)"; rc=$?
check "W&B's start line before training is not a close" '[ "$rc" = 124 ] && case $out in *training*) true ;; *) false ;; esac'

# report_every prints a progress line per that many iterations, metric grep or not (a copy polls every second)
sed 's/^POLL_S=120 /POLL_S=1 /' "$REPO/scripts/tools/watch_run.sh" > "$T/watch_fast.sh"
printf 'Learning iteration 50/4000\n' > "$T/progress.log"
( for i in 100 150 200 250; do sleep 0.6; printf 'Learning iteration %s/4000\n' "$i" >> "$T/progress.log"; done ) &
out="$(timeout 5 bash "$T/watch_fast.sh" probe "$T/progress.log" 4000 20 100 2>&1)"
wait
check "report_every prints progress without a metric grep" 'case $out in *"iter 1"[05]0*) true ;; *) false ;; esac && case $out in *"iter 2"[05]0*) true ;; *) false ;; esac'

# a probe that hangs (ssh waiting on a prompt) is cut off, and stall_min without an answer is BLIND, never silence
mkdir -p "$T/bin" && printf '#!/usr/bin/env bash\nsleep 600\n' > "$T/bin/ssh" && chmod +x "$T/bin/ssh"
sed 's/^POLL_S=120 /POLL_S=1 /; s/^PROBE_S=90 /PROBE_S=1 /' "$REPO/scripts/tools/watch_run.sh" > "$T/watch_blind.sh"
out="$(PATH="$T/bin:$PATH" timeout 10 bash "$T/watch_blind.sh" probe "x@host:/log" 4000 0 0 2>&1)"; rc=$?
check "a hanging probe ends as BLIND" '[ "$rc" = 1 ] && case $out in *BLIND*) true ;; *) false ;; esac'
out="$(PATH="$T/bin:$PATH" timeout 4 bash "$T/watch_blind.sh" probe "x@host:/log" 4000 20 0 2>&1)"; rc=$?
check "a few failed probes inside stall_min are retried" '[ "$rc" = 124 ] && [ -z "$out" ]'

# a login notice the remote shell prints on stdout (TACC's filesystem banner) is not the run's log
mkdir -p "$T/bannerbin"
printf '#!/usr/bin/env bash\necho "/work2 is mounted not \\"FULL\\" nor \\"IDLE\\"!"\nexec bash -s\n' > "$T/bannerbin/ssh"
chmod +x "$T/bannerbin/ssh"
printf 'Learning iteration 3486/4000\n' > "$T/banner_ok.log"
out="$(PATH="$T/bannerbin:$PATH" timeout 4 bash "$T/watch_fast.sh" probe "x@host:$T/banner_ok.log" 4000 20 0 2>&1)"; rc=$?
check "a login banner on a healthy run is not FAILED" '[ "$rc" = 124 ] && case $out in *FAILED*) false ;; *"training (iter 3486)"*) true ;; *) false ;; esac'
printf 'Learning iteration 10/4000\nTraceback (most recent call last):\nRuntimeError: boom\n' > "$T/banner_bad.log"
out="$(PATH="$T/bannerbin:$PATH" timeout 4 bash "$T/watch_fast.sh" probe "x@host:$T/banner_bad.log" 4000 20 0 2>&1)"; rc=$?
check "a real failure behind a banner is FAILED, reported from the log" \
    '[ "$rc" = 1 ] && case $out in *boom*) true ;; *) false ;; esac && case $out in *mounted*) false ;; *) true ;; esac'

# ray:<job>: the log comes through `pls jobs --on ray logs`, the job's state through `pls jobs --on ray list --status_of`
cat > "$T/pls" <<EOF
#!/usr/bin/env bash
case "\$4" in logs) cat "$T/ray.log" ;; list) cat "$T/ray.status" ;; esac
EOF
chmod +x "$T/pls"
sed 's/^POLL_S=120 /POLL_S=1 /' "$REPO/scripts/tools/watch_run.sh" > "$T/watch_ray.sh"
ray_watch() { WATCH_RUN_PLS="$T/pls" timeout 5 bash "$T/watch_ray.sh" probe ray:raysubmit_1 4000 20 0 2>&1; }

printf 'Learning iteration 120/4000\n' > "$T/ray.log"; echo RUNNING > "$T/ray.status"
out="$(ray_watch)"; rc=$?
check "a running Ray job is watched" '[ "$rc" = 124 ] && case $out in *"training (iter 120)"*) true ;; *) false ;; esac'

echo FAILED > "$T/ray.status"
out="$(ray_watch)"; rc=$?
check "a Ray job Ray reports FAILED is FAILED" '[ "$rc" = 1 ] && case $out in *"raysubmit_1 is FAILED"*) true ;; *) false ;; esac'

printf 'Learning iteration 3999/4000\n' > "$T/ray.log"; echo SUCCEEDED > "$T/ray.status"
out="$(ray_watch)"; rc=$?
check "a Ray job that reached its target finished" '[ "$rc" = 0 ] && case $out in *finished*) true ;; *) false ;; esac'

# --wandb: a log host that does not answer (login node down) hands over to the run's W&B state, never BLIND while it logs
printf '#!/usr/bin/env bash\ncat "%s/wb.state"\n' "$T" > "$T/wb_probe" && chmod +x "$T/wb_probe"
wb_watch() {
    WATCH_RUN_WANDB_PROBE="$T/wb_probe" PATH="$T/bin:$PATH" timeout "$1" bash "$T/watch_blind.sh" \
        --wandb e/p/r probe "x@host:/log" 4000 "$2" 0 2>&1
}
echo "state=running it=1200 age=30" > "$T/wb.state"
out="$(wb_watch 6 1)"; rc=$?
check "an unreachable log with a logging W&B run is followed, not BLIND" \
    '[ "$rc" = 124 ] && case $out in *"following W&B"*"iter 1200"*) true ;; *) false ;; esac && case $out in *BLIND*) false ;; *) true ;; esac'
echo "state=crashed it=1200 age=30" > "$T/wb.state"
out="$(wb_watch 6 1)"; rc=$?
check "a crashed W&B run is FAILED" '[ "$rc" = 1 ] && case $out in *"is crashed"*) true ;; *) false ;; esac'
echo "state=finished it=3999 age=30" > "$T/wb.state"
out="$(wb_watch 6 1)"; rc=$?
check "a W&B run finished at its target is finished" '[ "$rc" = 0 ] && case $out in *finished*) true ;; *) false ;; esac'
echo "state=finished it=900 age=30" > "$T/wb.state"
out="$(wb_watch 6 1)"; rc=$?
check "a W&B run finished short of its target is FAILED" '[ "$rc" = 1 ] && case $out in *"short of target"*) true ;; *) false ;; esac'
echo "state=running it=1200 age=7200" > "$T/wb.state"
out="$(wb_watch 6 20)"; rc=$?
check "a W&B run that stopped logging is DEAD" '[ "$rc" = 1 ] && case $out in *"logged nothing"*) true ;; *) false ;; esac'
: > "$T/wb.state"
out="$(wb_watch 10 0)"; rc=$?
check "with W&B silent too, the watch still ends BLIND" '[ "$rc" = 1 ] && case $out in *BLIND*) true ;; *) false ;; esac'

[ "$fails" -eq 0 ] || exit 1
echo "all watch_run exit tests passed"
