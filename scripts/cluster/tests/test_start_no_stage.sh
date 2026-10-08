#!/usr/bin/env bash
# `develop start` stages the workspace as the tree `default` before submitting the sentinel; --no-stage skips it.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
cleanup() {
    [ -f "$T/.cluster_dev/zz/watch.pid" ] && kill "$(cat "$T/.cluster_dev/zz/watch.pid")" 2>/dev/null
    chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"
}
trap cleanup EXIT
mkdir -p "$T/bin" "$T/scripts/cluster/config/zz" "$T/remote" "$T/local/resources/hcrl_isaaclab"
cp -r "$REPO/scripts/cluster/cluster_dev" "$REPO/scripts/cluster/tools" "$T/scripts/cluster/"
cat > "$T/bin/ssh" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [ "$a" = "-O" ] && exit 0; done
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-J|-i|-p|-l|-F|-E|-c|-m|-L|-R|-D|-W|-b|-e|-S|-w|-Q|-B) shift 2 ;;
        -*) shift ;;
        *) shift; break ;;
    esac
done
cd "$HOME" && exec bash -c "$*"  # remote commands run from the remote home
EOF
printf '#!/usr/bin/env bash\necho 4242\n' > "$T/bin/sbatch"
printf '#!/usr/bin/env bash\necho PENDING\n' > "$T/bin/squeue"
chmod +x "$T/bin/"*
(
    cd "$T/local/resources/hcrl_isaaclab" && git init -q -b main . && git config user.email t@t && git config user.name t
    echo code > a.py && git add -A && git commit -qm main && echo edited > a.py
)
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\nCLUSTER_MIN_FREE_GB=0\n' "$T/remote" > "$T/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -p test\n' > "$T/scripts/cluster/config/zz/submit_job_slurm.sh"
dev() { PATH="$T/bin:$PATH" HOME="$T" CLUSTER=zz LOCAL_ISAACLAB_DIR="$T/local" POLL_SECONDS=3600 \
    bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" "$@"; }
stop_watcher() { [ -f "$T/.cluster_dev/zz/watch.pid" ] && kill "$(cat "$T/.cluster_dev/zz/watch.pid")" 2>/dev/null; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

dev start --no-stage > "$T/out1" 2>&1
stop_watcher
check "--no-stage submits the sentinel" "grep -qx 'JOBID=4242' '$T/.cluster_dev/zz/state'"
check "--no-stage ships no code" "[ ! -e '$T/remote/trees' ] && [ ! -e '$T/remote/resources/hcrl_isaaclab' ]"
check "--no-stage says so" "grep -q 'Skipping the stage' '$T/out1'"
dev start --no-sync > "$T/out3" 2>&1
check "a retired or misspelled flag is refused before anything runs" \
    "[ $? -ne 0 ] && grep -q \"unknown argument '--no-sync'\" '$T/out3' && [ ! -e '$T/remote/trees' ]"
check "nothing lands in the caller's directory" "[ ! -e '$REPO/.cluster_dev_sentinel.sbatch' ] && [ -e '$T/.cluster_dev_sentinel.sbatch' ]"

dev start > "$T/out2" 2>&1
stop_watcher
check "plain start stages the workspace as default" "grep -qx edited \"\$(ls -d '$T'/remote/trees/default-* | head -1)/resources/hcrl_isaaclab/a.py\""
check "and nothing is mirrored onto the shared workspace" "[ ! -e '$T/remote/resources/hcrl_isaaclab' ]"
check "exec then runs that tree" "dev __resolve_tree default 2>/dev/null | grep -q '/trees/default-[0-9a-f]\{10\}$'"

if [ "$fails" -ne 0 ]; then
    for f in "$T"/out*; do echo "--- $f"; tail -10 "$f"; done
    exit 1
fi
echo "all checks passed"
