#!/usr/bin/env bash
# The free-space pre-flight and CLUSTER_TREES_DIR through ssh/quota stubs: a full quota refuses `develop stage`
# before anything is uploaded, --no-space-check bypasses it, and trees land in CLUSTER_TREES_DIR.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/scripts/cluster/config/zz" "$T/remote/resources/hcrl_robots" "$T/work"
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
exec bash -c "$*"
EOF
# Delta's quota table: the remote dir is under a 100G-soft quota with $T/used in use
# df reports ample room, so the quota table decides (the test machine's own disk is not the point)
printf '#!/usr/bin/env bash\necho "Filesystem 1024-blocks Used Available Capacity Mounted"\necho "fake 9999999999 0 9999999999 0%% /"\n' > "$T/bin/df"
cat > "$T/bin/quota" <<EOF
#!/usr/bin/env bash
echo "| Filesystem | Usage | Quota | Limit |"
echo "| $T/remote | \$(cat $T/used) | 102400M | 103G |"
EOF
chmod +x "$T/bin/"*
L="$T/local"
G="$L/resources/hcrl_isaaclab"
mkdir -p "$G" "$T/remote/resources/hcrl_isaaclab"
(cd "$G" && git init -q -b main . && git config user.email t@t && git config user.name t &&
    echo x > code.py && git add -A && git commit -qm init)
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\n' "$T/remote" > "$T/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -p test\n' > "$T/scripts/cluster/config/zz/submit_job_slurm.sh"
dev() { PATH="$T/bin:$PATH" HOME="$T" CLUSTER=zz LOCAL_ISAACLAB_DIR="$L" bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" "$@"; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

echo 98000M > "$T/used"
check "free space is read from the quota table" "[ \"\$(dev __free_gb '$T/remote/trees')\" = 4 ]"
dev stage full hcrl_isaaclab=main > "$T/out1" 2>&1
check "a nearly full quota refuses the stage" "[ $? -ne 0 ] && grep -q 'only 4 GB free for staged trees' '$T/out1' && ! ls -d '$T/remote/trees/full-'* 2>/dev/null"
dev stage --no-space-check full hcrl_isaaclab=main > "$T/out2" 2>&1
check "--no-space-check bypasses it" "[ $? -eq 0 ] && ls -d '$T/remote/trees/full-'* > /dev/null"

echo 50000M > "$T/used"
printf 'CLUSTER_TREES_DIR=%s\nCLUSTER_MIN_FREE_GB=0\n' "$T/work/trees" >> "$T/scripts/cluster/config/zz/.env.cluster"  # df of the test disk is not the point here
dev stage moved hcrl_isaaclab=main > "$T/out3" 2>&1
check "CLUSTER_TREES_DIR receives new trees" "[ $? -eq 0 ] && [ -f \"\$(ls -d '$T/work/trees/moved-'*)/.complete\" ]"
check "and resolve_tree finds them there" "dev __resolve_tree moved 2>/dev/null | grep -q '^$T/work/trees/moved-'"

if [ "$fails" -ne 0 ]; then
    for f in "$T"/out*; do echo "--- $f"; tail -5 "$f"; done
    exit 1
fi
echo "all checks passed"
