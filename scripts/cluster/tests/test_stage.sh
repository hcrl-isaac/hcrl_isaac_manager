#!/usr/bin/env bash
# `develop stage` through an ssh stub: named repos at named refs land in a new immutable tree, everything else links
# to the shared checkout, the shared checkout is untouched, and `develop sync` leaves trees alone.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/scripts/cluster/config/zz"
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
chmod +x "$T/bin/ssh"

L="$T/local"
R="$T/remote"
G="$L/resources/hcrl_isaaclab"
mkdir -p "$G" "$R/resources/hcrl_isaaclab" "$R/resources/hcrl_robots"
(
    cd "$G" && git init -q -b main . && git config user.email t@t && git config user.name t
    echo main > code.py && head -c 100000 /dev/urandom > big.bin && git add -A && git commit -qm main
    git switch -q -c feat && echo feat > code.py && git commit -qam feat && git switch -q main
)
cp "$G/code.py" "$G/big.bin" "$R/resources/hcrl_isaaclab/"
echo asset > "$R/resources/hcrl_robots/t1.urdf"
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\n' "$R" > "$T/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -p test\n' > "$T/scripts/cluster/config/zz/submit_job_slurm.sh"
dev() { PATH="$T/bin:$PATH" HOME="$T" CLUSTER=zz LOCAL_ISAACLAB_DIR="$L" bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" "$@"; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

dev stage t1 hcrl_isaaclab=feat > "$T/out1" 2>&1
check "stage exits 0" "[ $? -eq 0 ]"
tree="$(ls -d "$R"/trees/t1-* 2>/dev/null | head -1)"
check "tree created and complete" "[ -n '$tree' ] && [ -f '$tree/.complete' ]"
check "tree has the ref's content" "grep -qx feat '$tree/resources/hcrl_isaaclab/code.py'"
check "manifest records the commit" "grep -q \"hcrl_isaaclab feat \$(git -C '$G' rev-parse feat)\" '$tree/MANIFEST'"
check "unstaged repos link to the shared checkout" "[ -L '$tree/resources/hcrl_robots' ] && [ -f '$tree/resources/hcrl_robots/t1.urdf' ]"
check "tree carries its own node_exec.sh" "[ -f '$tree/scripts/cluster/cluster_dev/node_exec.sh' ]"
check "shared checkout untouched" "grep -qx main '$R/resources/hcrl_isaaclab/code.py'"
check "identical file hardlinked, not re-sent" \
    "[ \"\$(stat -c %i '$tree/resources/hcrl_isaaclab/big.bin')\" = \"\$(stat -c %i '$R/resources/hcrl_isaaclab/big.bin')\" ]"
check "temporary checkout cleaned up" "[ \"\$(git -C '$G' worktree list | wc -l)\" -eq 1 ]"

dev stage t1 hcrl_isaaclab=feat > "$T/out2" 2>&1
check "same refs reuse the tree" "grep -q 'already staged' '$T/out2' && [ \"\$(ls -d '$R'/trees/t1-* | wc -l)\" -eq 1 ]"

echo dirty >> "$G/code.py"
dev stage t1 "hcrl_isaaclab=$G" > "$T/out3" 2>&1
check "a dirty worktree is a new tree" "[ \"\$(ls -d '$R'/trees/t1-* | wc -l)\" -eq 2 ]"
check "the dirty edit is in it" "grep -l dirty '$R'/trees/t1-*/resources/hcrl_isaaclab/code.py | grep -q ."

rm "$R/resources/hcrl_isaaclab/big.bin"
dev stage t3 hcrl_isaaclab=feat > "$T/out7" 2>&1
t3="$(ls -d "$R"/trees/t3-* | head -1)"
check "earlier trees serve as hardlink sources too" \
    "[ \"\$(stat -c %i '$t3/resources/hcrl_isaaclab/big.bin')\" = \"\$(stat -c %i '$tree/resources/hcrl_isaaclab/big.bin')\" ]"

dev stage t2 hcrl_isaaclab=no-such-branch > "$T/out4" 2>&1
check "unknown ref fails without a tree" "[ $? -ne 0 ] && grep -q 'unknown ref' '$T/out4' && ! ls -d '$R'/trees/t2-* 2>/dev/null"

dev trees > "$T/out5" 2>&1
check "trees lists them" "[ \"\$(grep -c '^t1-' '$T/out5')\" -eq 2 ]"

dev sync --dry-run > "$T/out6" 2>&1
check "develop sync leaves trees alone" "! grep -q '^\*deleting *trees/' '$T/out6'"

if [ "$fails" -ne 0 ]; then
    ls -li "$R"/trees/*/resources/hcrl_isaaclab/big.bin
    for f in "$T"/out*; do echo "--- $f"; tail -15 "$f"; done
    exit 1
fi
echo "all checks passed"
