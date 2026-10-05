#!/usr/bin/env bash
# `develop stage` / `trees` through an ssh stub: named repos at named refs land in a new immutable tree, everything
# else links to the shared checkout, bad input touches nothing, and trees resolve and are removed safely.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"' EXIT
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
printf '#!/usr/bin/env bash\ncat "%s/running_jobs" 2>/dev/null\n' "$T" > "$T/bin/squeue"
chmod +x "$T/bin/ssh" "$T/bin/squeue"

L="$T/local"
R="$T/remote"
G="$L/resources/hcrl_isaaclab"
mkdir -p "$G" "$R/resources/hcrl_isaaclab" "$R/resources/hcrl_robots"
(
    cd "$G" && git init -q -b main . && git config user.email t@t && git config user.name t
    printf 'logs/\n' > .gitignore
    echo main > code.py && head -c 100000 /dev/urandom > big.bin && git add -A && git commit -qm main
    git switch -q -c feat && echo feat > code.py && git commit -qam feat && git switch -q main
)
cp "$G/code.py" "$G/big.bin" "$R/resources/hcrl_isaaclab/"
touch -d 2020-01-01 "$R/resources/hcrl_isaaclab/big.bin"  # an old copy: linking must go by content, not mtime
echo asset > "$R/resources/hcrl_robots/t1.urdf"
printf 'CLUSTER_ISAACLAB_DIR=%s\nCLUSTER_LOGIN=fake@host\nCLUSTER_SIF_PATH=/x\n' "$R" > "$T/scripts/cluster/config/zz/.env.cluster"
printf '#!/usr/bin/env bash\n#SBATCH -p test\n' > "$T/scripts/cluster/config/zz/submit_job_slurm.sh"
dev() { PATH="$T/bin:$PATH" HOME="$T" CLUSTER=zz LOCAL_ISAACLAB_DIR="$L" bash "$T/scripts/cluster/cluster_dev/cluster_dev.sh" "$@"; }
ntrees() { ls -d "$R"/trees/"$1"-* 2>/dev/null | grep -c . ; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

dev stage t1 hcrl_isaaclab=feat > "$T/out1" 2>&1
check "stage exits 0" "[ $? -eq 0 ]"
tree="$(ls -d "$R"/trees/t1-* 2>/dev/null | head -1)"
check "tree created and complete" "[ -n '$tree' ] && [ -f '$tree/.complete' ]"
check "tree has the ref's content" "grep -qx feat '$tree/resources/hcrl_isaaclab/code.py'"
check "manifest records the commit" "grep -q \"hcrl_isaaclab feat \$(git -C '$G' rev-parse feat) local-ref\" '$tree/MANIFEST'"
check "manifest lists shared repos as live" "grep -q '^hcrl_robots shared-live ' '$tree/MANIFEST'"
check "unstaged repos link to the shared checkout" "[ -L '$tree/resources/hcrl_robots' ] && [ -f '$tree/resources/hcrl_robots/t1.urdf' ]"
check "staged repo has writable mount points" "[ -d '$tree/resources/hcrl_isaaclab/logs' ] && [ -d '$tree/resources/hcrl_isaaclab/outputs' ] && [ -d '$tree/resources/hcrl_isaaclab/wandb' ]"
check "tree carries its own node_exec.sh" "[ -f '$tree/scripts/cluster/cluster_dev/node_exec.sh' ]"
check "shared checkout untouched" "grep -qx main '$R/resources/hcrl_isaaclab/code.py'"
check "identical file hardlinked, not re-sent" \
    "[ \"\$(stat -c %i '$tree/resources/hcrl_isaaclab/big.bin')\" = \"\$(stat -c %i '$R/resources/hcrl_isaaclab/big.bin')\" ]"
check "temporary checkout cleaned up" "[ \"\$(git -C '$G' worktree list | wc -l)\" -eq 1 ]"
check "run hint names the full tree id" "grep -q \"exec --tree \$(basename '$tree')\" '$T/out1'"

dev stage t1 hcrl_isaaclab=feat > "$T/out2" 2>&1
check "same refs reuse the tree" "grep -q 'already staged' '$T/out2' && [ \$(ntrees t1) -eq 1 ]"

rm "$R/resources/hcrl_isaaclab/big.bin"
dev stage t3 hcrl_isaaclab=feat > "$T/out_t3" 2>&1
check "earlier trees serve as hardlink sources too" \
    "[ \"\$(stat -c %i \"\$(ls -d '$R'/trees/t3-*)/resources/hcrl_isaaclab/big.bin\")\" = \"\$(stat -c %i '$tree/resources/hcrl_isaaclab/big.bin')\" ]"

# B1: names are validated before anything is touched
for spec in "../../../resources/hcrl_isaaclab=feat" "=feat" "hcrl_isaacla=feat" "hcrl_isaaclab=feat hcrl_isaaclab=main"; do
    # shellcheck disable=SC2086
    dev stage bad $spec > "$T/out_b1" 2>&1
    check "rejects spec '$spec'" "[ $? -ne 0 ] && [ \$(ntrees bad) -eq 0 ]"
done
dev stage "../x" hcrl_isaaclab=feat > "$T/out_b1" 2>&1
check "rejects a tree name with /" "[ $? -ne 0 ]"
check "bad input left the shared checkout alone" "grep -qx main '$R/resources/hcrl_isaaclab/code.py' && [ ! -e '$R/resources/hcrl_isaaclab/hcrl_isaaclab' ]"

# B2: untracked files are uploaded and fingerprinted
echo v1 > "$G/new_mod.py"
dev stage wt "hcrl_isaaclab=$G" > "$T/out_b2" 2>&1
echo v2 > "$G/new_mod.py"
dev stage wt "hcrl_isaaclab=$G" > "$T/out_b2b" 2>&1
check "an untracked edit is a new tree" "[ \$(ntrees wt) -eq 2 ]"
check "the newest tree carries it" "grep -qx v2 \"\$(ls -td '$R'/trees/wt-*/ | head -1)resources/hcrl_isaaclab/new_mod.py\""
check "ignored files are not uploaded" "mkdir -p '$G/logs' && touch '$G/logs/run.txt' && dev stage wt2 'hcrl_isaaclab=$G' >/dev/null 2>&1 && \
    [ ! -e \"\$(ls -d '$R'/trees/wt2-* | head -1)/resources/hcrl_isaaclab/logs/run.txt\" ]"
rm -f "$G/new_mod.py"

# B3: resolution is exact and refuses ambiguity
dev stage r hcrl_isaaclab=main > /dev/null 2>&1
dev stage r-x hcrl_isaaclab=main > /dev/null 2>&1
check "one tree named r resolves" "dev __resolve_tree r 2>/dev/null | grep -q '/trees/r-[0-9a-f]\{10\}$'"
dev stage r hcrl_isaaclab=feat > /dev/null 2>&1
check "two trees named r are refused" "! dev __resolve_tree r > /dev/null 2>&1"
check "the full id still resolves" "dev __resolve_tree \"\$(basename \"\$(ls -d '$R'/trees/r-* | grep -v r-x | head -1)\")\" > /dev/null 2>&1"
check "a prefix-sharing name is not matched" "dev __resolve_tree r-x 2>/dev/null | grep -q '/trees/r-x-'"

# B5: failures clean up and never report success
dev stage multi hcrl_isaaclab=feat hcrl_robots=nope > "$T/out_b5" 2>&1
check "a failing spec fails the stage" "[ $? -ne 0 ] && [ \$(ntrees multi) -eq 0 ]"
check "it leaves no temporary checkout" "[ \"\$(git -C '$G' worktree list | wc -l)\" -eq 1 ]"
check "it leaves no partial" "! ls -d '$R'/trees/*.partial.* 2>/dev/null"
chmod 555 "$R/trees"
dev stage ro hcrl_isaaclab=feat > "$T/out_b5b" 2>&1
rc=$?
chmod 755 "$R/trees"
check "an upload failure is not reported as staged" "[ $rc -ne 0 ] && ! grep -q '^\[cluster_dev\] Staged' '$T/out_b5b'"

# concurrent stages of the same refs end in one complete tree
dev stage conc hcrl_isaaclab=feat > "$T/out_c1" 2>&1 &
dev stage conc hcrl_isaaclab=feat > "$T/out_c2" 2>&1 &
wait
check "concurrent stages give one tree" "[ \$(ntrees conc) -eq 1 ] && [ -f \"\$(ls -d '$R'/trees/conc-*)/.complete\" ] && ! ls -d '$R'/trees/conc-*.partial.* 2>/dev/null"

dev trees > "$T/out5" 2>&1
check "trees lists them" "[ \"\$(grep -c '^t1-' '$T/out5')\" -eq 1 ]"

# trees rm: refused while a job that used the tree is running
id="$(basename "$tree")"
mkdir -p "$tree/.in-use" && touch "$tree/.in-use/4242"
echo 4242 > "$T/running_jobs"
dev trees rm "$id" > "$T/out6" 2>&1
check "rm refuses a tree in use" "[ $? -ne 0 ] && [ -d '$tree' ]"
: > "$T/running_jobs"
dev trees rm "$id" > "$T/out6" 2>&1
check "rm removes it once the job ended" "[ $? -eq 0 ] && [ ! -e '$tree' ]"
check "rm needs the full id" "! dev trees rm r > /dev/null 2>&1"
mkdir -p "$R/trees/old.partial.1" "$R/trees/new.partial.2" && touch -d '2 hours ago' "$R/trees/old.partial.1"
dev trees rm --partials > /dev/null 2>&1
check "rm --partials removes only stale partials" "[ ! -e '$R/trees/old.partial.1' ] && [ -e '$R/trees/new.partial.2' ]"

dev sync --dry-run > "$T/out7" 2>&1
check "develop sync leaves trees alone" "! grep -q '^\*deleting *trees/' '$T/out7'"

if [ "$fails" -ne 0 ]; then
    ls -la "$R/trees"
    for f in "$T"/out*; do echo "--- $f"; tail -15 "$f"; done
    exit 1
fi
echo "all checks passed"
