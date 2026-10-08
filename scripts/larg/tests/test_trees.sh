#!/usr/bin/env bash
# scripts/larg/trees.sh through an ssh stub against a fake box: a ref lands as a tree beside the workspace, unstaged repos
# link to it, the artifact root is the workspace's, and rm keeps a tree a live LARG run uses.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"' EXIT
# a copy of the scripts, so nothing beside the real ones (scripts/.env.*) is shipped
mkdir -p "$T/bin" "$T/scripts/cluster"
cp -r "$REPO/scripts/larg" "$T/scripts/"
cp -r "$REPO/scripts/cluster/cluster_dev" "$T/scripts/cluster/"
cat > "$T/bin/ssh" <<'EOF'
#!/usr/bin/env bash
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-J|-i|-p|-l|-F) shift 2 ;;
        -*) shift ;;
        *) shift; break ;;
    esac
done
exec bash -c "$*"
EOF
chmod +x "$T/bin/ssh"

L="$T/local"
R="$T/box"
G="$L/resources/hcrl_isaaclab"
mkdir -p "$G" "$R/resources/hcrl_isaaclab/.artifacts" "$R/resources/hcrl_robots"
(
    cd "$G" && git init -q -b main . && git config user.email t@t && git config user.name t
    echo main > code.py && touch setup.py && git add -A && git commit -qm main
    git switch -q -c feat && echo feat > code.py && git commit -qam feat && git switch -q main
)
cp "$G/code.py" "$R/resources/hcrl_isaaclab/"
echo asset > "$R/resources/hcrl_robots/t1.urdf"
larg() { env PATH="$T/bin:$PATH" LARG_REMOTE_DIR="$R" LARG_LOCAL_DIR="$L" bash "$T/scripts/larg/trees.sh" "$@"; }

fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

larg stage hazard t1 hcrl_isaaclab=feat > "$T/out1" 2>&1
check "stage exits 0" "[ $? -eq 0 ]"
tree="$(ls -d "$R"/trees/t1-* 2>/dev/null | head -1)"
check "the tree lands beside the box's workspace, complete" "[ -n '$tree' ] && [ -f '$tree/.complete' ]"
check "it holds the ref's content" "grep -qx feat '$tree/resources/hcrl_isaaclab/code.py'"
check "unstaged repos link to the workspace" "[ -L '$tree/resources/hcrl_robots' ] && [ -f '$tree/resources/hcrl_robots/t1.urdf' ]"
check "the artifact root is the workspace's" \
    "[ \"\$(readlink '$tree/resources/hcrl_isaaclab/.artifacts')\" = '$R/resources/hcrl_isaaclab/.artifacts' ]"
check "the closing hint names train.sh --tree with the tree and host" "grep -q 'Run with: scripts/larg/train.sh --tree t1-[0-9a-f]* hazard <task>' '$T/out1'"
larg list hazard > "$T/out2" 2>&1
check "list shows the tree and its manifest" "grep -q '^t1-' '$T/out2' && grep -q 'hcrl_isaaclab feat' '$T/out2'"

id="$(basename "$tree")"
mkdir -p "$tree/.in-use"
sleep 30 & live=$!
touch "$tree/.in-use/larg.$live"
larg rm hazard "$id" > "$T/out3" 2>&1
check "rm keeps a tree a live LARG run uses" "[ -d '$tree' ] && grep -q 'in use by larg.$live' '$T/out3'"
kill "$live" 2>/dev/null; wait "$live" 2>/dev/null
larg rm hazard "$id" > "$T/out4" 2>&1
check "and removes it once that run is gone" "[ ! -e '$tree' ]"

if [ "$fails" -ne 0 ]; then
    for f in "$T"/out*; do echo "--- $f"; tail -15 "$f"; done
    exit 1
fi
echo "all checks passed"
