#!/usr/bin/env bash
# trees.sh's data push through an ssh stub: new data files reach the cluster's shared copy, a Git LFS pointer never
# overwrites the real file there, and the data repo's worktrees never ship.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin"
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
export PATH="$T/bin:$PATH"

L="$T/local/resources/motion_datasets"
R="$T/cluster/resources/motion_datasets"
mkdir -p "$L/ibfm" "$L/amass" "$L/worktrees/x" "$R/amass"
head -c 5000 /dev/urandom > "$L/ibfm/new.pt"
echo '{}' > "$L/ibfm/new.manifest.json"
printf 'version https://git-lfs.github.com/spec/v1\noid sha256:%064d\nsize 20\n' 0 > "$L/amass/big.pt"
head -c 3000 /dev/urandom > "$L/worktrees/x/wt.pt"
echo notes > "$L/README.md"
head -c 4000 /dev/urandom > "$R/amass/big.pt"
cp "$R/amass/big.pt" "$T/big.orig"

log() { :; }
err() { echo "$*" >&2; }
LOCAL_ISAACLAB_DIR="$T/local" REMOTE_ISAACLAB_DIR="$T/cluster" CLUSTER_LOGIN=login SSH_OPTS=()
# shellcheck source=/dev/null
source "$REPO/scripts/cluster/cluster_dev/trees.sh"

fails=0
check() {
    if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}
_push_data > /dev/null 2>&1
check "a new data file ships" 'cmp -s "$L/ibfm/new.pt" "$R/ibfm/new.pt"'
check "its manifest ships" '[ -f "$R/ibfm/new.manifest.json" ]'
check "an LFS pointer leaves the real file alone" 'cmp -s "$T/big.orig" "$R/amass/big.pt"'
check "worktrees never ship" '[ ! -e "$R/worktrees" ]'
check "non-data files never ship" '[ ! -e "$R/README.md" ]'
[ "$fails" -eq 0 ]
