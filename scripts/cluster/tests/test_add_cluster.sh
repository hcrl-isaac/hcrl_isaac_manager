#!/usr/bin/env bash
# `add --update` with every default accepted on a profile whose paths are written through a variable: the paths must
# stay as written (and expand to the same directories), and .backup/ must hold the profile just written.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/scripts/cluster/config/zz" "$T/scripts/cluster/tools"
cp "$REPO/scripts/cluster/add_cluster.sh" "$T/scripts/cluster/"
cp -r "$REPO/scripts/cluster/tools/." "$T/scripts/cluster/tools/"
P="$T/scripts/cluster/config/zz"
cat > "$P/.env.cluster" <<'EOF'
BASE=/work/me
CLUSTER_LOGIN=me@login.example.edu
CLUSTER_ISAACLAB_DIR=${BASE}/isaaclab
CLUSTER_SIF_PATH=${BASE}/scratch
EOF
printf '#!/usr/bin/env bash\ncat <<EOT > job.sh\n#SBATCH -p q\n#SBATCH -n 2\n#SBATCH --cpus-per-task=4\nEOT\n' > "$P/submit_job_slurm.sh"
expanded() { bash -c 'source "$1"; printf "%s %s" "$CLUSTER_ISAACLAB_DIR" "$CLUSTER_SIF_PATH"' _ "$P/.env.cluster"; }
before="$(expanded)"

printf '\n\n\n\n\n\n\n\n\nme@x.edu\n' | bash "$T/scripts/cluster/add_cluster.sh" --update zz > "$T/out" 2>&1
fails=0
check() {
    if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}
check "update succeeds" "grep -q 'Wrote cluster profile' '$T/out'"
check "variable paths kept as written" "grep -qx 'CLUSTER_ISAACLAB_DIR=\${BASE}/isaaclab' '$P/.env.cluster'"
check "paths expand as before" "[ \"\$(expanded)\" = '$before' ]"
check ".backup/ holds the profile just written" "cmp -s '$P/.env.cluster' '$P/.backup/.env.cluster'"
if [ "$fails" -ne 0 ]; then
    echo "--- output"; cat "$T/out"; echo "--- env"; cat "$P/.env.cluster"
    exit 1
fi
echo "all checks passed"
