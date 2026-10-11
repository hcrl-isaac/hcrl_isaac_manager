#!/usr/bin/env bash
# Launch a training run on a LARG box from its synced workspace, detached (setsid nohup).
#
# A40 runs record video in-process (--video on; the Vulkan clamp layer in the LARG home lets them render); the A100s
# lack RT cores, so their runs pass --video async and render elsewhere. Extra train.py flags pass via `--` (a --video
# there wins). One GPU runs train.py directly; LARG_NPROC > 1 runs it under torchrun with --distributed. --tree runs
# from a code tree staged on the box (scripts/larg/trees.sh stage): its repos first on PYTHONPATH, the workspace's ilab.
#
# Usage:
#   scripts/larg/train.sh [--tree <name>] <host> <task> <run_name> [run_group] [num_envs] [-- extra train.py args]
# Poll:
#   scripts/larg/train.sh --log <host> <task>
# Env:
#   LARG_NPROC            GPUs for the run (default 1)
#   CUDA_VISIBLE_DEVICES  physical GPUs to pin the run to (e.g. 2 or 0,1); also tags the run dir
#   LARG_SCRATCH          per-box scratch for run logs and Kit caches (default /var/local/$LARG_USER)
#   LARG_HOLDER           your session name: lease the pinned cards with `pls res claim` before launching

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

tree=""
if [ "${1:-}" = "--tree" ]; then tree="${2:-}"; shift 2 || true; [ -n "$tree" ] || { echo "--tree needs a name"; exit 1; }; fi

TRAIN="resources/hcrl_isaaclab/scripts/train.py"
NPROC="${LARG_NPROC:-1}"
RUNS="${LARG_SCRATCH:-/var/local/$LARG_USER}/larg-runs"

if [ "${1:-}" = "--log" ]; then
  shift; host="$1"; task="$2"
  larg_ssh "$host" "ls -td $RUNS/*${task//\//_}* 2>/dev/null | head -1 | xargs -r -I{} tail -n 40 {}/train.log; \
    echo '--- proc ---'; pgrep -af '[t]orch.distributed.run|[t]rain.py' | head || echo '(no train proc)'; \
    echo '--- gpu ---'; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
  exit 0
fi

# positionals end at `--`, so an omitted run_group or num_envs never takes the separator as its value
pos=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do pos+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
extra=("$@")
host="${pos[0]:-}"; task="${pos[1]:-}"; run_name="${pos[2]:-}"; run_group="${pos[3]:-larg}"; num_envs="${pos[4]:-}"
[ -n "$host" ] && [ -n "$task" ] && [ -n "$run_name" ] && [ "${#pos[@]}" -le 5 ] || {
  echo "usage: $0 [--tree <name>] <host> <task> <run_name> [run_group] [num_envs] [-- extra]"; exit 1; }
[[ -z "$num_envs" || "$num_envs" =~ ^[0-9]+$ ]] || { echo "[larg] num_envs must be a number, got '$num_envs'"; exit 1; }

video=async
for h in "${LARG_A40_HOSTS[@]}"; do [ "${host%%.*}" = "$h" ] && video=on; done
args=(--video "$video" --task "$task" --run_name "$run_name" --run_group "$run_group")
[ -n "$num_envs" ] && args+=(--num_envs "$num_envs")
args+=("${extra[@]}")
if [ "$NPROC" -gt 1 ]; then
  launch=("\$WS/ilab/bin/python" -m torch.distributed.run --standalone --nnodes=1 "--nproc_per_node=$NPROC" "$TRAIN" --distributed)
else
  launch=("\$WS/ilab/bin/python" "$TRAIN")
fi

gpus="${CUDA_VISIBLE_DEVICES:-}"
if [ -n "${LARG_HOLDER:-}" ]; then
  [ -n "$gpus" ] || { echo "[larg] LARG_HOLDER needs CUDA_VISIBLE_DEVICES to name the cards to lease"; exit 1; }
  cards=(); for g in ${gpus//,/ }; do cards+=("${host%%.*}:$g"); done
  python3 "$HERE/../cluster/res/res.py" claim "${cards[@]}" --holder "$LARG_HOLDER" --note "$run_name" || exit 1
else
  echo "[larg] WARNING: LARG_HOLDER unset, so the run's cards are not leased; other sessions see them only as busy, and its Kit cache starts cold"
fi
tag="${task//\//_}_$(date +%Y%m%d-%H%M%S)${gpus:+_gpu${gpus//,/-}}"
run_dir="$RUNS/$tag"
# per run: TMPDIR and the log. Kit caches are per GPU set when the cards are leased (one run per card), else per run
if [ -n "${LARG_HOLDER:-}" ]; then cache="${RUNS%/*}/kit-cache/gpu${gpus}"; else cache="$run_dir/kit-cache"; fi
q() { printf '%q ' "$@"; }
# A tree run holds the tree's in-use mark (larg.<pid>, which trees.sh rm checks on the box) for as long as the run
# lasts: the command runs as a child, a TERM/INT/HUP to this group leader is passed on, and the mark goes only once the
# child has exited. A run without a tree is the command itself, so its pid is python's.
# shellcheck disable=SC2016
tree_wrap='m="$1.$$"; shift; touch "$m"; "$@" & c=$!; trap '"'"'kill -TERM "$c" 2>/dev/null'"'"' TERM INT HUP
while kill -0 "$c" 2>/dev/null; do wait "$c"; rc=$?; done; rm -f "$m"; exit "${rc:-1}"'
# the launch line keeps $WS unexpanded: the box's workspace path is only known there
cmd_line="$(q "${launch[@]}" "${args[@]}" | sed 's/\\\$WS/$WS/g')"
remote="set -u
WS=$(larg_remote_path)
mkdir -p $(q "$run_dir/tmp" "$cache/xdg" "$cache/omni") || exit 1
cd \"\$WS\" || { echo \"no workspace at \$WS\"; exit 1; }
export PATH=\$HOME/.local/bin:\$PATH ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export TMPDIR=$(q "$run_dir/tmp") XDG_CACHE_HOME=$(q "$cache/xdg") OMNI_CACHE_DIR=$(q "$cache/omni")
${gpus:+export CUDA_VISIBLE_DEVICES=$(q "$gpus")}
gomp=\"\$(ls \$WS/ilab/lib/python3.11/site-packages/torch/lib/libgomp-*.so.1 2>/dev/null | head -1)\"
[ -n \"\$gomp\" ] && export LD_PRELOAD=\"\$gomp\${LD_PRELOAD:+:\$LD_PRELOAD}\"
set -a; source scripts/.env.wandb 2>/dev/null; set +a
marker=/dev/null
if [ -n $(printf %q "${tree:-}") ]; then
  # the newest complete tree by that name (or the exact <name>-<fingerprint>)
  T=\$(ls -1dt \"\$WS/trees/\"$(printf %q "$tree") \"\$WS/trees/\"$(printf %q "$tree")-* 2>/dev/null | grep -v '\\.partial\\.' |
    while read -r d; do [ -f \"\$d/.complete\" ] && { echo \"\$d\"; break; }; done)
  [ -n \"\$T\" ] || { echo \"no staged tree $tree under \$WS/trees (scripts/larg/trees.sh stage)\"; exit 1; }
  cd \"\$T\" || exit 1
  pp=\"\"
  for d in resources/*/; do
    [ -L \"\${d%/}\" ] && continue
    [ -f \"\${d}setup.py\" ] || [ -f \"\${d}pyproject.toml\" ] || continue
    pp=\"\$T/\${d%/}\${pp:+:\$pp}\"
  done
  export PYTHONPATH=\"\$pp\${PYTHONPATH:+:\$PYTHONPATH}\"
  mkdir -p \"\$T/.in-use\" && marker=\"\$T/.in-use/larg\"
  echo \"tree: \$T\"; echo \"PYTHONPATH: \$pp\"
fi
if [ \"\$marker\" = /dev/null ]; then
  setsid nohup $cmd_line > $(q "$run_dir/train.log") 2>&1 < /dev/null &
  echo started pid \$!
else
  setsid nohup bash -c $(q "$tree_wrap") _ \"\$marker\" $cmd_line > $(q "$run_dir/train.log") 2>&1 < /dev/null &
  echo started pid \$!; echo \"stop: kill -- -\$!\"
fi
echo log: $(q "$run_dir/train.log")"

echo "=== train $task ($run_name) on $host: ${NPROC} GPU(s)${gpus:+ [$gpus]} ==="
larg_ssh "$host" "bash -lc $(printf %q "$remote")"
