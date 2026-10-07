#!/bin/bash
# Kit RTX render of a checkpoint's fixed-command rollout on a render box (venv built with EXTRA_EXTRAS="isaacsim video").
# usage: [NUM_ENVS=16] render_remote.sh ROOT GPU CHECKPOINT OUT_DIR [PHYSICS]   -> OUT_DIR/clip_0000.mp4 + OUT_DIR.npz
ROOT=$(readlink -f "$1"); GPU=$2; CKPT=$3; OUT=$4; PHYS=${5:-newton_mjwarp}
mkdir -p $ROOT/tmp/render $(dirname "$OUT")
export TMPDIR=$ROOT/tmp/render CUDA_VISIBLE_DEVICES=$GPU XDG_CACHE_HOME=$ROOT/tmp/render/cache OMNI_KIT_ACCEPT_EULA=YES
rm -rf "$OUT"
cd $ROOT/tmp/render
$ROOT/venv/bin/python $(dirname "$(readlink -f "$0")")/../probe/record_rollout.py "$CKPT" "$OUT.npz" physics=$PHYS \
  --num_envs ${NUM_ENVS:-1} --rtx_video "$OUT" > "$OUT.log" 2>&1
echo "exit $?"; grep -E "RECORDED|Traceback|vkclamp" "$OUT.log" | tail -3
