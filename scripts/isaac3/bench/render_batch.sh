#!/bin/bash
# Several RTX renders in a row on a render box.
# usage: render_batch.sh ROOT GPU NAME=CHECKPOINT:PHYSICS [...]   -> ROOT/renders/NAME/clip_0000.mp4
ROOT=$1; GPU=$2; shift 2
D=$(dirname "$(readlink -f "$0")")
for spec in "$@"; do
  name=${spec%%=*}; rest=${spec#*=}; ckpt=${rest%%:*}; phys=${rest##*:}
  echo "== $name"; bash $D/render_remote.sh $ROOT $GPU $ckpt $ROOT/renders/$name $phys
done
