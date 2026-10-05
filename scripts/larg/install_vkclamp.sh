#!/usr/bin/env bash
# Install the Vulkan clamp layer (scripts/vulkan/) for the LARG boxes: Isaac Sim 5.1's RTX renderer segfaults at
# startup on their driver 595.71. The home is one NFS share, so one install covers every box; the manifest goes in
# the loader's per-user implicit-layer dir, which needs no launcher change.
#
# Usage: scripts/larg/install_vkclamp.sh <host>
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

host="${1:?usage: install_vkclamp.sh <host>}"
remote_home="$(larg_ssh "$host" 'printf %s "$HOME"' 2>/dev/null)"
[ -n "$remote_home" ] || { echo "[vkclamp] cannot reach $host"; exit 1; }
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
bash "$HERE/../vulkan/install_clamp_layer.sh" "$build" "$remote_home/.local/share/vkclamp" >/dev/null
larg_ssh "$host" 'mkdir -p "$HOME/.local/share/vkclamp" "$HOME/.config/vulkan/implicit_layer.d"'
rsync -q "$build/libvkclamp.so" "$(larg_target "$host"):.local/share/vkclamp/"
rsync -q "$build/conf/vulkan/implicit_layer.d/VkLayer_vkclamp.json" "$(larg_target "$host"):.config/vulkan/implicit_layer.d/"
echo "[vkclamp] installed for every LARG box (shared home of $host); VKCLAMP_DISABLE=1 turns it off per run"
