#!/usr/bin/env bash
# Build the Vulkan clamp layer into DEST: DEST/libvkclamp.so plus DEST/conf/vulkan/implicit_layer.d/VkLayer_vkclamp.json.
# Point XDG_CONFIG_DIRS at DEST/conf (before /etc/xdg) and the Vulkan loader picks it up; Kit does not clear that
# variable. RUNTIME_DIR is where DEST appears to the process that loads it (e.g. a container bind target).
#
# Usage: install_clamp_layer.sh <dest> [<runtime dir>]
# Env:   VULKAN_HEADERS  an existing Vulkan-Headers checkout (default: clone v1.3.239 into a temp dir)
set -euo pipefail
dest="${1:?usage: install_clamp_layer.sh <dest> [<runtime dir>]}"
runtime="${2:-$dest}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
headers="${VULKAN_HEADERS:-}"
tmp=""
trap '[ -n "$tmp" ] && rm -rf "$tmp"' EXIT
if [ -z "$headers" ]; then
    tmp="$(mktemp -d)"
    git clone -q --depth 1 --branch v1.3.239 https://github.com/KhronosGroup/Vulkan-Headers.git "$tmp/Vulkan-Headers"
    headers="$tmp/Vulkan-Headers"
fi
mkdir -p "$dest/conf/vulkan/implicit_layer.d"
gcc -shared -fPIC -O2 -fvisibility=hidden -I"$headers/include" -o "$dest/libvkclamp.so" "$here/clamp_layer.c" -lpthread
cat > "$dest/conf/vulkan/implicit_layer.d/VkLayer_vkclamp.json" <<EOF
{
  "file_format_version": "1.1.0",
  "layer": {
    "name": "VK_LAYER_clamp_max_alloc",
    "type": "GLOBAL",
    "library_path": "$runtime/libvkclamp.so",
    "api_version": "1.3.239",
    "implementation_version": "1",
    "description": "Clamp maxMemoryAllocationSize for Isaac Sim's RTX renderer on newer drivers",
    "functions": {
      "vkGetInstanceProcAddr": "vkclamp_GetInstanceProcAddr",
      "vkGetDeviceProcAddr": "vkclamp_GetDeviceProcAddr"
    },
    "disable_environment": { "VKCLAMP_DISABLE": "1" }
  }
}
EOF
echo "[vkclamp] built $dest/libvkclamp.so; set XDG_CONFIG_DIRS=$runtime/conf:\${XDG_CONFIG_DIRS:-/etc/xdg}"
