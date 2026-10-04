#!/usr/bin/env bash
# Restore every profile's submit_job_slurm.sh that the pull untracking scripts/cluster/config/ deleted beside a
# kept .env.cluster, from the commit that removed it. Sourced by cluster_interface.sh and cluster_dev.sh.
restore_profiles() {
    local repo cfg rel sha tmp
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    for cfg in "$repo"/scripts/cluster/config/*/; do
        rel="scripts/cluster/config/$(basename "$cfg")/submit_job_slurm.sh"
        [ -f "${cfg}.env.cluster" ] && [ ! -f "$repo/$rel" ] || continue
        sha="$(git -C "$repo" log --diff-filter=D -1 --format=%H -- "$rel" 2>/dev/null)" || continue
        [ -n "$sha" ] || continue
        tmp="$(mktemp)"
        if git -C "$repo" show "${sha}^:${rel}" > "$tmp" 2>/dev/null && [ -s "$tmp" ]; then
            chmod 755 "$tmp" && mv "$tmp" "$repo/$rel" && echo "[INFO] Restored $rel from ${sha:0:7}^."
        else
            rm -f "$tmp"
        fi
    done
}
restore_profiles
