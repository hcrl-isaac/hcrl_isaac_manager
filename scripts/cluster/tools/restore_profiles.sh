#!/usr/bin/env bash
# Guards the gitignored profiles in scripts/cluster/config/ against branch switches: refuses on a branch that tracks
# them, restores a deleted submit script from .backup/ (else git history), snapshots changed profiles to .backup/.
_profiles_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

_profiles_snapshot=1
if _tracked="$(git -C "$_profiles_repo" ls-files -- scripts/cluster/config 2>/dev/null)"; then
    if [ -n "$_tracked" ]; then
        echo "[ERROR] This branch still tracks scripts/cluster/config/ (it predates per-user profiles), so git may" \
            "have replaced your profiles. Merge main into it first; your copies are in config/<name>/.backup/." >&2
        exit 1
    fi
elif [ -e "$_profiles_repo/.git" ]; then
    # cannot tell whether this branch tracks the profiles: never let a snapshot overwrite a good backup
    echo "[WARN] git ls-files failed; not snapshotting profiles into .backup/ this time." >&2
    _profiles_snapshot=0
fi

restore_profiles() {
    local cfg rel sha tmp
    for cfg in "$_profiles_repo"/scripts/cluster/config/*/; do
        [ -f "${cfg}.env.cluster" ] || continue
        rel="scripts/cluster/config/$(basename "$cfg")/submit_job_slurm.sh"
        if [ ! -f "$_profiles_repo/$rel" ]; then
            if [ -s "${cfg}.backup/submit_job_slurm.sh" ]; then
                cp -p "${cfg}.backup/submit_job_slurm.sh" "$_profiles_repo/$rel" && echo "[INFO] Restored $rel from its .backup/."
            else
                sha="$(git -C "$_profiles_repo" log --diff-filter=D -1 --format=%H -- "$rel" 2>/dev/null)" || continue
                [ -n "$sha" ] || continue
                tmp="$(mktemp)"
                if git -C "$_profiles_repo" show "${sha}^:${rel}" > "$tmp" 2>/dev/null && [ -s "$tmp" ]; then
                    chmod 755 "$tmp" && mv "$tmp" "$_profiles_repo/$rel"
                    echo "[WARN] Restored $rel from git history (${sha:0:7}^): no .backup/, so local edits are lost."
                else
                    rm -f "$tmp"
                fi
            fi
        fi
        [ -f "$_profiles_repo/$rel" ] && [ "$_profiles_snapshot" = 1 ] || continue
        if ! cmp -s "${cfg}.env.cluster" "${cfg}.backup/.env.cluster" 2>/dev/null \
            || ! cmp -s "$_profiles_repo/$rel" "${cfg}.backup/submit_job_slurm.sh" 2>/dev/null; then
            mkdir -p "${cfg}.backup" && cp -p "${cfg}.env.cluster" "$_profiles_repo/$rel" "${cfg}.backup/"
        fi
    done
}
restore_profiles
