#!/usr/bin/env bash
# Immutable code trees on the remote: <remote>/trees/<name>-<fingerprint>/ holds named repos at named refs (no
# --delete, the shared checkout untouched); other repos are links to the shared tree. Sourced by cluster_dev.sh.

TREES_DIR="${REMOTE_ISAACLAB_DIR}/trees"

# Check out <ref> of a local repo into a fresh dir (a detached git worktree, so LFS files are real files).
_tree_checkout() {  # _tree_checkout REPO_DIR REF DEST -> prints the commit
    local repo="$1" ref="$2" dest="$3" commit
    git -C "$repo" fetch -q origin 2>/dev/null || true
    commit="$(git -C "$repo" rev-parse --verify -q "origin/${ref}^{commit}" || git -C "$repo" rev-parse --verify -q "${ref}^{commit}")" ||
        { err "$(basename "$repo"): unknown ref '${ref}' (push it, or pass a local worktree path)"; return 1; }
    git -C "$repo" worktree add -q --detach "$dest" "$commit" >/dev/null || return 1
    git -C "$dest" lfs pull >/dev/null 2>&1 || true
    echo "$commit"
}

cmd_stage() {  # stage NAME REPO=REF|REPO=PATH ... : upload those repos as a new immutable tree
    local name="${1:-}"; shift || true
    [ -n "$name" ] && [ $# -gt 0 ] || { err "usage: stage <name> <repo>=<ref|local worktree path> ..."; exit 1; }
    case "$name" in *[!A-Za-z0-9._-]*) err "tree name '${name}': use letters, digits, . _ -"; exit 1 ;; esac
    ensure_master
    local work; work="$(mktemp -d "${TMPDIR:-/tmp}/tree-stage.XXXXXX")"
    local manifest="" spec repo ref src dirty commit srcs=() checkouts=()
    for spec in "$@"; do
        repo="${spec%%=*}"; ref="${spec#*=}"
        [ "$repo" != "$spec" ] && [ -n "$ref" ] || { err "bad spec '${spec}' (want repo=ref)"; exit 1; }
        if [ -d "$ref" ]; then
            src="$(cd "$ref" && pwd)"
            commit="$(git -C "$src" rev-parse HEAD)"
            dirty="$(git -C "$src" diff HEAD | sha256sum | cut -c1-12)"
            [ -z "$(git -C "$src" status --porcelain --untracked-files=no)" ] && dirty=""
        else
            src="${LOCAL_ISAACLAB_DIR}/resources/${repo}"
            [ -d "$src/.git" ] || [ -f "$src/.git" ] || { err "no local repo at ${src}"; exit 1; }
            commit="$(_tree_checkout "$src" "$ref" "$work/$repo")" || exit 1
            checkouts+=("$src"); src="$work/$repo"; dirty=""
        fi
        srcs+=("$repo=$src")
        manifest+="${repo} ${ref} ${commit} ${dirty}"$'\n'
    done
    # the tree carries its own node_exec.sh, so a stale copy in the shared checkout cannot run it
    manifest+="node_exec $(sha256sum "${SCRIPT_DIR}/node_exec.sh" | cut -c1-12)"$'\n'
    local fp tree; fp="$(printf '%s' "$manifest" | sort | sha256sum | cut -c1-10)"
    tree="${TREES_DIR}/${name}-${fp}"
    if on_login "[ -f '${tree}/.complete' ]"; then
        log "Tree already staged: ${tree}"
    else
        local part="${tree}.partial.$$" entry
        on_login "mkdir -p '${part}/resources' '${part}/scripts/cluster/cluster_dev'"
        for entry in "${srcs[@]}"; do
            repo="${entry%%=*}"; src="${entry#*=}"
            log "Uploading ${repo} -> ${part}/resources/${repo}"
            # unchanged files (e.g. LFS policies) are hardlinked from the shared checkout or the newest earlier
            # trees holding this repo instead of re-sent (no -t: a fresh checkout's mtimes never match, content does)
            local links=(--link-dest="${REMOTE_ISAACLAB_DIR}/resources/${repo}/") prev
            while IFS= read -r prev; do
                [ -n "$prev" ] && links+=(--link-dest="${prev}/")
            done < <(on_login "for d in \$(ls -1dt '${TREES_DIR}'/*/resources/'${repo}' 2>/dev/null); do \
                [ -L \"\$d\" ] || echo \"\$d\"; done | head -3")
            rsync -rlp --checksum --info=progress2 --exclude='.git' --exclude='logs/' --exclude='outputs/' \
                --exclude='**/worktrees/' --exclude='__pycache__/' --exclude='.claude/' "${links[@]}" \
                -e "ssh ${SSH_OPTS[*]}" "${src}/" "${CLUSTER_LOGIN}:${part}/resources/${repo}/"
        done
        rsync -t -e "ssh ${SSH_OPTS[*]}" "${SCRIPT_DIR}/node_exec.sh" \
            "${CLUSTER_LOGIN}:${part}/scripts/cluster/cluster_dev/node_exec.sh"
        [ -f "${SCRIPT_DIR}/../../.env.wandb" ] && rsync -t -e "ssh ${SSH_OPTS[*]}" "${SCRIPT_DIR}/../../.env.wandb" \
            "${CLUSTER_LOGIN}:${part}/scripts/.env.wandb"
        printf '%s' "$manifest" | on_login "cat > '${part}/MANIFEST'"
        # every other repo (assets, unstaged packages) is the shared one, so RESOURCES_DIR stays complete
        on_login "for d in '${REMOTE_ISAACLAB_DIR}'/resources/*/; do n=\$(basename \"\$d\"); \
            [ -e '${part}/resources/'\"\$n\" ] || ln -s \"\${d%/}\" '${part}/resources/'\"\$n\"; done; \
            touch '${part}/.complete'; mv -T '${part}' '${tree}' 2>/dev/null || rm -rf '${part}'"
        log "Staged ${tree}"
    fi
    rm -rf "$work"
    for src in "${checkouts[@]}"; do git -C "$src" worktree prune; done  # drop the temporary checkouts
    printf '%s' "$manifest" | sed 's/^/  /'
    echo "Run with: just cluster ${CLUSTER} develop exec --tree ${name} -- <cmd>"
}

# Newest complete tree named NAME (or NAME-<fingerprint> exactly).
resolve_tree() {  # resolve_tree NAME -> remote path
    local found
    found="$(on_login "if [ -f '${TREES_DIR}/$1/.complete' ]; then echo '${TREES_DIR}/$1'; else \
        ls -1td '${TREES_DIR}/$1'-*/.complete 2>/dev/null | head -1 | xargs -r dirname; fi")"
    [ -n "$found" ] || { err "no staged tree '$1' under ${TREES_DIR} (see: develop trees)"; return 1; }
    echo "$found"
}

cmd_trees() {  # trees : list staged trees with their manifests
    ensure_master
    on_login "for t in \$(ls -1td '${TREES_DIR}'/*/ 2>/dev/null); do [ -f \"\$t/.complete\" ] || continue; \
        echo \"\$(basename \"\$t\")  (\$(date -r \"\$t/.complete\" +%F' '%H:%M))\"; sed 's/^/  /' \"\$t/MANIFEST\"; done"
}
