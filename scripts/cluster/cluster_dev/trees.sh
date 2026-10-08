#!/usr/bin/env bash
# Staged code trees on the remote: <remote>/trees/<name>-<fingerprint>/ holds named repos at named refs (no
# --delete, the shared checkout untouched), and every other repo is a link to the shared one. Sourced by cluster_dev.sh.

TREES_DIR="${CLUSTER_TREES_DIR:-${REMOTE_ISAACLAB_DIR}/trees}"  # a profile points it off a quota'd home
TREE_RW_DIRS=(logs outputs wandb)  # mount points that redirect run output out of a staged repo

_tree_valid() { [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] && [ "$1" != "." ] && [ "$1" != ".." ]; }

# The files a tree carries: tracked and untracked, not ignored, present on disk (NUL-separated, relative).
_tree_files() {
    local f
    git -C "$1" ls-files -z --cached --others --exclude-standard | while IFS= read -r -d '' f; do
        [ -e "$1/$f" ] && printf './%s\0' "$f"
    done
}

# Hash of exactly the carried files: modes, symlink targets and regular-file content.
_tree_content_hash() {  # _tree_content_hash DIR -> 12 hex chars
    (
        set -o pipefail
        cd "$1" || exit 1
        list="$(mktemp)" || exit 1
        trap 'rm -f "$list"' EXIT
        _tree_files . | sort -z > "$list" || exit 1
        {
            xargs -0 -r stat -c '%A %n' < "$list" || exit 1
            xargs -0 -r sh -c 'for f; do [ -L "$f" ] && printf "%s -> %s\n" "$f" "$(readlink "$f")"; done; true' sh < "$list"
            xargs -0 -r sh -c 'for f; do [ -L "$f" ] || [ -d "$f" ] || printf "%s\0" "$f"; done' sh < "$list" |
                xargs -0 -r sha256sum || exit 1
        } | sha256sum | cut -c1-12
    )
}

# Check out <ref> of a local repo into DEST (a detached git worktree, so LFS files are real files).
_tree_checkout() {  # _tree_checkout REPO_DIR REF DEST -> prints "<commit> <origin|local-ref>"
    local repo="$1" ref="$2" dest="$3" commit kind=origin
    git -C "$repo" fetch -q origin 2>/dev/null || true
    commit="$(git -C "$repo" rev-parse --verify -q "origin/${ref}^{commit}")" || {
        commit="$(git -C "$repo" rev-parse --verify -q "${ref}^{commit}")" || {
            err "$(basename "$repo"): unknown ref '${ref}' (push it, or pass a local worktree path)"; return 1; }
        kind=local-ref
    }
    git -C "$repo" worktree add -q --detach "$dest" "$commit" >/dev/null || return 1  # the caller registered DEST
    if git -C "$dest" lfs ls-files >/dev/null 2>&1 && [ -n "$(git -C "$dest" lfs ls-files)" ]; then
        git -C "$dest" lfs pull >/dev/null 2>&1
        if git -C "$dest" lfs ls-files | grep -q '^[0-9a-f]* - '; then
            err "$(basename "$repo")@${ref}: LFS objects missing (would upload pointer files); run git lfs fetch"
            return 1
        fi
    fi
    echo "$commit $kind"
}

cmd_stage() {  # stage [--no-space-check] NAME REPO=REF|REPO=PATH ... : upload those repos as a new tree
    local space_check=1
    [ "${1:-}" = --no-space-check ] && { space_check=""; shift; }
    local name="${1:-}"; shift || true
    [ -n "$name" ] && [ $# -gt 0 ] || { err "usage: stage <name> <repo>=<ref|/path/to/worktree> ..."; exit 1; }
    _tree_valid "$name" || { err "tree name '${name}': use letters, digits, . _ -"; exit 1; }
    local spec repo ref seen=" "
    for spec in "$@"; do  # validate everything before touching anything
        repo="${spec%%=*}"; ref="${spec#*=}"
        [ "$repo" != "$spec" ] && [ -n "$ref" ] || { err "bad spec '${spec}' (want repo=ref)"; exit 1; }
        _tree_valid "$repo" || { err "repo '${repo}': use letters, digits, . _ -"; exit 1; }
        case "$repo" in  # IsaacLab is a separate overlay
            IsaacLab) err "${repo} cannot be staged: it stays linked to the shared checkout"; exit 1 ;;
        esac
        [ -d "${LOCAL_ISAACLAB_DIR}/resources/${repo}" ] || { err "no repo resources/${repo} in ${LOCAL_ISAACLAB_DIR}"; exit 1; }
        case "$seen" in *" ${repo} "*) err "repo '${repo}' named twice"; exit 1 ;; esac
        seen+="${repo} "
    done
    ensure_master
    [ -n "$space_check" ] && check_space "$TREES_DIR" "staged trees"
    STAGE_WORK="$(mktemp -d "${TMPDIR:-/tmp}/tree-stage.XXXXXX")"
    STAGE_CHECKOUTS=(); STAGE_PART=""
    trap _stage_cleanup EXIT
    local manifest="" src commit kind content out srcs=()
    for spec in "$@"; do
        repo="${spec%%=*}"; ref="${spec#*=}"
        case "$ref" in
            /* | ./* | ../*)
                src="$(cd "$ref" 2>/dev/null && pwd -P)" || { err "no directory ${ref}"; exit 1; }
                [ "$(git -C "$src" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" = \
                  "$(git -C "${LOCAL_ISAACLAB_DIR}/resources/${repo}" rev-parse --path-format=absolute --git-common-dir)" ] &&
                [ "$(cd "$(git -C "$src" rev-parse --show-toplevel)" && pwd -P)" = "$src" ] ||
                    { err "${ref} is not the top of a worktree of ${repo}"; exit 1; }
                ref="$src"
                commit="$(git -C "$src" rev-parse HEAD)"
                kind=worktree; [ -n "$(git -C "$src" status --porcelain)" ] && kind=worktree-dirty
                ;;
            *)
                STAGE_CHECKOUTS+=("${LOCAL_ISAACLAB_DIR}/resources/${repo}=${STAGE_WORK}/${repo}")
                out="$(_tree_checkout "${LOCAL_ISAACLAB_DIR}/resources/${repo}" "$ref" "${STAGE_WORK}/${repo}")" || exit 1
                commit="${out% *}"; kind="${out#* }"
                src="${STAGE_WORK}/${repo}"
                ;;
        esac
        content="$(_tree_content_hash "$src")" || { err "could not hash the files of ${repo} (${src})"; exit 1; }
        srcs+=("$repo=$src")
        manifest+="${repo} ${ref} ${commit} ${kind} ${content}"$'\n'
    done
    manifest+="node_exec $(sha256sum "${SCRIPT_DIR}/node_exec.sh" | cut -c1-12)"$'\n'
    local fp tree; fp="$(printf '%s' "$manifest" | sort | sha256sum | cut -c1-10)"
    tree="${TREES_DIR}/${name}-${fp}"
    if on_login "[ -f '${tree}/.complete' ]"; then
        log "Tree already staged: ${tree}"
    else
        STAGE_PART="${tree}.partial.$$"
        on_login "mkdir -p '${STAGE_PART}/resources' '${STAGE_PART}/scripts/cluster/cluster_dev'" || exit 1
        local entry links mode prev d
        for entry in "${srcs[@]}"; do
            repo="${entry%%=*}"; src="${entry#*=}"
            log "Uploading ${repo} -> ${STAGE_PART}/resources/${repo}"
            # identical files are hardlinked from the newest earlier trees with this repo, never the mutable shared copy
            links=() mode=(--chmod=Fa-w)
            case "$repo" in
                # an asset repo is written to by the run (URDF -> USD conversion rewrites files beside the URDF), so
                # its files stay writable and are its own: a file hardlinked between trees would change in all of them
                *_robots) mode=() ;;
                *) while IFS= read -r prev; do
                       [ -n "$prev" ] && links+=(--link-dest="${prev}/")
                   done < <(on_login "for d in \$(ls -1dt '${TREES_DIR}'/*/resources/'${repo}' 2>/dev/null); do \
                       case \"\$d\" in *.partial.*) continue ;; esac; [ -L \"\$d\" ] || echo \"\$d\"; done | head -3") ;;
            esac
            _tree_files "$src" | rsync -rlp "${mode[@]}" --checksum --info=progress2 --from0 --files-from=- "${links[@]}" \
                -e "ssh ${SSH_OPTS[*]}" "${src}/" "${CLUSTER_LOGIN}:${STAGE_PART}/resources/${repo}/" || exit 1
            for d in "${TREE_RW_DIRS[@]}"; do on_login "mkdir -p '${STAGE_PART}/resources/${repo}/${d}'" || exit 1; done
            [ "$repo" = hcrl_isaaclab ] && { on_login "mkdir -p '${STAGE_PART}/resources/${repo}/.artifacts'" || exit 1; }
            # W&B artifacts are gitignored links into the shared artifact root: carry the shared checkout's links
            on_login "cd '${REMOTE_ISAACLAB_DIR}/resources/${repo}' 2>/dev/null || exit 0; \
                find . -path ./worktrees -prune -o -type l -lname '*/.artifacts/*' -print | while IFS= read -r l; do \
                t='${STAGE_PART}/resources/${repo}/'\"\$l\"; [ -e \"\$t\" ] || [ -L \"\$t\" ] && continue; \
                mkdir -p \"\$(dirname \"\$t\")\" && cp -P \"\$l\" \"\$t\"; done" || exit 1
        done
        rsync -t -e "ssh ${SSH_OPTS[*]}" "${SCRIPT_DIR}/node_exec.sh" \
            "${CLUSTER_LOGIN}:${STAGE_PART}/scripts/cluster/cluster_dev/node_exec.sh" || exit 1
        local f
        # the API key: owner-only here, which -p carries to the tree
        [ ! -f "${SCRIPT_DIR}/../../.env.wandb" ] || chmod go-rwx "${SCRIPT_DIR}/../../.env.wandb"
        for f in .env.wandb .env.base; do
            [ -f "${SCRIPT_DIR}/../../${f}" ] && { rsync -tp -e "ssh ${SSH_OPTS[*]}" "${SCRIPT_DIR}/../../${f}" \
                "${CLUSTER_LOGIN}:${STAGE_PART}/scripts/${f}" || exit 1; }
        done
        # unstaged repos are the shared ones, which a later sync can change under a running tree job
        printf '%s' "$manifest" | on_login "{ echo '# writable tree: a file hardlinked between trees changes in all of them if written in place'; cat; } > '${STAGE_PART}/MANIFEST'; \
            for d in '${REMOTE_ISAACLAB_DIR}'/resources/*/; do n=\$(basename \"\$d\"); \
            [ -e '${STAGE_PART}/resources/'\"\$n\" ] && continue; ln -s \"\${d%/}\" '${STAGE_PART}/resources/'\"\$n\"; \
            echo \"\$n shared-live \${d%/}\" >> '${STAGE_PART}/MANIFEST'; done" || exit 1
        # marked complete before the rename, so a tree under its final name is always complete
        on_login "touch '${STAGE_PART}/.complete'" || exit 1
        if on_login "mv -T '${STAGE_PART}' '${tree}'"; then
            STAGE_PART=""
        elif on_login "[ -f '${tree}/.complete' ]"; then
            log "Another stage of the same refs finished first; using it."
        else
            err "could not move ${STAGE_PART} to ${tree}"; exit 1
        fi
        log "Staged ${tree}"
    fi
    printf '%s' "$manifest" | sed 's/^/  /'
    echo "Run with: just cluster ${CLUSTER} develop exec --tree $(basename "$tree") -- <cmd>"
}

_stage_cleanup() {  # trap: drop this run's temporary checkouts and any partial upload
    local entry
    for entry in "${STAGE_CHECKOUTS[@]}"; do
        git -C "${entry%%=*}" worktree remove --force "${entry#*=}" 2>/dev/null
    done
    [ -n "${STAGE_WORK:-}" ] && rm -rf "$STAGE_WORK"
    [ -n "${STAGE_PART:-}" ] && on_login "rm -rf '${STAGE_PART}'" 2>/dev/null
    return 0
}

# The complete tree NAME-<fingerprint>, or the only complete tree named NAME (refuses if there are several).
resolve_tree() {  # resolve_tree ID -> remote path
    local id="$1" found n
    _tree_valid "$id" || { err "tree '${id}': use letters, digits, . _ -"; return 1; }
    if [[ "$id" =~ -[0-9a-f]{10}$ ]] && on_login "[ -f '${TREES_DIR}/${id}/.complete' ]"; then
        echo "${TREES_DIR}/${id}"; return 0
    fi
    found="$(on_login "for t in '${TREES_DIR}/${id}'-*; do [ -f \"\$t/.complete\" ] && basename \"\$t\"; done 2>/dev/null" |
        grep -E "^${id//./\\.}-[0-9a-f]{10}$" || true)"
    n="$(printf '%s' "$found" | grep -c . || true)"
    if [ "$n" -eq 0 ]; then
        err "no staged tree '${id}' under ${TREES_DIR} (see: develop trees)"; return 1
    elif [ "$n" -gt 1 ]; then
        err "several trees are named '${id}'; pass the full id:"$'\n'"$(printf '%s' "$found" | sed 's/^/  /')"; return 1
    fi
    echo "${TREES_DIR}/${found}"
}

cmd_trees() {  # trees [rm <name>-<fp> [--force] | rm --partials] : list, or remove trees
    ensure_master
    if [ "${1:-}" = rm ]; then
        shift; _trees_rm "$@"; return
    fi
    on_login "for t in \$(ls -1td '${TREES_DIR}'/*/ 2>/dev/null); do case \"\$t\" in *.partial.*) continue ;; esac; \
        [ -f \"\$t/.complete\" ] || continue; echo \"\$(basename \"\$t\")  (\$(date -r \"\$t/.complete\" +%F' '%H:%M))\"; sed '/^#/d; s/^/  /' \"\$t/MANIFEST\"; done"
}

_trees_rm() {
    if [ "${1:-}" = --partials ]; then
        # a partial is abandoned once nothing in it has changed for an hour (an upload keeps writing into it)
        on_login "for p in '${TREES_DIR}'/*.partial.*; do [ -d \"\$p\" ] || continue; \
            [ -n \"\$(find \"\$p\" -mmin -60 -print -quit)\" ] && continue; echo \"\$p\"; rm -rf \"\$p\"; done"
        return
    fi
    local id="${1:-}" force="${2:-}" marker live=""
    _tree_valid "$id" && [[ "$id" =~ -[0-9a-f]{10}$ ]] || { err "usage: trees rm <name>-<fingerprint> [--force] | --partials"; exit 1; }
    on_login "[ -d '${TREES_DIR}/${id}' ]" || { err "no tree ${id}"; exit 1; }
    # node_exec.sh holds .in-use/<job>.<step|nostep> while each run lasts, and one squeue no longer lists has ended
    if [ "$force" != --force ]; then
        local job sq rc
        while IFS= read -r marker; do
            [ -n "$marker" ] || continue
            if [[ "$marker" =~ ^([0-9]+)\.([0-9]+|nostep)$ ]]; then
                job="${BASH_REMATCH[1]}"
                sq="$(on_login "squeue -h -s -j '${job}' -o %i 2>&1")" && rc=0 || rc=$?
                if [ "$rc" -ne 0 ]; then
                    grep -q "Invalid job id" <<< "$sq" && continue
                    err "cannot check job ${job}: $(head -1 <<< "$sq"); pass --force if its runs are gone"; exit 1
                fi
                if [ "${marker#*.}" = nostep ]; then
                    grep -q "^${job}\." <<< "$sq" && live+="${marker} "
                else
                    grep -qx "$marker" <<< "$sq" && live+="${marker} "
                fi
            else
                live+="${marker} "
            fi
        done < <(on_login "ls -1 '${TREES_DIR}/${id}/.in-use' 2>/dev/null")
        [ -z "$live" ] || { err "tree ${id} is in use by ${live}(pass --force if those runs are gone)"; exit 1; }
    fi
    on_login "rm -rf '${TREES_DIR}/${id}'" && log "Removed ${TREES_DIR}/${id}"
}
