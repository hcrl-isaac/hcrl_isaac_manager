#!/usr/bin/env bash
# Staged code trees on the remote, the only way code reaches a cluster: <remote>/trees/<name>-<fingerprint>/ holds
# repos at refs or as on disk. `stage` with no specs stages the whole local workspace as the tree `default`, which
# exec and batch jobs run unless given --tree; a partial tree takes its other repos from the newest `default` (the
# shared checkout's, where none is staged). Sourced by cluster_dev.sh.

TREES_DIR="${CLUSTER_TREES_DIR:-${REMOTE_ISAACLAB_DIR}/trees}"  # a profile points it off a quota'd home
TREE_RW_DIRS=(logs outputs wandb)  # mount points that redirect run output out of a staged repo
TREE_DATA_REPOS=(motion_datasets)  # cluster-side data, shared by every tree: a whole stage adds new files, never trees
TREE_KEEP=5  # complete trees kept per name (besides any a run still uses); older ones go after the next stage
TREE_MDT_MAX=98  # refuse to stage once the metadata target holding the trees is this % full (Lustre inodes)

# The whole workspace as on disk: every git repo under resources/, less the data repos and the repos a profile
# sharing this destination keeps off it (its .rsync-exclude, one repo name per line).
_default_specs() {
    local d n cfg dest skip=" ${TREE_DATA_REPOS[*]} "
    for cfg in "${SCRIPT_DIR}"/../config/*/; do
        [ -f "${cfg}.rsync-exclude" ] && [ -f "${cfg}.env.cluster" ] || continue
        dest="$(bash -c 'source "$1" >/dev/null 2>&1; printf %s "${CLUSTER_ISAACLAB_DIR:-}"' _ "${cfg}.env.cluster")"
        [ "$dest" = "$REMOTE_ISAACLAB_DIR" ] || [ "$(basename "$cfg")" = "${CLUSTER:-}" ] || continue
        skip+="$(sed 's/#.*//; s|/*$||; s|.*/||' "${cfg}.rsync-exclude" | tr '\n' ' ') "
    done
    for d in "${LOCAL_ISAACLAB_DIR}"/resources/*/; do
        n="$(basename "$d")"
        [ -e "${d}.git" ] || continue  # not a repo (wandb/, a retired package's leftovers)
        case "$skip" in *" ${n} "*) continue ;; esac
        printf '%s=%s\n' "$n" "${d%/}"
    done
}

# Refuse when the Lustre metadata target holding DIR is nearly out of inodes: every new file there fails then.
_check_inodes() {  # _check_inodes DIR
    local pct
    pct="$(on_login "command -v lfs >/dev/null 2>&1 || exit 0; d=$(printf %q "$1"); mkdir -p \"\$d\"; \
        i=\$(lfs getstripe -m \"\$d\" 2>/dev/null) || exit 0; m=\$(printf 'MDT%04x' \"\$i\"); \
        lfs df -i \"\$d\" 2>/dev/null | awk -v m=\"\$m\" 'index(\$1, m) { sub(/%/, \"\", \$5); print \$5; exit }'" 2>/dev/null)"
    if [ -n "$pct" ] && [ "$pct" -gt "$TREE_MDT_MAX" ] 2>/dev/null; then
        err "the metadata target holding $1 is ${pct}% out of inodes (over ${TREE_MDT_MAX}%): new files there fail; pass --no-space-check to try anyway"
        exit 1
    fi
}

# The data repos' training files (the allowlist sync used), added to the shared copy: never deleted, never a tree.
_push_data() {
    local repo
    for repo in "${TREE_DATA_REPOS[@]}"; do
        [ -d "${LOCAL_ISAACLAB_DIR}/resources/${repo}" ] || continue
        log "Adding new ${repo} files -> ${REMOTE_ISAACLAB_DIR}/resources/${repo} (shared, never deleted)"
        rsync -rlpt --info=progress2 --include='*/' --include='*.pt' --include='*.arena.json' --include='*.courts.json' \
            --include='*.manifest.json' --exclude='*' --prune-empty-dirs -e "ssh ${SSH_OPTS[*]}" \
            --rsync-path="mkdir -p $(printf %q "${REMOTE_ISAACLAB_DIR}/resources/${repo}") && rsync" \
            "${LOCAL_ISAACLAB_DIR}/resources/${repo}/" "${CLUSTER_LOGIN}:${REMOTE_ISAACLAB_DIR}/resources/${repo}/" || return 1
    done
}

# Keep the newest TREE_KEEP complete trees of NAME; an older one goes unless a run still uses it.
_prune_trees() {  # _prune_trees NAME
    local id
    while IFS= read -r id; do
        [ -n "$id" ] || continue
        if ( _trees_rm "$id" ) > /dev/null 2>&1; then log "Pruned ${id}"; else log "Kept ${id} (in use)"; fi
    done < <(on_login "ls -1td '${TREES_DIR}/${1}'-*/ 2>/dev/null" | sed 's|/$||; s|.*/||' |
        grep -E "^${1//./\\.}-[0-9a-f]{10}$" | tail -n +$((TREE_KEEP + 1)))
}

_tree_valid() { [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] && [ "$1" != "." ] && [ "$1" != ".." ]; }

# The files a tree carries: tracked and untracked, not ignored, present on disk (NUL-separated, relative). A checkout's
# session state (worktrees/, .claude/) and run output (the TREE_RW_DIRS, mounted over in a tree) never ship.
_tree_files() {
    local f
    git -C "$1" ls-files -z --cached --others --exclude-standard | while IFS= read -r -d '' f; do
        case "$f" in worktrees/* | .claude/* | logs/* | outputs/* | wandb/*) continue ;; esac
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

cmd_stage() {  # stage [--no-space-check] [NAME] [REPO=REF|REPO=PATH ...] : upload repos as a new tree
    local space_check=1 name=default whole=""
    [ "${1:-}" = --no-space-check ] && { space_check=""; shift; }
    if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then name="$1"; shift; fi
    _tree_valid "$name" || { err "tree name '${name}': use letters, digits, . _ -"; exit 1; }
    if [ $# -eq 0 ]; then  # the whole workspace as on disk, uncommitted work included
        whole=1
        local specs; mapfile -t specs < <(_default_specs)
        [ "${#specs[@]}" -gt 0 ] || { err "no repos under ${LOCAL_ISAACLAB_DIR}/resources to stage"; exit 1; }
        set -- "${specs[@]}"
    fi
    local spec repo ref seen=" "
    for spec in "$@"; do  # validate everything before touching anything
        repo="${spec%%=*}"; ref="${spec#*=}"
        [ "$repo" != "$spec" ] && [ -n "$ref" ] || { err "bad spec '${spec}' (want repo=ref)"; exit 1; }
        _tree_valid "$repo" || { err "repo '${repo}': use letters, digits, . _ -"; exit 1; }
        case " ${TREE_DATA_REPOS[*]} " in *" ${repo} "*) err "${repo} is shared data, not staged in a tree"; exit 1 ;; esac
        [ -d "${LOCAL_ISAACLAB_DIR}/resources/${repo}" ] || { err "no repo resources/${repo} in ${LOCAL_ISAACLAB_DIR}"; exit 1; }
        case "$seen" in *" ${repo} "*) err "repo '${repo}' named twice"; exit 1 ;; esac
        seen+="${repo} "
    done
    ensure_master
    [ -n "$space_check" ] && { check_space "$TREES_DIR" "staged trees"; _check_inodes "$TREES_DIR"; }
    # a partial tree's other repos come from the newest complete `default`, so it never runs the shared checkout
    local base=""
    [ -z "$whole" ] && base="$(resolve_tree default 2>/dev/null || true)"
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
    [ -n "$base" ] && manifest+="base $(basename "$base")"$'\n'
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
        # the base's other repos, hardlinked (an asset repo copied: runs write to it), then links to the shared data
        printf '%s' "$manifest" | on_login "{ echo '# writable tree: a file hardlinked between trees changes in all of them if written in place'; cat; } > '${STAGE_PART}/MANIFEST'; \
            for d in ${base:+'${base}'/resources/*/}; do n=\$(basename \"\$d\"); \
            [ -e '${STAGE_PART}/resources/'\"\$n\" ] && continue; \
            if [ -L \"\${d%/}\" ]; then cp -P \"\${d%/}\" '${STAGE_PART}/resources/'; \
            else case \"\$n\" in *_robots) cp -a \"\${d%/}\" '${STAGE_PART}/resources/' ;; \
                *) cp -al \"\${d%/}\" '${STAGE_PART}/resources/' ;; esac; fi || exit 1; \
            echo \"\$n base ${base##*/}\" >> '${STAGE_PART}/MANIFEST'; done; \
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
    if [ -n "$whole" ]; then _push_data || { err "could not add the data repos' new files"; exit 1; }; fi
    _prune_trees "$name"
    printf '%s' "$manifest" | sed 's/^/  /'
    local hint="${TREE_RUN_HINT:-pls cluster ${CLUSTER:-} develop exec --tree <id> -- <cmd>}"
    echo "Run with: ${hint//<id>/$(basename "$tree")}"
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

# The complete tree NAME-<fingerprint>, or the newest complete tree named NAME (each stage of a name is newer).
resolve_tree() {  # resolve_tree ID -> remote path
    local id="$1" found
    _tree_valid "$id" || { err "tree '${id}': use letters, digits, . _ -"; return 1; }
    if [[ "$id" =~ -[0-9a-f]{10}$ ]] && on_login "[ -f '${TREES_DIR}/${id}/.complete' ]"; then
        echo "${TREES_DIR}/${id}"; return 0
    fi
    found="$(on_login "for t in '${TREES_DIR}/${id}'-*; do [ -f \"\$t/.complete\" ] && \
        find \"\$t/.complete\" -printf \"%T@ \$(basename \"\$t\")\\n\"; done 2>/dev/null" |
        grep -E " ${id//./\\.}-[0-9a-f]{10}$" | sort -rn | head -1 | cut -d' ' -f2 || true)"
    [ -n "$found" ] || { err "no staged tree '${id}' under ${TREES_DIR} (stage it: develop stage)"; return 1; }
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
            elif [[ "$marker" =~ ^larg\.([0-9]+)$ ]]; then
                # a LARG run (scripts/larg/train.sh --tree): live while its process is
                on_login "kill -0 ${BASH_REMATCH[1]} 2>/dev/null" && live+="${marker} "
            else
                live+="${marker} "
            fi
        done < <(on_login "ls -1 '${TREES_DIR}/${id}/.in-use' 2>/dev/null")
        [ -z "$live" ] || { err "tree ${id} is in use by ${live}(pass --force if those runs are gone)"; exit 1; }
    fi
    on_login "rm -rf '${TREES_DIR}/${id}'" && log "Removed ${TREES_DIR}/${id}"
}
