#!/usr/bin/env bash
#
# cluster_dev.sh -- turn an HPC compute node into a persistent (<=walltime) dev box.
#
# WHY this shape: interactive partitions are usually short-capped, but *batch* jobs get a
# long walltime, and many HPC sites allow "direct ssh to a compute node in a running job".
# So we submit a long-lived "sentinel" batch job that just holds a node, then ssh into it
# (proxied through a persistent login-node ControlMaster) and run/develop there. Where SSH
# keys are disabled (password+2FA every login, e.g. NCSA Delta), the ControlMaster socket --
# opened ONCE with one 2FA approval and kept warm -- is the only way to avoid re-auth all day.
#
# Cluster-agnostic: all site specifics (login host, account, partition, resources) come from
# config/<cluster>/.env.cluster, selected with CLUSTER=<name>. Nothing here is Delta-specific.
#
# Open master only:         ./cluster_dev.sh open             (approve ONE 2FA prompt; no sync)
# Queue a job:              ./cluster_dev.sh start [--no-sync]   (approve ONE 2FA prompt)
# Then it self-tracks the (possibly multi-hour) queue wait in the background.
# Check anytime:            ./cluster_dev.sh status
# Mirror code:              ./cluster_dev.sh sync [--dry-run] (--dry-run lists what would change or be deleted)
# Stage a code tree:        ./cluster_dev.sh stage <name> <repo>=<ref|/path/to/worktree> ...  (no --delete)
#                           ./cluster_dev.sh exec --tree <name>[-<fp>] -- <cmd>   (run against that tree)
#                           ./cluster_dev.sh trees [rm <name>-<fp> | rm --partials]   (list / remove trees)
# Use it:                   ./cluster_dev.sh attach           (interactive shell on the node)
#                           ./cluster_dev.sh exec -- <cmd>    (run in container, SSH-tethered)
#                           ./cluster_dev.sh exec --detach -- <cmd>   (run in container, detached
#                                                                      from SSH master; survives
#                                                                      master drops; log on login
#                                                                      node, follow with `tail`)
#                           ./cluster_dev.sh tail             (follow the latest --detach log)
# Stop launched runs:       ./cluster_dev.sh kill             (list running steps)
#                           ./cluster_dev.sh kill --all | <step>...   (scancel steps; the job,
#                                                                      sentinel, and SSH master
#                                                                      all keep running)
# Tear down:                ./cluster_dev.sh stop
#
# Everything except the first 2FA is non-interactive, so a Claude session can drive
# `status` / `exec` / `attach` over the live master without any credentials.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

#============================================================================
# Config -- sourced from the selected cluster's .env.cluster, with dev-box overrides.
#============================================================================
# Cluster config: CLUSTER=<name> picks ../config/<name>/.env.cluster (default "default"); matches
# cluster_interface.sh, which sets CLUSTER when invoked as `cluster_interface.sh develop`.
CLUSTER="${CLUSTER:-default}"
ENV_FILE="${SCRIPT_DIR}/../config/${CLUSTER}/.env.cluster"
# profiles are gitignored and per user, so a manager worktree has none: it borrows the main checkout's for the
# commands that leave the shared sentinel state alone (see MAIN_PROFILE_COMMANDS)
MAIN_CHECKOUT=""
if [ ! -f "$ENV_FILE" ]; then
    _common="$(git -C "$SCRIPT_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    _main_env="$(dirname "${_common:-/nonexistent}")/scripts/cluster/config/${CLUSTER}/.env.cluster"
    if [ -n "$_common" ] && [ -f "$_main_env" ]; then
        ENV_FILE="$_main_env"; MAIN_CHECKOUT="$(dirname "$_common")"
        echo "[cluster_dev] using the main checkout's profile ${ENV_FILE}" >&2
    fi
fi
MAIN_PROFILE_COMMANDS=" stage trees exec status tail open attach __resolve_tree __free_gb help -h --help "
if [ -n "$MAIN_CHECKOUT" ] && [[ "$MAIN_PROFILE_COMMANDS" != *" ${1:-help} "* ]]; then
    echo "[cluster_dev] '${1}' changes the shared sentinel state; run it from the main checkout ${MAIN_CHECKOUT}" >&2
    exit 1
fi
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

CLUSTER_LOGIN="${CLUSTER_LOGIN:?CLUSTER_LOGIN not set (expected from config/${CLUSTER}/.env.cluster)}"
DEV_USER="${CLUSTER_LOGIN%@*}"
CLUSTER_LOGIN_HOST="${CLUSTER_LOGIN_HOST:-${CLUSTER_LOGIN#*@}}"   # login host; round-robin DNS is fine since we always multiplex over one master.
REMOTE_ISAACLAB_DIR="${CLUSTER_ISAACLAB_DIR:?CLUSTER_ISAACLAB_DIR not set}"

source "${SCRIPT_DIR}/../tools/restore_profiles.sh"
# Sentinel resources reuse the #SBATCH config from this cluster's submit_job_slurm.sh (the same one the
# `job` path uses), so the dev box matches it with no separate config.
SUBMIT_SLURM="$(dirname "$ENV_FILE")/submit_job_slurm.sh"
[ -f "$SUBMIT_SLURM" ] || { echo "[cluster_dev] no submit_job_slurm.sh next to $ENV_FILE" >&2; exit 1; }
# its #SBATCH directives, minus job-name/output (the sentinel sets its own).
SBATCH_DIRECTIVES="$(grep -E '^#SBATCH' "$SUBMIT_SLURM" | grep -vE -- '--job-name|--output')"
# the few values srun --overlap (attach/exec) needs, pulled from those directives.
_sbatch_val() { printf '%s\n' "$SBATCH_DIRECTIVES" | grep -oE -- "(^|[[:space:]])$1[= ][^[:space:]]+" | head -1 | sed -E "s/.*$1[= ]//" || true; }
DEV_PARTITION="$(_sbatch_val -p)"; DEV_ACCOUNT="$(_sbatch_val -A)"
DEV_TIME="$(_sbatch_val --time)"; DEV_GPUS="$(_sbatch_val --gpus-per-node)"
DEV_CPUS="$(_sbatch_val --cpus-per-task)"

# How to reach the node for attach/exec: "auto" probes login->node ssh, else srun --overlap.
CLUSTER_ATTACH_MODE="${CLUSTER_ATTACH_MODE:-auto}"
# Some sites' submit filters require -p/-A (and a gres for GPU) on the --overlap step too.
SRUN_GRES_OPT=""; [ -n "$DEV_GPUS" ]      && SRUN_GRES_OPT="--gres=gpu:${DEV_GPUS}"
SRUN_PART_OPT=""; [ -n "$DEV_PARTITION" ] && SRUN_PART_OPT="-p ${DEV_PARTITION}"
SRUN_ACCT_OPT=""; [ -n "$DEV_ACCOUNT" ]   && SRUN_ACCT_OPT="-A ${DEV_ACCOUNT}"
# Without an explicit cpu request srun --overlap binds each step to ONE cpu, so every exec (training
# included) ran pinned to core 0 of the whole allocation -- measured ~2.9x slower. Reuse the sentinel's
# own --cpus-per-task so a step gets the same share the batch job asked for.
SRUN_CPUS_OPT=""; [ -n "$DEV_CPUS" ]      && SRUN_CPUS_OPT="--cpus-per-task=${DEV_CPUS}"
_srun_opts() { DEV_SRUN_OPTS="${SRUN_PART_OPT} ${SRUN_ACCT_OPT} -N 1 -n 1 -t ${DEV_TIME:-48:00:00} ${SRUN_GRES_OPT} ${SRUN_CPUS_OPT} ${CLUSTER_SRUN_EXTRA:-}"; }
_srun_opts

# Local code to mirror to the cluster (the manager workspace root -- flat layout: scripts/ + the
# resources/<pkg> repos; the shared .sif provides isaacsim + Isaac Lab so no IsaacLab tree is required).
LOCAL_ISAACLAB_DIR="${LOCAL_ISAACLAB_DIR:-${MAIN_CHECKOUT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}}"

# Local state -- keyed by CLUSTER so concurrent dev sessions on different clusters (e.g. a
# Delta box and a TACC box at the same time) keep separate state and don't clobber each other.
STATE_DIR="${HOME}/.cluster_dev/${CLUSTER}"
STATE_FILE="${STATE_DIR}/state"          # KEY=VALUE: JOBID, JOB_STATE, NODE, SUBMIT_TS, START_TS
WATCH_LOG="${STATE_DIR}/watch.log"
WATCH_PID="${STATE_DIR}/watch.pid"
POLL_SECONDS="${POLL_SECONDS:-120}"

# SSH multiplexing (mirrors cluster_interface.sh but with a 48h-persistent master).
SSH_CONTROL_DIR="${HOME}/.ssh/cm"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${SSH_CONTROL_DIR}/%C" -o ControlPersist=48h \
          -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes -o ConnectTimeout=60)

mkdir -p "$STATE_DIR" "$SSH_CONTROL_DIR"; chmod 700 "$SSH_CONTROL_DIR" "$STATE_DIR"

#============================================================================
# Helpers
#============================================================================
log() { echo -e "[cluster_dev] $*"; }
err() { echo -e "\033[31m[cluster_dev] ERROR: $*\033[0m" >&2; }

state_get() { [ -f "$STATE_FILE" ] && grep -E "^$1=" "$STATE_FILE" | tail -1 | cut -d= -f2- || true; }
state_set() {  # state_set KEY VALUE  (idempotent upsert)
    mkdir -p "$STATE_DIR"; touch "$STATE_FILE"
    grep -v -E "^$1=" "$STATE_FILE" > "${STATE_FILE}.tmp" 2>/dev/null || true
    echo "$1=$2" >> "${STATE_FILE}.tmp"; mv "${STATE_FILE}.tmp" "$STATE_FILE"
}

master_alive() { ssh "${SSH_OPTS[@]}" -O check "$CLUSTER_LOGIN" >/dev/null 2>&1; }

# This user's RUNNING sentinels on the cluster, newest first: "<jobid> <node> <time left>" per line.
live_sentinels() { on_login "squeue -u \$USER -h -t RUNNING -n cluster-dev-box -o '%i %N %L' 2>/dev/null" | sort -rn || true; }

ensure_master() {
    if master_alive; then log "SSH master already open to $CLUSTER_LOGIN."; return 0; fi
    log "Opening SSH master to $CLUSTER_LOGIN -- APPROVE THE 2FA PROMPT NOW (one time, if your site uses it)."
    # -f backgrounds only AFTER auth completes, so the 2FA/password prompt is interactive.
    ssh -fN "${SSH_OPTS[@]}" "$CLUSTER_LOGIN"
    master_alive && log "Master established (persists 48h, kept warm by keepalives)." \
                 || { err "Master failed to open."; return 1; }
}

# Run a command on the cluster login node over the master (no 2FA).
on_login() { ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" "$@"; }

# Re-read the tracked job's state from Slurm into the state file. The watcher is the only other writer, so
# once it dies the file keeps its last value; squeue drops ended jobs, so fall back to sacct for them.
refresh_job_state() {
    local jobid row state; jobid="$(state_get JOBID)"
    [ -n "$jobid" ] && master_alive || return 0
    # squeue exits non-zero once a job has left the queue -- the case this exists for
    row="$(on_login "squeue -j $jobid -h -o '%T %N' 2>/dev/null" | tail -n1 || true)"
    state="$(echo "$row" | awk '{print $1}')"
    if [ -n "$state" ]; then
        [ "$state" = "RUNNING" ] && state_set NODE "$(echo "$row" | awk '{print $2}')"
    else
        state="$(on_login "sacct -j $jobid -X -n -o State%30 2>/dev/null" | awk 'NF{print $1; exit}' || true)"
    fi
    [ -n "$state" ] && state_set JOB_STATE "$state"
}

# node_exec.sh must live inside the synced workspace so it rides the rsync to the cluster.
stage_node_exec() {
    local dst_file="${LOCAL_ISAACLAB_DIR}/scripts/cluster/cluster_dev/node_exec.sh"
    [ -e "$dst_file" ] && [ "${SCRIPT_DIR}/node_exec.sh" -ef "$dst_file" ] && return 0
    mkdir -p "$(dirname "$dst_file")"
    cp "${SCRIPT_DIR}/node_exec.sh" "$dst_file"
    chmod +x "$dst_file"
}

# The selected config's env goes straight to its remote config dir, where node_exec.sh reads it
# (NODE_EXEC_ENV); no local or remote slot is shared between configs.
REMOTE_ENV_FILE="${REMOTE_ISAACLAB_DIR}/scripts/cluster/config/${CLUSTER}/.env.cluster"
push_env_cluster() {
    [ -f "$ENV_FILE" ] || { err "Cluster env file not found: $ENV_FILE"; return 1; }
    on_login "mkdir -p '$(dirname "$REMOTE_ENV_FILE")'"
    rsync -t -e "ssh ${SSH_OPTS[*]}" "$ENV_FILE" "${CLUSTER_LOGIN}:${REMOTE_ENV_FILE}"
}

rsync_code() {
    # Honor .dockerignore + prune git/venv/logs/wandb/exports/sif. No -z (assets are incompressible);
    # -t preserves mtimes so re-syncs skip unchanged assets; --info=progress2 shows overall progress.
    # A per-cluster config/<name>/.rsync-exclude (rsync exclude patterns, one per line) prunes repos that
    # must not deploy to THIS cluster (e.g. another session's *_pbfm forks, which shadow package names).
    local extra_excludes=() cfg dest remote_only name
    # Every config that syncs to this destination contributes its .rsync-exclude, so a sync through one
    # config cannot delete what a sibling config protects. Destinations compare as expanded values.
    for cfg in "${SCRIPT_DIR}"/../config/*/; do
        dest=""
        [ -f "${cfg}.env.cluster" ] && \
            dest="$(bash -c 'source "$1" >/dev/null 2>&1; printf %s "${CLUSTER_ISAACLAB_DIR:-}"' _ "${cfg}.env.cluster")"
        [ "$dest" = "$REMOTE_ISAACLAB_DIR" ] || [ "$(basename "$cfg")" = "$CLUSTER" ] || continue
        [ -f "${cfg}.rsync-exclude" ] && extra_excludes+=(--exclude-from="${cfg}.rsync-exclude")
    done
    # the API key: owner-only here, which -p carries to the cluster
    [ ! -f "${LOCAL_ISAACLAB_DIR}/scripts/.env.wandb" ] || chmod go-rwx "${LOCAL_ISAACLAB_DIR}/scripts/.env.wandb"
    # A fresh profile's workspace may not exist yet, nor its parent (rsync creates only the last level). A dry run
    # leaves the remote alone, and a created directory is named, so a mistyped path shows.
    if [[ " $* " != *" -n "* ]]; then
        on_login "[ -d '${REMOTE_ISAACLAB_DIR}' ] || { mkdir -p '${REMOTE_ISAACLAB_DIR}' && \
            echo '[cluster_dev] created ${REMOTE_ISAACLAB_DIR}'; }" || {
            err "cannot create ${REMOTE_ISAACLAB_DIR} on the remote"; return 1; }
    fi
    # resources/* repos that exist only on the remote (cluster-only forks, or retired here with only
    # worktrees/ left) are never deleted.
    remote_only="$(on_login "[ ! -d '${REMOTE_ISAACLAB_DIR}/resources' ] || ls -1 '${REMOTE_ISAACLAB_DIR}/resources'")" || {
        err "cannot list ${REMOTE_ISAACLAB_DIR}/resources on the remote; refusing to sync with --delete"; return 1; }
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        if [ ! -e "${LOCAL_ISAACLAB_DIR}/resources/${name}" ] || \
           [ -z "$(ls -A "${LOCAL_ISAACLAB_DIR}/resources/${name}" 2>/dev/null | grep -vx worktrees)" ]; then
            extra_excludes+=(--exclude="/resources/${name}")
        fi
    done <<< "$remote_only"
    # extra rsync args from the caller, e.g. `sync --dry-run` -> -n --itemize-changes
    rsync -rlptvh --delete --info=progress2 "$@" \
        `# worktrees are per-session state: local ones never ship, remote ones are never deleted` \
        --exclude='**/worktrees/' \
        `# hydra run dirs are run output, like logs/` \
        --exclude='/outputs/' --exclude='/resources/*/outputs/' \
        `# local working docs and session worktrees never ship` \
        --exclude='/.claude/' --exclude='/resources/*/.claude/' \
        `# the selected config's env is read from config/<name>/ on the node, not from a shared slot` \
        --exclude='/scripts/cluster/.env.cluster' \
        `# legacy pre-reorg tree: un-protect it so --delete can clear it despite excluded contents` \
        --filter='R /source/***' \
        `# artifacts/ is the out-of-sync tree both ways: local exports never ship, and cluster-only data` \
        `# lives under the remote artifacts/ where --delete cannot touch it -- put new excludable data there` \
        --exclude='/artifacts' \
        `# staged code trees live only on the remote` \
        --exclude='/trees/' \
        `# FIRST match wins, so per-cluster protection must precede the allowlist below -- an include` \
        `# that matched first would mark cluster-only state as syncable and --delete would erase it` \
        "${extra_excludes[@]}" \
        --filter=':- .dockerignore' \
        --exclude='*.git*' --exclude='ilab/' --exclude='.venv/' \
        --exclude='wandb/' --exclude='logs/' --exclude='.vscode/' \
        --filter='-p **/__pycache__/' --exclude='scripts/cluster/exports/' --exclude='*.sif' --exclude='*.tar' --exclude='.backup/' \
        `# motion_datasets ALLOWLIST: sync only training .pt + sidecars; any new intermediate type is dropped by default` \
        `# remote-only bundles are protected: P is receiver-side, so the allowlist still decides what ships` \
        --filter='P /resources/motion_datasets/**' \
        --include='resources/motion_datasets/**/' \
        --include='resources/motion_datasets/**.pt' \
        --include='resources/motion_datasets/**.arena.json' --include='resources/motion_datasets/**.courts.json' \
        --include='resources/motion_datasets/**.manifest.json' \
        --exclude='resources/motion_datasets/**' \
        -e "ssh ${SSH_OPTS[*]}" \
        "${LOCAL_ISAACLAB_DIR}/" "${CLUSTER_LOGIN}:${REMOTE_ISAACLAB_DIR}/"
}

# Resolve the node of the current sentinel job from squeue (authoritative).
job_node() {  # job_node JOBID -> nodename or empty
    on_login "squeue -j $1 -h -o '%N'" 2>/dev/null | tr -d '[:space:]'
}
job_state() { on_login "squeue -j $1 -h -o '%T'" 2>/dev/null | tr -d '[:space:]'; }

#============================================================================
# Subcommands
#============================================================================
cmd_start() {
    local sync=1
    while [ $# -gt 0 ]; do
        case "$1" in
            --no-sync) sync=""; shift ;;
            *) err "start: unknown argument '$1' (usage: start [--no-sync])"; exit 1 ;;
        esac
    done
    ensure_master
    # 1) mirror local code up first so the node has the latest on attach.
    if [ -z "$sync" ]; then
        log "Skipping the code sync: run from staged trees (develop stage / exec --tree) or 'develop sync' later."
        local remote_sha
        remote_sha="$(on_login "sha256sum '${REMOTE_ISAACLAB_DIR}/scripts/cluster/cluster_dev/node_exec.sh' 2>/dev/null" | cut -c1-64 || true)"
        [ "$remote_sha" = "$(sha256sum "${SCRIPT_DIR}/node_exec.sh" | cut -c1-64)" ] ||
            log "WARNING: the remote shared node_exec.sh differs from this one; shared-mode exec runs the remote copy."
    elif [ -d "$LOCAL_ISAACLAB_DIR" ]; then
        log "Syncing code -> ${REMOTE_ISAACLAB_DIR} (excludes git/venv/logs/wandb)..."
        stage_node_exec
        rsync_code || err "rsync failed (continuing; you can re-run './cluster_dev.sh sync')."
    fi
    push_env_cluster || err "could not push ${ENV_FILE} (exec falls back to scripts/cluster/.env.cluster)."
    # 2) render + submit the sentinel sbatch from the template. Only $SBATCH_DIRECTIVES is substituted;
    # runtime refs ($SLURM_JOB_ID, $HOME, $(hostname)) are left for the job to evaluate on the node.
    local sbatch_remote=".cluster_dev_sentinel.sbatch"
    export SBATCH_DIRECTIVES
    envsubst '$SBATCH_DIRECTIVES' < "${SCRIPT_DIR}/sentinel.sbatch" | on_login "cat > ${sbatch_remote}"
    local jobid raw
    raw="$(on_login "sbatch --parsable ${sbatch_remote}")" || { err "sbatch failed."; exit 1; }
    # --parsable prints "<jobid>[;<cluster>]". Some login shells (e.g. TACC) emit banner/
    # balance lines first, so DON'T strip digits globally -- take the LAST non-empty line and
    # the field before any ';', then keep only its digits.
    jobid="$(printf '%s\n' "$raw" | sed '/^[[:space:]]*$/d' | tail -n1 | cut -d';' -f1 | tr -dc '0-9')"
    [ -n "$jobid" ] || { err "Could not parse job id from sbatch output: ${raw}"; exit 1; }
    : > "$STATE_FILE"
    state_set JOBID "$jobid"; state_set JOB_STATE "SUBMITTED"; state_set NODE ""
    state_set SUBMIT_TS "$(date -u +%FT%TZ)"
    log "Submitted sentinel job ${jobid} (partition=${DEV_PARTITION:-default}, gpus=${DEV_GPUS:-default}, time=${DEV_TIME:-default})."
    # 3) background watcher tracks the (possibly long) queue wait.
    cmd_watch_start "$jobid"
    log "Watcher started. It may queue for hours -- check './cluster_dev.sh status' anytime."
    log "When NODE is set + JOB_STATE=RUNNING, use './cluster_dev.sh attach' or 'exec'."
}

cmd_watch_start() {  # spawn the detached poll loop
    local jobid="$1"
    [ -f "$WATCH_PID" ] && kill "$(cat "$WATCH_PID")" 2>/dev/null || true
    nohup "${BASH_SOURCE[0]}" __watch "$jobid" >>"$WATCH_LOG" 2>&1 &
    echo $! > "$WATCH_PID"; disown || true
}

cmd_watch_loop() {  # internal: poll squeue until RUNNING/terminal, record node
    local jobid="$1" st node
    echo "[$(date -u +%FT%TZ)] watching job $jobid (poll ${POLL_SECONDS}s)" >> "$WATCH_LOG"
    while :; do
        if ! master_alive; then
            echo "[$(date -u +%FT%TZ)] master down -- cannot poll; will retry" >> "$WATCH_LOG"
            sleep "$POLL_SECONDS"; continue
        fi
        st="$(job_state "$jobid")"
        if [ -z "$st" ]; then
            # not in squeue anymore -> finished/failed/cancelled
            state_set JOB_STATE "GONE"
            echo "[$(date -u +%FT%TZ)] job $jobid no longer in queue (ended/cancelled)" >> "$WATCH_LOG"
            break
        fi
        state_set JOB_STATE "$st"
        if [ "$st" = "RUNNING" ]; then
            node="$(job_node "$jobid")"
            state_set NODE "$node"; state_set START_TS "$(date -u +%FT%TZ)"
            echo "[$(date -u +%FT%TZ)] job $jobid RUNNING on $node" >> "$WATCH_LOG"
            # Auto-detect q4: can we ssh login->node WITHOUT interactive auth (2FA)?
            # BatchMode=yes makes ssh fail fast instead of prompting if creds are needed.
            if [ -n "$node" ] && ssh "${SSH_OPTS[@]}" -o BatchMode=yes -o ConnectTimeout=15 \
                    -J "$CLUSTER_LOGIN" "${DEV_USER}@${node}" true 2>/dev/null; then
                state_set SSH_NODE_OK yes
                echo "[$(date -u +%FT%TZ)] login->node ssh works passwordlessly -> attach via direct ssh" >> "$WATCH_LOG"
            else
                state_set SSH_NODE_OK no
                echo "[$(date -u +%FT%TZ)] login->node ssh needs auth/blocked -> attach via srun --overlap" >> "$WATCH_LOG"
            fi
            break
        fi
        echo "[$(date -u +%FT%TZ)] job $jobid state=$st (queued)" >> "$WATCH_LOG"
        sleep "$POLL_SECONDS"
    done
}

cmd_status() {
    local jobid; jobid="$(state_get JOBID)"
    refresh_job_state
    echo "-- cluster_dev status --"
    echo "  job id     : ${jobid:-<none>}"
    echo "  job state  : $(state_get JOB_STATE)"
    echo "  node       : $(state_get NODE)"
    echo "  attach via : $(s=$(state_get SSH_NODE_OK); [ "$s" = yes ] && echo "direct ssh" || { [ "$s" = no ] && echo "srun --overlap" || echo "(probed at job start)"; })"
    echo "  submitted  : $(state_get SUBMIT_TS)"
    echo "  started    : $(state_get START_TS)"
    echo "  master     : $(master_alive && echo UP || echo DOWN)"
    echo "  watcher    : $([ -f "$WATCH_PID" ] && kill -0 "$(cat "$WATCH_PID")" 2>/dev/null && echo "alive (pid $(cat "$WATCH_PID"))" || echo "not running")"
    if [ -n "$jobid" ] && master_alive; then
        echo "  live squeue:"; on_login "squeue -j $jobid 2>/dev/null" | sed 's/^/    /' || true
    fi
    if [ "$(state_get JOB_STATE)" != RUNNING ] && master_alive; then
        echo "  RUNNING sentinels (exec uses the only one, or pass DEV_JOBID=<id>):"
        live_sentinels | sed 's/^/    /'
    fi
    [ -f "$WATCH_LOG" ] && { echo "  recent watch log:"; tail -3 "$WATCH_LOG" | sed 's/^/    /'; }
}

# Free GB on the filesystem holding a remote dir: the smaller of df and, where `quota -s` prints a table row
# (Delta's format: | path | used | soft quota | ...) for a prefix of the dir, the quota headroom.
remote_free_gb() {  # remote_free_gb DIR -> integer GB, or nothing if it cannot be read
    on_login "d=$(printf %q "$1"); mkdir -p \"\$d\" 2>/dev/null; r=\$(readlink -f \"\$d\");
        df_gb=\$(df -Pk \"\$r\" 2>/dev/null | awk 'NR==2 {printf \"%d\", \$4 / 1048576}');
        q_gb=\$(quota -s 2>/dev/null | awk -F'|' -v d=\"\$r\" '
            function gb(x,  n, u) { gsub(/ /, \"\", x); n = x + 0; u = substr(x, length(x));
                return u == \"T\" ? n * 1024 : u == \"G\" ? n : u == \"M\" ? n / 1024 : n / 1048576 }
            NF > 4 { p = \$2; gsub(/ /, \"\", p); if (p != \"\" && index(d \"/\", p \"/\") == 1)
                printf \"%d\\n\", gb(\$4) - gb(\$3) }' | sort -n | head -1);
        echo \$(printf '%s\\n' \$df_gb \$q_gb | sort -n | head -1)" 2>/dev/null
}

# Refuse to start when a remote dir has less than CLUSTER_MIN_FREE_GB free: a full quota fails every checkpoint
# write while the run keeps going (python.sh still exits 0).
check_space() {  # check_space DIR WHAT
    local free min="${CLUSTER_MIN_FREE_GB:-10}"
    free="$(remote_free_gb "$1")"
    if [ -z "$free" ]; then
        log "WARNING: could not read the free space of $1; not checking"
    elif [ "$free" -lt "$min" ]; then
        err "only ${free} GB free for $2 at $1 (CLUSTER_MIN_FREE_GB=${min}); free space or pass --no-space-check"
        exit 1
    fi
}

# Ensures job RUNNING + master up; sets DD_JOBID, DD_NODE, DD_MODE (ssh|srun).
require_running() {
    # DEV_JOBID overrides the tracked state file: target any RUNNING job of this user (e.g. a
    # second sentinel the watcher isn't tracking). Such jobs are always driven via srun --overlap
    # (no ssh-node probe), so it works even when the state file points at a different job.
    if [ -n "${DEV_JOBID:-}" ]; then
        DD_JOBID="$DEV_JOBID"
        master_alive || { err "SSH master is down -- re-run './cluster_dev.sh start' (needs 2FA)."; exit 1; }
        local row state part acct gpus; row="$(on_login "squeue -j ${DD_JOBID} -h -o '%T %N %P %a %b' 2>/dev/null")"
        state="$(echo "$row" | awk '{print $1}')"
        [ "$state" = "RUNNING" ] || { err "Job ${DD_JOBID} not RUNNING (state=${state:-gone})."; exit 1; }
        DD_NODE="$(echo "$row" | awk '{print $2}')"; DD_MODE="srun"
        # step with the job's own partition, account and GPU request: a site with several projects (TACC) refuses a
        # step without -A, and this profile's sbatch need not be the one that submitted the job
        part="$(echo "$row" | awk '{print $3}')"; acct="$(echo "$row" | awk '{print $4}')"
        gpus="$(echo "$row" | awk '{print $5}' | grep -oE 'gpu(:[A-Za-z0-9_-]+)?:[0-9]+' | grep -oE '[0-9]+$' | head -1 || true)"
        [ -n "$part" ] && SRUN_PART_OPT="-p ${part}"
        # squeue reports the account lowercased (cda26011), which TACC's submit filter refuses: the profile's own
        # spelling wins when it names the same account
        [ -n "$DEV_ACCOUNT" ] && [ "${acct,,}" = "${DEV_ACCOUNT,,}" ] && acct="$DEV_ACCOUNT"
        [ -n "$acct" ] && [ "$acct" != "(null)" ] && SRUN_ACCT_OPT="-A ${acct}"
        [ -n "$gpus" ] && SRUN_GRES_OPT="--gres=gpu:${gpus}"
        _srun_opts
        log "[override] targeting job ${DD_JOBID} on ${DD_NODE} via srun --overlap"
        return
    fi
    refresh_job_state
    DD_JOBID="$(state_get JOBID)"; DD_NODE="$(state_get NODE)"
    if [ "$(state_get JOB_STATE)" != "RUNNING" ] || [ -z "$DD_JOBID" ]; then
        # the tracked job is gone: fall back to this user's only RUNNING sentinel (any session's)
        local live=""; master_alive && live="$(live_sentinels)"
        if [ -n "$live" ] && [ "$(printf '%s\n' "$live" | wc -l)" -eq 1 ]; then
            log "Tracked job ${DD_JOBID:-<none>} is $(state_get JOB_STATE); using the only RUNNING sentinel ${live%% *}"
            DEV_JOBID="${live%% *}"; require_running; return
        elif [ -n "$live" ]; then
            err "Tracked job ${DD_JOBID:-<none>} is $(state_get JOB_STATE). Pick a RUNNING sentinel with DEV_JOBID=<id>:"
            printf '%s\n' "$live" | sed 's/^/    /' >&2; exit 1
        fi
        err "No running job yet (state=$(state_get JOB_STATE)). Run './cluster_dev.sh status'."; exit 1
    fi
    master_alive || { err "SSH master is down -- re-run './cluster_dev.sh start' (needs 2FA)."; exit 1; }
    DD_MODE="$CLUSTER_ATTACH_MODE"
    if [ "$DD_MODE" = "auto" ]; then
        [ "$(state_get SSH_NODE_OK)" = "yes" ] && [ -n "$DD_NODE" ] && DD_MODE="ssh" || DD_MODE="srun"
    fi
}

cmd_attach() {  # interactive shell on the compute node
    [ "${1:-}" = "--ssh" ] && { CLUSTER_ATTACH_MODE="ssh"; shift; }
    [ "${1:-}" = "--srun" ] && { CLUSTER_ATTACH_MODE="srun"; shift; }
    require_running
    if [ "$DD_MODE" = "ssh" ]; then
        log "ssh -> ${DD_NODE} (direct, proxied via master). Ctrl-D leaves node; job keeps running."
        ssh "${SSH_OPTS[@]}" -t -J "$CLUSTER_LOGIN" "${DEV_USER}@${DD_NODE}" "${@:-bash -l}"
    else
        log "srun --overlap onto job ${DD_JOBID}'s node (auth-safe). Ctrl-D leaves; job keeps running."
        ssh "${SSH_OPTS[@]}" -t "$CLUSTER_LOGIN" \
            "srun --jobid=${DD_JOBID} --overlap ${DEV_SRUN_OPTS} --pty bash -l"
    fi
}

cmd_exec() {  # cluster_dev.sh exec [--detach] [--log FILE] -- <command...>
    #
    # Default (foreground): run the container command tethered to this SSH master. Stdout/stderr
    # stream back to the caller; exit code propagates. Good for short ops (smoke tests, status
    # queries). FRAGILE for long runs -- an SSH master drop kills the in-container process.
    #
    # --detach: spawn a `nohup setsid` wrapper ON THE LOGIN NODE that owns srun (or the
    # inner ssh-to-compute) for the lifetime of the training task. Once disowned, the local
    # SSH master can drop without taking down the wrapper or its srun child. Output is
    # redirected to a log file on the login node (default: $HOME/cluster_dev_run_<ts>.log).
    # Follow it with `cluster_dev.sh tail`. The latest --detach log path is recorded in
    # ~/.cluster_dev/state (LAST_RUN_LOG) so `tail` finds it without args.
    local detach="" logfile="" tree="" space_check=1
    while [ $# -gt 0 ]; do
        case "${1:-}" in
            --detach) detach="1"; shift;;
            --no-space-check) space_check=""; shift;;
            --log) logfile="$2"; shift 2;;
            --tree) [ -n "${2:-}" ] || { err "--tree needs a tree name or <name>-<fingerprint>"; exit 1; }
                    tree="$2"; shift 2;;
            --) shift; break;;
            *) break;;
        esac
    done
    require_running
    [ -n "$space_check" ] && check_space "${CLUSTER_LOGS_DIR:-${REMOTE_ISAACLAB_DIR}/resources/hcrl_isaaclab/logs}" "run logs"
    # exec always runs with the current local config, even on a destination that hasn't been synced
    push_env_cluster || { err "could not push ${ENV_FILE} to ${REMOTE_ENV_FILE}"; exit 1; }
    # every hop re-parses the command, so each one gets its own %q layer and the argv arrives intact
    local args; args="$(printf '%q ' "$@")"
    local nodecmd="NODE_EXEC_ENV=${REMOTE_ENV_FILE} bash ${REMOTE_ISAACLAB_DIR}/scripts/cluster/cluster_dev/node_exec.sh ${args}"
    if [ -n "$tree" ]; then
        tree="$(resolve_tree "$tree")" || exit 1
        log "Using tree ${tree}"
        nodecmd="NODE_EXEC_ENV=${REMOTE_ENV_FILE} NODE_EXEC_RESOURCES=${tree}/resources"
        nodecmd+=" bash ${tree}/scripts/cluster/cluster_dev/node_exec.sh ${args}"
    fi
    if [ -z "$detach" ]; then
        log "[${DD_MODE}] container exec on job ${DD_JOBID}: ${args}"
        if [ "$DD_MODE" = "ssh" ]; then
            ssh "${SSH_OPTS[@]}" -J "$CLUSTER_LOGIN" "${DEV_USER}@${DD_NODE}" "$nodecmd"
        else
            ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" \
                "srun --jobid=${DD_JOBID} --overlap ${DEV_SRUN_OPTS} bash -lc $(printf %q "$nodecmd")"
        fi
        return
    fi
    # --detach path. The default log lives in the login node's $HOME, resolved here so the printed and
    # recorded path is absolute (watchers cannot expand a literal $HOME).
    if [ -z "$logfile" ]; then
        local rhome; rhome="$(on_login 'printf %s "$HOME"')"
        [ -n "$rhome" ] || { err "could not resolve \$HOME on ${CLUSTER_LOGIN}"; exit 1; }
        logfile="${rhome}/cluster_dev_run_$(date -u +%Y%m%d-%H%M%S).log"
    fi
    # Build the inner command (what the detached wrapper will exec). Always go via the login
    # node -- some sites (e.g. Delta) block direct login->node ssh without re-auth in srun-mode, so even in
    # ssh-mode we keep the detach point on the login node for consistency.
    local inner
    if [ "$DD_MODE" = "ssh" ]; then
        inner="ssh -J ${CLUSTER_LOGIN} ${DEV_USER}@${DD_NODE} bash -lc $(printf %q "$(printf %q "$nodecmd")")"
    else
        inner="srun --jobid=${DD_JOBID} --overlap ${DEV_SRUN_OPTS} bash -lc $(printf %q "$nodecmd")"
    fi
    log "[${DD_MODE} detached] container exec on job ${DD_JOBID}: ${args}"
    log "Log on login node: ${logfile}   (follow with: $(basename "${BASH_SOURCE[0]}") tail)"
    ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" \
        "nohup setsid bash -c $(printf %q "$inner") > ${logfile} 2>&1 < /dev/null & disown; sleep 0.3; echo \"[cluster_dev] login-side wrapper pid=\$(pgrep -nf 'nohup setsid bash' || echo ?)\""
    # DEV_JOBID means an ad-hoc target, typically ANOTHER session's job -- recording the log path would
    # overwrite the tracked session's LAST_RUN_LOG, so print it instead and leave shared state alone.
    if [ -n "${DEV_JOBID:-}" ]; then
        log "DEV_JOBID set: not recording LAST_RUN_LOG. Follow with: tail -f ${logfile} on ${CLUSTER_LOGIN}."
    else
        state_set LAST_RUN_LOG "$logfile"
    fi
}

cmd_tail() {  # cluster_dev.sh tail [LOGFILE]  : follow a detached --detach log on the login node
    local logfile="${1:-}"
    [ -z "$logfile" ] && logfile="$(state_get LAST_RUN_LOG)"
    [ -z "$logfile" ] && { err "No detached run on record. Use 'exec --detach -- <cmd>' first."; exit 1; }
    ensure_master
    log "Tailing ${logfile} on ${CLUSTER_LOGIN} (Ctrl-C to stop; training keeps running)..."
    # -F so it survives the file being missing or rotated.
    ssh "${SSH_OPTS[@]}" -t "$CLUSTER_LOGIN" "tail -F ${logfile}"
}

cmd_sync() {  # sync [--dry-run] : re-mirror local code -> cluster isaaclab dir (and onto the live node workspace)
    ensure_master
    if [ "${1:-}" = "--dry-run" ]; then
        rsync_code -n --itemize-changes
        log "Dry run: nothing was transferred or deleted (lines starting '*deleting' would be removed)."
        return
    fi
    stage_node_exec
    rsync_code
    push_env_cluster
    log "Synced to ${REMOTE_ISAACLAB_DIR}."
}

cmd_open() {  # open (or confirm) the SSH control master only -- no sync, no job actions
    ensure_master
}

cmd_kill() {  # cluster_dev.sh kill [--all | STEP...] : scancel launched run steps, keep the dev job alive
    # `exec` runs live in their own SLURM step but a fresh container (own PID namespace), so pkill
    # from a later exec can NOT see them -- step-scoped scancel from the login node is the reliable kill.
    # DEV_JOBID names a job other than the tracked one, as for exec (two dev jobs on one profile)
    local jobid; jobid="${DEV_JOBID:-$(state_get JOBID)}"
    [ -z "$jobid" ] && { err "No dev job on record. Use 'start' first."; exit 1; }
    ensure_master
    # every step except batch (the sentinel holding the node) and extern (slurm bookkeeping) is a launched run
    local steps
    steps="$(on_login "squeue -s -j $jobid -h -o '%i %M'" | grep -vE "\.(batch|extern) " || true)"
    if [ $# -eq 0 ]; then
        if [ -z "$steps" ]; then
            log "No launched steps running on job $jobid."
        else
            log "Running steps on job $jobid (STEPID  ELAPSED); cancel with 'kill --all' or 'kill <step>...':"
            printf '%s\n' "$steps"
        fi
        return 0
    fi
    local targets=()
    if [ "${1}" = "--all" ]; then
        while read -r sid _; do [ -n "$sid" ] && targets+=("$sid"); done <<< "$steps"
    else
        local s
        for s in "$@"; do  # a full <job>.<step> as given, a bare step on this job
            if [[ "$s" =~ ^[0-9]+\.[0-9]+$ ]]; then targets+=("$s"); else targets+=("${jobid}.${s}"); fi
        done
    fi
    [ ${#targets[@]} -eq 0 ] && { log "Nothing to cancel."; return 0; }
    log "Cancelling step(s): ${targets[*]} (job $jobid and its SSH master stay up)"
    on_login "scancel ${targets[*]}"
}

cmd_stop() {
    local jobid; jobid="$(state_get JOBID)"
    [ -f "$WATCH_PID" ] && kill "$(cat "$WATCH_PID")" 2>/dev/null || true; rm -f "$WATCH_PID"
    if [ -n "$jobid" ] && master_alive; then
        log "Cancelling job $jobid..."; on_login "scancel $jobid" || true
    fi
    state_set JOB_STATE "STOPPED"
    log "Cancelled. Closing SSH master."; ssh "${SSH_OPTS[@]}" -O exit "$CLUSTER_LOGIN" 2>/dev/null || true
}

usage() {
    sed -n '2,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

source "${SCRIPT_DIR}/trees.sh"

case "${1:-}" in
    stage)    shift; cmd_stage "$@" ;;
    trees)    shift; cmd_trees "$@" ;;
    __resolve_tree) shift; ensure_master; resolve_tree "$@" ;;   # internal (tests)
    __free_gb) shift; ensure_master >/dev/null; remote_free_gb "$1" ;;   # internal (tests)
    start)    shift; cmd_start "$@" ;;
    open)     shift; cmd_open "$@" ;;
    status)   shift; cmd_status "$@" ;;
    attach)   shift; cmd_attach "$@" ;;
    exec)     shift; cmd_exec "$@" ;;
    tail)     shift; cmd_tail "$@" ;;
    sync)     shift; cmd_sync "$@" ;;
    kill)     shift; cmd_kill "$@" ;;
    stop)     shift; cmd_stop "$@" ;;
    __watch)  shift; cmd_watch_loop "$@" ;;   # internal (used by nohup)
    ""|-h|--help|help) usage ;;
    *) err "Unknown command '${1}'."; usage; exit 1 ;;
esac
