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
# <cluster>_config/.env.cluster, selected with CLUSTER=<name>. Nothing here is Delta-specific.
#
# Queue a job:              ./cluster_dev.sh start            (approve ONE 2FA prompt)
# Then it self-tracks the (possibly multi-hour) queue wait in the background.
# Check anytime:            ./cluster_dev.sh status
# Use it:                   ./cluster_dev.sh attach           (interactive shell on the node)
#                           ./cluster_dev.sh exec -- <cmd>    (run in container, SSH-tethered)
#                           ./cluster_dev.sh exec --detach -- <cmd>   (run in container, detached
#                                                                      from SSH master; survives
#                                                                      master drops; log on login
#                                                                      node, follow with `tail`)
#                           ./cluster_dev.sh tail             (follow the latest --detach log)
# Tear down:                ./cluster_dev.sh stop
#
# Everything except the first 2FA is non-interactive, so a Claude session can drive
# `status` / `exec` / `attach` over the live master without any credentials.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

#============================================================================
# Config -- sourced from the selected cluster's .env.cluster, with dev-box overrides.
#============================================================================
# Cluster config: CLUSTER=<name> picks ../<name>_config/.env.cluster (default "default"); matches
# cluster_interface.sh, which sets CLUSTER when invoked as `cluster_interface.sh develop`.
CLUSTER="${CLUSTER:-default}"
ENV_FILE="${SCRIPT_DIR}/../${CLUSTER}_config/.env.cluster"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

CLUSTER_LOGIN="${CLUSTER_LOGIN:?CLUSTER_LOGIN not set (expected from ${CLUSTER}_config/.env.cluster)}"
CDEV_USER="${CLUSTER_LOGIN%@*}"
CDEV_LOGIN_HOST="${CDEV_LOGIN_HOST:-${CLUSTER_LOGIN#*@}}"   # login host; round-robin DNS is fine since we always multiplex over one master.
REMOTE_ISAACLAB_DIR="${CLUSTER_ISAACLAB_DIR:?CLUSTER_ISAACLAB_DIR not set}"

# Sentinel job resources. These are cluster-agnostic: set whatever the target queue needs
# in <cluster>_config/.env.cluster (CDEV_ACCOUNT, CDEV_PARTITION, CDEV_GPUS_PER_NODE,
# CDEV_CPUS, CDEV_MEM, CDEV_EXCLUSIVE). Anything left UNSET emits no corresponding #SBATCH
# directive, so SLURM falls back to the queue's own default rather than a forced value.
CDEV_ACCOUNT="${CDEV_ACCOUNT:-}"          # e.g. an allocation/charge code; omit where not required
CDEV_PARTITION="${CDEV_PARTITION:-}"      # e.g. gpuA40x4 (Delta), rtx-small (TACC); omit for queue default
CDEV_TIME="${CDEV_TIME:-48:00:00}"        # walltime to hold the node; cap to the queue max
CDEV_GPUS_PER_NODE="${CDEV_GPUS_PER_NODE:-}"  # set to force --gpus-per-node=N; unset -> queue default
CDEV_CPUS="${CDEV_CPUS:-}"                # set to force --cpus-per-task=N; unset -> queue default
CDEV_MEM="${CDEV_MEM:-}"                  # set to force --mem; unset -> queue default
CDEV_EXCLUSIVE="${CDEV_EXCLUSIVE:-}"      # set 1/true/yes to request a whole node (--exclusive)

# Build each #SBATCH directive line; an empty string means the directive is omitted entirely.
# NB: use `[ -n "$X" ] && VAR=...` (not `VAR="$( ... && echo )"`) -- a top-level assignment
# whose command substitution exits non-zero trips `set -e` and would abort the script silently.
CDEV_ACCOUNT_DIRECTIVE=""
CDEV_PARTITION_DIRECTIVE=""
CDEV_GPUS_DIRECTIVE=""
CDEV_CPUS_DIRECTIVE=""
CDEV_MEM_DIRECTIVE=""
CDEV_EXCLUSIVE_DIRECTIVE=""
[ -n "$CDEV_ACCOUNT" ]       && CDEV_ACCOUNT_DIRECTIVE="#SBATCH -A ${CDEV_ACCOUNT}"
[ -n "$CDEV_PARTITION" ]     && CDEV_PARTITION_DIRECTIVE="#SBATCH -p ${CDEV_PARTITION}"
[ -n "$CDEV_GPUS_PER_NODE" ] && CDEV_GPUS_DIRECTIVE="#SBATCH --gpus-per-node=${CDEV_GPUS_PER_NODE}"
[ -n "$CDEV_CPUS" ]          && CDEV_CPUS_DIRECTIVE="#SBATCH --cpus-per-task=${CDEV_CPUS}"
[ -n "$CDEV_MEM" ]           && CDEV_MEM_DIRECTIVE="#SBATCH --mem=${CDEV_MEM}"
case "$CDEV_EXCLUSIVE" in
    1|true|TRUE|True|yes|YES|Yes) CDEV_EXCLUSIVE_DIRECTIVE="#SBATCH --exclusive" ;;
esac

# How to get onto the node for attach/exec: "auto" probes at job start whether
# login->node ssh works without re-auth; else falls back to srun --overlap.
# Override with CDEV_ATTACH_MODE=ssh|srun. For srun-mode GPU access we request the gres
# only when a count is known; otherwise the --overlap step inherits the job's allocation.
CDEV_ATTACH_MODE="${CDEV_ATTACH_MODE:-auto}"
if [ -n "$CDEV_GPUS_PER_NODE" ]; then
    CDEV_SRUN_GRES="${CDEV_SRUN_GRES:-gpu:${CDEV_GPUS_PER_NODE}}"
fi
CDEV_SRUN_GRES="${CDEV_SRUN_GRES:-}"
if [ -n "$CDEV_SRUN_GRES" ]; then
    CDEV_SRUN_GRES_OPT="--gres=${CDEV_SRUN_GRES}"
else
    CDEV_SRUN_GRES_OPT=""
fi
# Some sites (e.g. TACC) enforce a submit filter on EVERY srun step -- including an --overlap
# step joining a running job -- requiring -p, -N and -n (and sometimes -A). Pass them through.
# -N 1 -n 1 is correct for a single-node dev box; override the whole set via CDEV_SRUN_EXTRA.
CDEV_SRUN_PART_OPT=""; [ -n "$CDEV_PARTITION" ] && CDEV_SRUN_PART_OPT="-p ${CDEV_PARTITION}"
CDEV_SRUN_ACCT_OPT="";  [ -n "$CDEV_ACCOUNT" ]   && CDEV_SRUN_ACCT_OPT="-A ${CDEV_ACCOUNT}"
CDEV_SRUN_OPTS="${CDEV_SRUN_PART_OPT} ${CDEV_SRUN_ACCT_OPT} -N 1 -n 1 -t ${CDEV_TIME} ${CDEV_SRUN_GRES_OPT} ${CDEV_SRUN_EXTRA:-}"

# Local code to mirror to the cluster (the manager workspace root -- flat layout: scripts/ + the
# resources/<pkg> repos; the shared .sif provides isaacsim + Isaac Lab so no IsaacLab tree is required).
LOCAL_ISAACLAB_DIR="${LOCAL_ISAACLAB_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

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

# node_exec.sh must live INSIDE the IsaacLab tree (so it rides the rsync to the cluster and can
# source the staged docker/cluster/.env.* like run_singularity.sh does). Copy it in from
# our source-of-truth before each sync.
stage_node_exec() {
    local dst="${LOCAL_ISAACLAB_DIR}/scripts/cluster/cluster_dev"
    local src="${SCRIPT_DIR}/node_exec.sh"
    local dst_file="${dst}/node_exec.sh"
    mkdir -p "$dst"
    # If src == dst (e.g. cluster_dev.sh was copied INTO LOCAL_ISAACLAB_DIR by scripts/cluster.sh,
    # so SCRIPT_DIR is already the staging dir), skip the cp -- it would fail with
    # "are the same file".
    if [ -e "$dst_file" ] && [ "$src" -ef "$dst_file" ]; then
        chmod +x "$dst_file"
        stage_env_cluster
        return 0
    fi
    cp "$src" "$dst_file"
    chmod +x "$dst_file"
    stage_env_cluster
}

# Stage the SELECTED cluster's .env.cluster into the IsaacLab tree so it rides the rsync and
# node_exec.sh (which sources ../.env.cluster ON THE NODE) picks up THIS cluster's paths/modules
# (CLUSTER_ISAACLAB_DIR, CLUSTER_SIF_PATH, CDEV_MODULE_LOAD, ...) -- not whatever another cluster's
# session last left in the shared staging slot. This is what makes node_exec cluster-agnostic.
stage_env_cluster() {
    local env_dst="${LOCAL_ISAACLAB_DIR}/scripts/cluster/.env.cluster"
    [ -f "$ENV_FILE" ] || { err "Cluster env file not found: $ENV_FILE"; return 1; }
    [ "$ENV_FILE" -ef "$env_dst" ] && return 0
    mkdir -p "$(dirname "$env_dst")"
    cp "$ENV_FILE" "$env_dst"
}

rsync_code() {
    # Match cluster_interface.sh: honor .dockerignore (prunes docker/cluster/exports = 50GB,
    # .git, logs, wandb, etc.). No -z: assets are large/incompressible, compression just
    # bottlenecks CPU. -t preserves mtimes so re-syncs skip the 22GB of unchanged assets.
    # --info=progress2 gives a visible overall progress bar.
    rsync -rlptvh --delete --info=progress2 \
        --filter=':- .dockerignore' \
        --exclude='*.git*' --exclude='ilab/' --exclude='.venv/' \
        --exclude='wandb/' --exclude='logs/' --exclude='.vscode/' \
        --exclude='**/__pycache__/' --exclude='docker/cluster/exports/' --exclude='scripts/cluster/exports/' --exclude='*.sif' --exclude='*.tar' \
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
    ensure_master
    # 1) mirror local code up first so the node has the latest on attach.
    if [ -d "$LOCAL_ISAACLAB_DIR" ]; then
        log "Syncing code -> ${REMOTE_ISAACLAB_DIR} (excludes git/venv/logs/wandb)..."
        stage_node_exec
        rsync_code || err "rsync failed (continuing; you can re-run './cluster_dev.sh sync')."
    fi
    # 2) render + submit the sentinel sbatch from the template.
    local sbatch_remote=".cluster_dev_sentinel.sbatch"
    # IMPORTANT: pass an explicit allowlist to envsubst so it ONLY substitutes our config
    # vars and leaves runtime refs ($SLURM_JOB_ID, $HOME, $(hostname), ${CDEV_TIME_SECONDS:-...})
    # untouched for the job to evaluate on the node.
    export CDEV_TIME CDEV_GPUS_PER_NODE \
        CDEV_ACCOUNT_DIRECTIVE CDEV_PARTITION_DIRECTIVE CDEV_GPUS_DIRECTIVE \
        CDEV_CPUS_DIRECTIVE CDEV_MEM_DIRECTIVE CDEV_EXCLUSIVE_DIRECTIVE
    envsubst '$CDEV_TIME $CDEV_GPUS_PER_NODE $CDEV_ACCOUNT_DIRECTIVE $CDEV_PARTITION_DIRECTIVE $CDEV_GPUS_DIRECTIVE $CDEV_CPUS_DIRECTIVE $CDEV_MEM_DIRECTIVE $CDEV_EXCLUSIVE_DIRECTIVE' \
        < "${SCRIPT_DIR}/sentinel.sbatch" | on_login "cat > ${sbatch_remote}"
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
    log "Submitted sentinel job ${jobid} (partition=${CDEV_PARTITION:-queue-default}, gpus=${CDEV_GPUS_PER_NODE:-queue-default}, time=${CDEV_TIME})."
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
                    -J "$CLUSTER_LOGIN" "${CDEV_USER}@${node}" true 2>/dev/null; then
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
    [ -f "$WATCH_LOG" ] && { echo "  recent watch log:"; tail -3 "$WATCH_LOG" | sed 's/^/    /'; }
}

# Ensures job RUNNING + master up; sets DD_JOBID, DD_NODE, DD_MODE (ssh|srun).
require_running() {
    # CDEV_JOBID overrides the tracked state file: target any RUNNING job of this user (e.g. a
    # second sentinel the watcher isn't tracking). Such jobs are always driven via srun --overlap
    # (no ssh-node probe), so it works even when the state file points at a different job.
    if [ -n "${CDEV_JOBID:-}" ]; then
        DD_JOBID="$CDEV_JOBID"
        master_alive || { err "SSH master is down -- re-run './cluster_dev.sh start' (needs 2FA)."; exit 1; }
        local row state; row="$(on_login "squeue -j ${DD_JOBID} -h -o '%T %N' 2>/dev/null")"
        state="$(echo "$row" | awk '{print $1}')"
        [ "$state" = "RUNNING" ] || { err "Job ${DD_JOBID} not RUNNING (state=${state:-gone})."; exit 1; }
        DD_NODE="$(echo "$row" | awk '{print $2}')"; DD_MODE="srun"
        log "[override] targeting job ${DD_JOBID} on ${DD_NODE} via srun --overlap"
        return
    fi
    DD_JOBID="$(state_get JOBID)"; DD_NODE="$(state_get NODE)"
    [ "$(state_get JOB_STATE)" = "RUNNING" ] && [ -n "$DD_JOBID" ] || {
        err "No running job yet (state=$(state_get JOB_STATE)). Run './cluster_dev.sh status'."; exit 1; }
    master_alive || { err "SSH master is down -- re-run './cluster_dev.sh start' (needs 2FA)."; exit 1; }
    DD_MODE="$CDEV_ATTACH_MODE"
    if [ "$DD_MODE" = "auto" ]; then
        [ "$(state_get SSH_NODE_OK)" = "yes" ] && [ -n "$DD_NODE" ] && DD_MODE="ssh" || DD_MODE="srun"
    fi
}

cmd_attach() {  # interactive shell on the compute node
    [ "${1:-}" = "--ssh" ] && { CDEV_ATTACH_MODE="ssh"; shift; }
    [ "${1:-}" = "--srun" ] && { CDEV_ATTACH_MODE="srun"; shift; }
    require_running
    if [ "$DD_MODE" = "ssh" ]; then
        log "ssh -> ${DD_NODE} (direct, proxied via master). Ctrl-D leaves node; job keeps running."
        ssh "${SSH_OPTS[@]}" -t -J "$CLUSTER_LOGIN" "${CDEV_USER}@${DD_NODE}" "${@:-bash -l}"
    else
        log "srun --overlap onto job ${DD_JOBID}'s node (auth-safe). Ctrl-D leaves; job keeps running."
        ssh "${SSH_OPTS[@]}" -t "$CLUSTER_LOGIN" \
            "srun --jobid=${DD_JOBID} --overlap ${CDEV_SRUN_OPTS} --pty bash -l"
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
    local detach="" logfile=""
    while [ $# -gt 0 ]; do
        case "${1:-}" in
            --detach) detach="1"; shift;;
            --log) logfile="$2"; shift 2;;
            --) shift; break;;
            *) break;;
        esac
    done
    require_running
    local nodecmd="bash ${REMOTE_ISAACLAB_DIR}/scripts/cluster/cluster_dev/node_exec.sh $*"
    if [ -z "$detach" ]; then
        log "[${DD_MODE}] container exec on job ${DD_JOBID}: $*"
        if [ "$DD_MODE" = "ssh" ]; then
            ssh "${SSH_OPTS[@]}" -J "$CLUSTER_LOGIN" "${CDEV_USER}@${DD_NODE}" "$nodecmd"
        else
            ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" \
                "srun --jobid=${DD_JOBID} --overlap ${CDEV_SRUN_OPTS} bash -lc '${nodecmd}'"
        fi
        return
    fi
    # --detach path. Default log lives in $HOME on the login node (literal `$HOME` is
    # preserved through the local->ssh hop and expanded by the remote bash).
    [ -z "$logfile" ] && logfile="\$HOME/cluster_dev_run_$(date -u +%Y%m%d-%H%M%S).log"
    # Build the inner command (what the detached wrapper will exec). Always go via the login
    # node -- some sites (e.g. Delta) block direct login->node ssh without re-auth in srun-mode, so even in
    # ssh-mode we keep the detach point on the login node for consistency.
    local inner
    if [ "$DD_MODE" = "ssh" ]; then
        inner="ssh -J ${CLUSTER_LOGIN} ${CDEV_USER}@${DD_NODE} bash -lc '${nodecmd}'"
    else
        inner="srun --jobid=${DD_JOBID} --overlap ${CDEV_SRUN_OPTS} bash -lc '${nodecmd}'"
    fi
    log "[${DD_MODE} detached] container exec on job ${DD_JOBID}: $*"
    log "Log on login node: ${logfile}   (follow with: $(basename "${BASH_SOURCE[0]}") tail)"
    # Wrap `inner` in double-quotes for bash -c, escaping any inner " (rare for python args).
    local wrapped="${inner//\"/\\\"}"
    ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" \
        "nohup setsid bash -c \"${wrapped}\" > ${logfile} 2>&1 < /dev/null & disown; sleep 0.3; echo \"[cluster_dev] login-side wrapper pid=\$(pgrep -nf 'nohup setsid bash' || echo ?)\""
    state_set LAST_RUN_LOG "$logfile"
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

cmd_sync() {  # re-mirror local code -> cluster isaaclab dir (and onto the live node workspace)
    ensure_master
    stage_node_exec
    rsync_code
    log "Synced to ${REMOTE_ISAACLAB_DIR}."
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
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
    start)    shift; cmd_start "$@" ;;
    status)   shift; cmd_status "$@" ;;
    attach)   shift; cmd_attach "$@" ;;
    exec)     shift; cmd_exec "$@" ;;
    tail)     shift; cmd_tail "$@" ;;
    sync)     shift; cmd_sync "$@" ;;
    stop)     shift; cmd_stop "$@" ;;
    __watch)  shift; cmd_watch_loop "$@" ;;   # internal (used by nohup)
    ""|-h|--help|help) usage ;;
    *) err "Unknown command '${1}'."; usage; exit 1 ;;
esac
