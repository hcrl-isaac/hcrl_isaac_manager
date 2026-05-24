#!/usr/bin/env bash
#
# cluster_dev.sh — turn an NCSA Delta gpuA40x4 node into a persistent (≤48h) dev box.
#
# WHY this shape: Delta's *interactive* partition caps at 1h, but *batch* jobs get 48h,
# and NCSA enables "direct ssh to a compute node in a running job". So we submit a 48h
# batch "sentinel" job that just holds a node, then ssh into it (proxied through a
# persistent login-node ControlMaster) and run/develop there. SSH keys are disabled on
# Delta (password+Duo every login), so the ControlMaster socket — opened ONCE with one
# Duo approval and kept warm — is the only way to avoid re-authenticating all day.
#
# Tomorrow's one command:   ./cluster_dev.sh start          (approve ONE Duo push)
# Then it self-tracks the (possibly multi-hour) queue wait in the background.
# Check anytime:            ./cluster_dev.sh status
# Use it:                   ./cluster_dev.sh attach          (interactive shell on the node)
#                           ./cluster_dev.sh exec -- <cmd>   (run a command in the container)
# Tear down:                ./cluster_dev.sh stop
#
# Everything except the first Duo is non-interactive, so a Claude session can drive
# `status` / `exec` / `attach` over the live master without any credentials.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

#============================================================================
# Config — sourced from the existing Delta config, with dev-box overrides.
# Values marked [VERIFY] depend on answers to the questions in the README.
#============================================================================
# Cluster config: CLUSTER=<name> picks ../<name>_config/.env.cluster (default "default"); matches
# cluster_interface.sh, which sets CLUSTER when invoked as `cluster_interface.sh develop`.
CLUSTER="${CLUSTER:-default}"
ENV_FILE="${SCRIPT_DIR}/../${CLUSTER}_config/.env.cluster"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

CLUSTER_LOGIN="${CLUSTER_LOGIN:?CLUSTER_LOGIN not set (expected from ${CLUSTER}_config/.env.cluster)}"
DELTA_USER="${CLUSTER_LOGIN%@*}"
DELTA_LOGIN_HOST="${DELTA_LOGIN_HOST:-${CLUSTER_LOGIN#*@}}"   # [VERIFY q9] pin dt-login01? round-robin is OK because we always multiplex.
REMOTE_ISAACLAB_DIR="${CLUSTER_ISAACLAB_DIR:?CLUSTER_ISAACLAB_DIR not set}"

# Sentinel job resources — [VERIFY q2,q3] account/partition/time/gpus.
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bggq-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA40x4}"
DELTA_TIME="${DELTA_TIME:-48:00:00}"
DELTA_GPUS_PER_NODE="${DELTA_GPUS_PER_NODE:-4}"   # whole node; set 1 to save SUs
DELTA_CPUS="${DELTA_CPUS:-64}"
DELTA_MEM="${DELTA_MEM:-200g}"
# How to get onto the node for attach/exec: "auto" probes at job start whether
# login→node ssh works without re-auth (q4); else falls back to srun --overlap.
# Override with DELTA_ATTACH_MODE=ssh|srun. For srun-mode GPU access we request the gres.
DELTA_ATTACH_MODE="${DELTA_ATTACH_MODE:-auto}"
DELTA_SRUN_GRES="${DELTA_SRUN_GRES:-gpu:${DELTA_GPUS_PER_NODE}}"

# Local code to mirror to Delta (the IsaacLab working tree).
LOCAL_ISAACLAB_DIR="${LOCAL_ISAACLAB_DIR:-/home/emily/hcrl_isaac_manager/resources/IsaacLab}"

# Local state.
STATE_DIR="${HOME}/.cluster_dev"
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
    log "Opening SSH master to $CLUSTER_LOGIN — APPROVE THE DUO PROMPT NOW (one time)."
    # -f backgrounds only AFTER auth completes, so the Duo/password prompt is interactive.
    ssh -fN "${SSH_OPTS[@]}" "$CLUSTER_LOGIN"
    master_alive && log "Master established (persists 48h, kept warm by keepalives)." \
                 || { err "Master failed to open."; return 1; }
}

# Run a command on the Delta login node over the master (no Duo).
on_login() { ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" "$@"; }

# node_exec.sh must live INSIDE the IsaacLab tree (so it rides the rsync to Delta and can
# source the staged docker/cluster/.env.* like run_singularity.sh does). Copy it in from
# our source-of-truth before each sync.
stage_node_exec() {
    local dst="${LOCAL_ISAACLAB_DIR}/docker/cluster/cluster_dev"
    mkdir -p "$dst"
    cp "${SCRIPT_DIR}/node_exec.sh" "${dst}/node_exec.sh"
    chmod +x "${dst}/node_exec.sh"
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
        --exclude='**/__pycache__/' --exclude='docker/cluster/exports/' \
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
        log "Syncing code → ${REMOTE_ISAACLAB_DIR} (excludes git/venv/logs/wandb)…"
        stage_node_exec
        rsync_code || err "rsync failed (continuing; you can re-run './cluster_dev.sh sync')."
    fi
    # 2) render + submit the sentinel sbatch from the template.
    local sbatch_remote=".cluster_dev_sentinel.sbatch"
    # IMPORTANT: pass an explicit allowlist to envsubst so it ONLY substitutes our config
    # vars and leaves runtime refs ($SLURM_JOB_ID, $HOME, $(hostname), ${DELTA_TIME_SECONDS:-…})
    # untouched for the job to evaluate on the node.
    export DELTA_ACCOUNT DELTA_PARTITION DELTA_TIME DELTA_GPUS_PER_NODE DELTA_CPUS DELTA_MEM
    envsubst '$DELTA_ACCOUNT $DELTA_PARTITION $DELTA_TIME $DELTA_GPUS_PER_NODE $DELTA_CPUS $DELTA_MEM' \
        < "${SCRIPT_DIR}/sentinel.sbatch" | on_login "cat > ${sbatch_remote}"
    local jobid
    jobid="$(on_login "sbatch --parsable ${sbatch_remote}")" || { err "sbatch failed."; exit 1; }
    jobid="${jobid//[!0-9]/}"
    [ -n "$jobid" ] || { err "Could not parse job id from sbatch."; exit 1; }
    : > "$STATE_FILE"
    state_set JOBID "$jobid"; state_set JOB_STATE "SUBMITTED"; state_set NODE ""
    state_set SUBMIT_TS "$(date -u +%FT%TZ)"
    log "Submitted sentinel job ${jobid} (${DELTA_PARTITION}, ${DELTA_GPUS_PER_NODE} GPU, ${DELTA_TIME})."
    # 3) background watcher tracks the (possibly long) queue wait.
    cmd_watch_start "$jobid"
    log "Watcher started. It may queue for hours — check './cluster_dev.sh status' anytime."
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
            echo "[$(date -u +%FT%TZ)] master down — cannot poll; will retry" >> "$WATCH_LOG"
            sleep "$POLL_SECONDS"; continue
        fi
        st="$(job_state "$jobid")"
        if [ -z "$st" ]; then
            # not in squeue anymore → finished/failed/cancelled
            state_set JOB_STATE "GONE"
            echo "[$(date -u +%FT%TZ)] job $jobid no longer in queue (ended/cancelled)" >> "$WATCH_LOG"
            break
        fi
        state_set JOB_STATE "$st"
        if [ "$st" = "RUNNING" ]; then
            node="$(job_node "$jobid")"
            state_set NODE "$node"; state_set START_TS "$(date -u +%FT%TZ)"
            echo "[$(date -u +%FT%TZ)] job $jobid RUNNING on $node 🎉" >> "$WATCH_LOG"
            # Auto-detect q4: can we ssh login→node WITHOUT interactive auth (Duo)?
            # BatchMode=yes makes ssh fail fast instead of prompting if creds are needed.
            if [ -n "$node" ] && ssh "${SSH_OPTS[@]}" -o BatchMode=yes -o ConnectTimeout=15 \
                    -J "$CLUSTER_LOGIN" "${DELTA_USER}@${node}" true 2>/dev/null; then
                state_set SSH_NODE_OK yes
                echo "[$(date -u +%FT%TZ)] login→node ssh works passwordlessly → attach via direct ssh" >> "$WATCH_LOG"
            else
                state_set SSH_NODE_OK no
                echo "[$(date -u +%FT%TZ)] login→node ssh needs auth/blocked → attach via srun --overlap" >> "$WATCH_LOG"
            fi
            break
        fi
        echo "[$(date -u +%FT%TZ)] job $jobid state=$st (queued)" >> "$WATCH_LOG"
        sleep "$POLL_SECONDS"
    done
}

cmd_status() {
    local jobid; jobid="$(state_get JOBID)"
    echo "── cluster_dev status ──"
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
    DD_JOBID="$(state_get JOBID)"; DD_NODE="$(state_get NODE)"
    [ "$(state_get JOB_STATE)" = "RUNNING" ] && [ -n "$DD_JOBID" ] || {
        err "No running job yet (state=$(state_get JOB_STATE)). Run './cluster_dev.sh status'."; exit 1; }
    master_alive || { err "SSH master is down — re-run './cluster_dev.sh start' (needs Duo)."; exit 1; }
    DD_MODE="$DELTA_ATTACH_MODE"
    if [ "$DD_MODE" = "auto" ]; then
        [ "$(state_get SSH_NODE_OK)" = "yes" ] && [ -n "$DD_NODE" ] && DD_MODE="ssh" || DD_MODE="srun"
    fi
}

cmd_attach() {  # interactive shell on the compute node
    [ "${1:-}" = "--ssh" ] && { DELTA_ATTACH_MODE="ssh"; shift; }
    [ "${1:-}" = "--srun" ] && { DELTA_ATTACH_MODE="srun"; shift; }
    require_running
    if [ "$DD_MODE" = "ssh" ]; then
        log "ssh → ${DD_NODE} (direct, proxied via master). Ctrl-D leaves node; job keeps running."
        ssh "${SSH_OPTS[@]}" -t -J "$CLUSTER_LOGIN" "${DELTA_USER}@${DD_NODE}" "${@:-bash -l}"
    else
        log "srun --overlap onto job ${DD_JOBID}'s node (auth-safe). Ctrl-D leaves; job keeps running."
        ssh "${SSH_OPTS[@]}" -t "$CLUSTER_LOGIN" \
            "srun --jobid=${DD_JOBID} --overlap --gres=${DELTA_SRUN_GRES} --pty bash -l"
    fi
}

cmd_exec() {  # cluster_dev.sh exec -- <command...>  : run inside the Apptainer container on the node
    [ "${1:-}" = "--" ] && shift
    require_running
    local nodecmd="bash ${REMOTE_ISAACLAB_DIR}/docker/cluster/cluster_dev/node_exec.sh $*"
    log "[${DD_MODE}] container exec on job ${DD_JOBID}: $*"
    if [ "$DD_MODE" = "ssh" ]; then
        ssh "${SSH_OPTS[@]}" -J "$CLUSTER_LOGIN" "${DELTA_USER}@${DD_NODE}" "$nodecmd"
    else
        ssh "${SSH_OPTS[@]}" "$CLUSTER_LOGIN" \
            "srun --jobid=${DD_JOBID} --overlap --gres=${DELTA_SRUN_GRES} bash -lc '${nodecmd}'"
    fi
}

cmd_sync() {  # re-mirror local code → Delta isaaclab dir (and onto the live node workspace)
    ensure_master
    stage_node_exec
    rsync_code
    log "Synced to ${REMOTE_ISAACLAB_DIR}."
}

cmd_stop() {
    local jobid; jobid="$(state_get JOBID)"
    [ -f "$WATCH_PID" ] && kill "$(cat "$WATCH_PID")" 2>/dev/null || true; rm -f "$WATCH_PID"
    if [ -n "$jobid" ] && master_alive; then
        log "Cancelling job $jobid…"; on_login "scancel $jobid" || true
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
    sync)     shift; cmd_sync "$@" ;;
    stop)     shift; cmd_stop "$@" ;;
    __watch)  shift; cmd_watch_loop "$@" ;;   # internal (used by nohup)
    ""|-h|--help|help) usage ;;
    *) err "Unknown command '${1}'."; usage; exit 1 ;;
esac
