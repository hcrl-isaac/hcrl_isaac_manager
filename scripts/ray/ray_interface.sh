#!/usr/bin/env bash

#==
# Configurations
#==

set -e

tabs 4

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Prefer the ilab venv's python/ray so commands work without activating it first.
VENV_BIN="$( cd "$SCRIPT_DIR/../.." && pwd )/ilab/bin"
[ -d "$VENV_BIN" ] && export PATH="$VENV_BIN:$PATH"

#==
# Functions
#==

check_docker_version() {
    if ! command -v docker &> /dev/null; then
        echo "[Error] Docker is not installed! Please check the 'Docker Guide' for instruction." >&2;
        exit 1
    fi
}

# Sync managed large-file resources to W&B so the cluster fetches current versions; failure is non-fatal.
sync_resources() {
    local mgr up
    mgr="$( cd "$SCRIPT_DIR/../.." && pwd )"
    up="$mgr/resources/hcrl_isaaclab/scripts/tools/upload_artifacts.py"
    if [ ! -f "$mgr/scripts/.env.wandb" ] || [ ! -f "$up" ]; then
        return 0
    fi
    echo "[INFO] Syncing managed resources to W&B (refresh)..."
    ( set -a; . "$mgr/scripts/.env.wandb"; set +a; python "$up" ) || echo "[WARN] resource sync failed; submitting with existing artifacts."
}

# Render the job configs from their templates. WT=<name> (or HCRL_WT) routes ext_dir + file mounts
# through resources/<repo>/worktrees/<name> wherever one exists.
render_job_configs() {  # render_job_configs <ut_eid>
    local ut_eid="$1" manager_dir venv_py
    manager_dir="$( cd "$SCRIPT_DIR/../.." && pwd )"
    venv_py="$manager_dir/ilab/bin/python"
    [ -x "$venv_py" ] || venv_py="python3"
    export HCRL_WT="${HCRL_WT:-${WT:-}}"
    export WORKSPACE_FILE_MOUNTS="$("$venv_py" "$SCRIPT_DIR/build_file_mounts.py")"
    WORKSPACE_EXT_DIR="$manager_dir/resources/hcrl_isaaclab/scripts"
    if [ -n "$HCRL_WT" ] && [ -d "$manager_dir/resources/hcrl_isaaclab/worktrees/$HCRL_WT" ]; then
        WORKSPACE_EXT_DIR="$manager_dir/resources/hcrl_isaaclab/worktrees/$HCRL_WT/scripts"
    fi
    [ -n "$HCRL_WT" ] && echo "[INFO] Worktree set '$HCRL_WT': ext_dir + mounts via resources/<repo>/worktrees/$HCRL_WT" >&2
    export WORKSPACE_EXT_DIR
    local cfg
    for cfg in job_config bench_job_config job_config_distributed; do
        UT_EID="$ut_eid" MANAGER_DIR="$manager_dir" \
            envsubst '$UT_EID $MANAGER_DIR $WORKSPACE_FILE_MOUNTS $WORKSPACE_EXT_DIR' \
            < "$SCRIPT_DIR/tools/$cfg.template.yaml" > "$SCRIPT_DIR/$cfg.yaml"
    done
}

# Re-render the job configs before a submit so the mounts match this job's WT, not the one setup saw.
prepare_submit() {
    if [ ! -f "$SCRIPT_DIR/.env.ray" ]; then
        echo "[ERROR] $SCRIPT_DIR/.env.ray not found. Run 'just ray setup' first." >&2
        exit 1
    fi
    local ut_eid
    ut_eid="$( . "$SCRIPT_DIR/.env.ray"; echo "$UT_EID" )"
    render_job_configs "$ut_eid"
}

#==
# Main
#==

help() {
    echo -e "\nusage: $(basename "$0") [-h] <command> [<job_args>...] -- Utility for interfacing between IsaacLab and Ray clusters."
    echo -e "\noptions:"
    echo -e "  -h              Display this help message."
    echo -e "\ncommands:"
    echo -e "  setup                                Generate the Ray config files (.env.ray + job configs)."
    echo -e "  push                                 Build + push the shared Isaac image to Docker Hub (pulled by the cluster on next startup)."
    echo -e "  job [<job_args>]                     Submit a job to the cluster."
    echo -e "  bench [<job_args>]                   Submit an FPS-benchmark job (sweeps num_envs)."
    echo -e "  stop [<run_id>] [<script_args>]      Stop a currently running job."
    echo -e "  list [<script_args>]                 View existing jobs on the cluster."
    echo -e "  logs [<run_id>] [<script_file>]      Print logs from a run."
    echo -e "\nwhere:"
    echo -e "  <job_args> are optional arguments specific to the job command."
    echo -e "  <script_args> are the per-script arguments (see Ray documentation and list_jobs.py)."
    echo -e "\n" >&2
}

while getopts ":h" opt; do
    case ${opt} in
        h )
            help
            exit 0
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            help
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

if [ $# -lt 1 ]; then
    echo "Error: Command is required." >&2
    help
    exit 1
fi

command=$1
shift

# Every job-submitting subcommand belongs in this list: it re-renders the job configs and syncs resources.
case "$command" in
    job|job_distributed|run|bench)
        prepare_submit; sync_resources
        # every dir the job leaves to W&B artifacts must have one, or the job would only fail on the cluster
        venv_py="$( cd "$SCRIPT_DIR/../.." && pwd )/ilab/bin/python"
        [ -x "$venv_py" ] || venv_py="python3"
        ( mgr="$( cd "$SCRIPT_DIR/../.." && pwd )"; set -a; [ -f "$mgr/scripts/.env.wandb" ] && . "$mgr/scripts/.env.wandb"
          set +a; "$venv_py" "$SCRIPT_DIR/preflight.py" ) || exit 1
        ;;
esac

case $command in
    setup)
        MANAGER_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
        if [ ! -f "$MANAGER_DIR/scripts/.env.wandb" ]; then
            echo "[ERROR] $MANAGER_DIR/scripts/.env.wandb not found. Run 'just deps' first." >&2
            exit 1
        fi
        read -p "UT EID: " ut_eid
        source "$MANAGER_DIR/scripts/.env.wandb"
        UT_EID=$ut_eid envsubst < "$SCRIPT_DIR/tools/.env.ray.template" > "$SCRIPT_DIR/.env.ray"
        render_job_configs "$ut_eid"
        echo "[INFO] Created Ray config files in $SCRIPT_DIR (.env.ray + job_config/bench_job_config/job_config_distributed .yaml)."
        ;;
    push)
        if [ $# -gt 1 ]; then
            echo "Error: Too many arguments for push command." >&2
            help
            exit 1
        fi
        echo "Building and pushing the shared Isaac image for Ray"
        check_docker_version
        "$SCRIPT_DIR/../docker/docker_interface.sh" build
        docker tag hcrl-isaac:latest esturman/isaac-ray:latest
        docker push esturman/isaac-ray:latest
        ;;
    job)
        job_args=("$@")
        echo "[INFO] Executing job command"
        [ ${#job_args[@]} -gt 0 ] && echo -e "\tJob arguments: ${job_args[*]}"
        job_config=$SCRIPT_DIR/job_config.yaml
        echo "[INFO] Executing job script..."
        RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1 python $SCRIPT_DIR/submit_job.py \
            --config_file $SCRIPT_DIR/ray.cfg \
            --job_config $job_config \
            --aggregate_jobs ray/wrap_resources.py \
                --gpu_per_worker 1 \
                "${job_args[@]}"
        ;;
    run)
        # A one-off script (eval, render, census) instead of train.py: `run <repo>/<path>.py [args]`, shipped and
        # queued like a job (Ray holds it until a GPU frees), WT=<name> routing the worktree set as for `job`.
        script="${1:-}"; shift || true
        worker="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); from job_args import ext_script
print(ext_script(sys.argv[2]))' "$SCRIPT_DIR" "$script")" || exit 1
        repo="${script%%/*}"; rest="${script#*/}"
        mgr="$( cd "$SCRIPT_DIR/../.." && pwd )"
        local_copy="$mgr/resources/$repo/$rest"
        wt="${HCRL_WT:-${WT:-}}"  # as render_job_configs reads it
        [ -n "$wt" ] && [ -d "$mgr/resources/$repo/worktrees/$wt" ] && local_copy="$mgr/resources/$repo/worktrees/$wt/$rest"
        [ -f "$local_copy" ] || { echo "[ERROR] no $local_copy to ship" >&2; exit 1; }
        echo "[INFO] Running $script on the Ray cluster ($worker)"
        RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1 python $SCRIPT_DIR/submit_job.py \
            --config_file $SCRIPT_DIR/ray.cfg \
            --job_config $SCRIPT_DIR/job_config.yaml \
            --python_script "$worker" \
            --aggregate_jobs ray/wrap_resources.py \
                --gpu_per_worker 1 \
                "$@"
        ;;
    job_distributed)
        # One Ray submission spawning a sub-job per GPU node (torchrun_wrapper.py -> train.py --distributed).
        job_args=("$@")
        echo "[INFO] Executing distributed job command"
        [ ${#job_args[@]} -gt 0 ] && echo -e "\tJob arguments: ${job_args[*]}"
        job_config=$SCRIPT_DIR/job_config_distributed.yaml
        echo "[INFO] Executing distributed job script..."
        RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1 python $SCRIPT_DIR/submit_job.py \
            --config_file $SCRIPT_DIR/ray.cfg \
            --job_config $job_config \
            --aggregate_jobs ray/wrap_resources.py \
                --gpu_per_worker 1 \
                "${job_args[@]}"
        ;;
    bench)
        job_args=("$@")
        echo "[INFO] Executing bench command"
        [ ${#job_args[@]} -gt 0 ] && echo -e "\tBench arguments: ${job_args[*]}"
        job_config=$SCRIPT_DIR/bench_job_config.yaml
        echo "[INFO] Executing bench script..."
        RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1 python $SCRIPT_DIR/submit_job.py \
            --config_file $SCRIPT_DIR/ray.cfg \
            --job_config $job_config \
            --aggregate_jobs ray/wrap_resources.py \
                --gpu_per_worker 1 \
                "${job_args[@]}"
        ;;
    stop)
        job_id=$1
        shift
        stop_args="$@"
        source $SCRIPT_DIR/.env.ray
        if python $SCRIPT_DIR/list_jobs.py --user_id $UT_EID --check_id $job_id; then
            ray job stop --address http://100.95.64.90:8265 $job_id $stop_args
        else
            echo "[ERROR] The specified job $job_id cannot be stopped."
            echo "[ERROR] Only running jobs started by you can be cancelled."
            echo "[ERROR] You can view these jobs with \`scripts/ray.sh list\`." 
            exit 1
        fi
        ;;
    list)
        list_args="$@"
        source $SCRIPT_DIR/.env.ray
        python $SCRIPT_DIR/list_jobs.py --user_id $UT_EID $list_args
        ;;
    logs)
        job_id=$1
        shift 1
        logs_args="$@"
        source $SCRIPT_DIR/.env.ray
        if python $SCRIPT_DIR/list_jobs.py --user_id $UT_EID --all_statuses --check_id $job_id; then
            ray job logs --address http://100.95.64.90:8265 $job_id $logs_args
        else
            echo "[ERROR] The specified job $job_id cannot be stopped."
            echo "[ERROR] You may only view the logs of jobs started by you."
            echo "[ERROR] You can view these jobs with \`scripts/ray.sh list --all_statuses\`." 
            exit 1
        fi
        ;;
    *)
        echo "Error: Invalid command: $command" >&2
        help
        exit 1
        ;;
esac
