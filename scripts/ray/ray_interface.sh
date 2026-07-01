#!/usr/bin/env bash

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Prefer the ilab venv's python/ray so commands work without activating it first.
VENV_BIN="$( cd "$SCRIPT_DIR/../.." && pwd )/ilab/bin"
[ -d "$VENV_BIN" ] && export PATH="$VENV_BIN:$PATH"

#==
# Functions
#==

# Function to check docker versions
check_docker_version() {
    # check if docker is installed
    if ! command -v docker &> /dev/null; then
        echo "[Error] Docker is not installed! Please check the 'Docker Guide' for instruction." >&2;
        exit 1
    fi
}

# Sync managed large-file resources to W&B before a job so the cluster fetches the current versions.
# Idempotent (W&B dedups unchanged content); non-fatal so a sync hiccup never blocks a submit.
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

# Parse options
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

# Check for command
if [ $# -lt 1 ]; then
    echo "Error: Command is required." >&2
    help
    exit 1
fi

command=$1
shift

# Any subcommand that submits work to the cluster first syncs managed resources to W&B, so the job
# fetches current asset versions. Add new job-submitting subcommands to this list.
case "$command" in
    job|job_distributed|bench) sync_resources ;;
esac

case $command in
    setup)
        # Generate the Ray config files (.env.ray + job configs) from the templates.
        MANAGER_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
        VENV_PY="$MANAGER_DIR/ilab/bin/python"
        [ -x "$VENV_PY" ] || VENV_PY="python3"
        if [ ! -f "$MANAGER_DIR/scripts/.env.wandb" ]; then
            echo "[ERROR] $MANAGER_DIR/scripts/.env.wandb not found. Run 'just deps' first." >&2
            exit 1
        fi
        read -p "UT EID: " ut_eid
        source "$MANAGER_DIR/scripts/.env.wandb"
        UT_EID=$ut_eid envsubst < "$SCRIPT_DIR/tools/.env.ray.template" > "$SCRIPT_DIR/.env.ray"
        export WORKSPACE_FILE_MOUNTS="$("$VENV_PY" "$SCRIPT_DIR/build_file_mounts.py")"
        for cfg in job_config bench_job_config job_config_distributed; do
            UT_EID=$ut_eid MANAGER_DIR="$MANAGER_DIR" \
                envsubst '$UT_EID $MANAGER_DIR $WORKSPACE_FILE_MOUNTS' \
                < "$SCRIPT_DIR/tools/$cfg.template.yaml" > "$SCRIPT_DIR/$cfg.yaml"
        done
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
        # Build the shared decoupled image (isaacsim base + pip isaaclab) and push it for the cluster to pull.
        "$SCRIPT_DIR/../docker/docker_interface.sh" build
        docker tag hcrl-isaac:latest esturman/isaac-ray:latest
        docker push esturman/isaac-ray:latest
        ;;
    job)
        job_args="$@"
        echo "[INFO] Executing job command"
        [ -n "$job_args" ] && echo -e "\tJob arguments: $job_args"
        job_config=$SCRIPT_DIR/job_config.yaml
        # Submit job
        echo "[INFO] Executing job script..."
        RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1 python $SCRIPT_DIR/submit_job.py \
            --config_file $SCRIPT_DIR/ray.cfg \
            --job_config $job_config \
            --aggregate_jobs ray/wrap_resources.py \
                --gpu_per_worker 1 \
                $job_args
        ;;
    job_distributed)
        # Distributed variant of `job`: one Ray submission spawns two sub-jobs (one per GPU node) that
        # run torchrun_wrapper.py -> train.py --distributed, for a single run with a 2-node global batch.
        job_args="$@"
        echo "[INFO] Executing distributed job command"
        [ -n "$job_args" ] && echo -e "\tJob arguments: $job_args"
        job_config=$SCRIPT_DIR/job_config_distributed.yaml
        echo "[INFO] Executing distributed job script..."
        RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1 python $SCRIPT_DIR/submit_job.py \
            --config_file $SCRIPT_DIR/ray.cfg \
            --job_config $job_config \
            --aggregate_jobs ray/wrap_resources.py \
                --gpu_per_worker 1 \
                $job_args
        ;;
    bench)
        job_args="$@"
        echo "[INFO] Executing bench command"
        [ -n "$job_args" ] && echo -e "\tBench arguments: $job_args"
        job_config=$SCRIPT_DIR/bench_job_config.yaml
        echo "[INFO] Executing bench script..."
        RAY_RUNTIME_ENV_IGNORE_GITIGNORE=1 python $SCRIPT_DIR/submit_job.py \
            --config_file $SCRIPT_DIR/ray.cfg \
            --job_config $job_config \
            --aggregate_jobs ray/wrap_resources.py \
                --gpu_per_worker 1 \
                $job_args
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
