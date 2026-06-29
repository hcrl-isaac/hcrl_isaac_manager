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

# Cluster selection: CLUSTER=<name> picks the config in config/<name>/ (default: "default").
# Config dirs live under the manager's scripts/cluster, reached relative to this (possibly synced) script.
CLUSTER="${CLUSTER:-default}"
CLUSTER_CONFIG_DIR="${SCRIPT_DIR}/../../../../scripts/cluster"
CLUSTER_ENV_FILE="${CLUSTER_CONFIG_DIR}/config/${CLUSTER}/.env.cluster"
# Source the selected cluster's env (CLUSTER_LOGIN, CLUSTER_ISAACLAB_DIR, CLUSTER_SIF_PATH, ...).
source_cluster_env() {
    if [ ! -f "$CLUSTER_ENV_FILE" ]; then
        echo "[ERROR] Cluster config not found: $CLUSTER_ENV_FILE" >&2
        echo "[ERROR] Set CLUSTER=<name> for a config/<name>/ dir (available:" \
            "$(ls "$CLUSTER_CONFIG_DIR/config" 2>/dev/null | paste -sd, -))." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$CLUSTER_ENV_FILE"
}

# Reuse one authenticated connection for up to 1h -- prevent 2FA reprompting
SSH_CONTROL_DIR="${HOME}/.ssh/cm"
mkdir -p "$SSH_CONTROL_DIR" && chmod 700 "$SSH_CONTROL_DIR"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=${SSH_CONTROL_DIR}/%C -o ControlPersist=1h"

#==
# Functions
#==
# Function to display warnings in red
display_warning() {
    echo -e "\033[31mWARNING: $1\033[0m"
}

# Open the SSH control master. Subsequent ssh/scp/rsync calls reuse the same socket.
ensure_ssh_master() {
    echo "[INFO] Opening SSH control master to $CLUSTER_LOGIN (enter 2FA once)"
    ssh $SSH_OPTS -o ConnectTimeout=60 "$CLUSTER_LOGIN" true
}

# Helper function to compare version numbers
version_gte() {
    # Returns 0 if the first version is greater than or equal to the second, otherwise 1
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" == "$2" ]
}

# Function to check docker versions
check_docker_version() {
    # check if docker is installed
    if ! command -v docker &> /dev/null; then
        echo "[Error] Docker is not installed! Please check the 'Docker Guide' for instruction." >&2;
        exit 1
    fi
    # Retrieve Docker version
    docker_version=$(docker --version | awk '{ print $3 }')
    apptainer_version=$(apptainer --version | awk '{ print $3 }')

    # Check if Docker version is exactly 24.0.7 or Apptainer version is exactly 1.2.5
    if [ "$docker_version" = "24.0.7" ] && [ "$apptainer_version" = "1.2.5" ]; then
        echo "[INFO]: Docker version ${docker_version} and Apptainer version ${apptainer_version} are tested and compatible."

    # Check if Docker version is >= 27.0.0 and Apptainer version is >= 1.3.4
    elif version_gte "$docker_version" "27.0.0" && version_gte "$apptainer_version" "1.3.4"; then
        echo "[INFO]: Docker version ${docker_version} and Apptainer version ${apptainer_version} are tested and compatible."

    # Else, display a warning for non-tested versions
    else
        display_warning "Docker version ${docker_version} and Apptainer version ${apptainer_version} are non-tested versions. There could be issues, please try to update them. More info: https://isaac-sim.github.io/IsaacLab/source/deployment/cluster.html"
    fi
}

# Checks if a docker image exists, otherwise prints warning and exists
check_image_exists() {
    image_name="$1"
    if ! docker image inspect $image_name &> /dev/null; then
        echo "[Error] The '$image_name' image does not exist!" >&2;
        echo "[Error] You might be able to build it with /IsaacLab/docker/container.py." >&2;
        exit 1
    fi
}

# Check if the singularity image exists on the remote host, otherwise print warning and exit
check_singularity_image_exists() {
    image_name="$1"
    if ! ssh $SSH_OPTS "$CLUSTER_LOGIN" "[ -f $CLUSTER_SIF_PATH/$image_name.tar ]"; then
        echo "[Error] The '$image_name' image does not exist on the remote host $CLUSTER_LOGIN!" >&2;
        exit 1
    fi
}

submit_job() {
    echo "[INFO] Arguments passed to job script ${@}"
    case $CLUSTER_JOB_SCHEDULER in
        "SLURM")
            job_script_file=submit_job_slurm.sh
            ;;
        "PBS")
            job_script_file=submit_job_pbs.sh
            ;;
        *)
            echo "[ERROR] Unsupported job scheduler specified: '$CLUSTER_JOB_SCHEDULER'. Supported options are: ['SLURM', 'PBS']"
            exit 1
            ;;
    esac
    ssh $SSH_OPTS $CLUSTER_LOGIN "cd $CLUSTER_ISAACLAB_DIR && bash $CLUSTER_ISAACLAB_DIR/docker/cluster/$job_script_file \"$CLUSTER_ISAACLAB_DIR\" \"isaac-lab-$profile\" ${@}"
}

#==
# Main
#==

help() {
    echo -e "\nusage: $(basename "$0") [-h] <command> [<profile>] [<job_args>...] -- Utility for interfacing between IsaacLab and compute clusters."
    echo -e "\noptions:"
    echo -e "  -h              Display this help message."
    echo -e "\ncommands:"
    echo -e "  push [<profile>]              Push the docker image to the cluster."
    echo -e "  repush [<profile>]            Repush existing SIF image to the cluster."
    echo -e "  job [<profile>] [<job_args>]  Submit a job to the cluster."
    echo -e "  develop [<subcmd>] [<args>]   Manage a persistent dev node (start/status/attach/exec/sync/stop)."
    echo -e "\nwhere:"
    echo -e "  <profile>  is the optional container profile specification. Defaults to 'base'."
    echo -e "  <job_args> are optional arguments specific to the job command."
    echo -e "\nenv:"
    echo -e "  CLUSTER    selects the cluster config in config/<CLUSTER>/ (default: 'default')."
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
profile="base"

case $command in
    push)
        if [ $# -gt 1 ]; then
            echo "Error: Too many arguments for push command." >&2
            help
            exit 1
        fi
        [ $# -eq 1 ] && profile=$1
        echo "Executing push command"
        [ -n "$profile" ] && echo "Using profile: $profile"
        if ! command -v apptainer &> /dev/null; then
            echo "[INFO] Exiting because apptainer was not installed"
            echo "[INFO] You may follow the installation procedure from here: https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages"
            exit
        fi
        # Check if Docker image exists
        check_image_exists isaac-lab-$profile:latest
        # Check docker and apptainer version
        check_docker_version
        # source env file to get cluster login and path information
        source_cluster_env
        # make sure exports directory exists
        mkdir -p /$SCRIPT_DIR/exports
        # clear old exports for selected profile
        rm -rf /$SCRIPT_DIR/exports/isaac-lab-$profile*
        # create singularity image
        cd /$SCRIPT_DIR/exports
        APPTAINER_NOHTTPS=1 apptainer build --sandbox --fakeroot isaac-lab-$profile.sif docker-daemon://isaac-lab-$profile:latest
        # tar image (faster to send single file as opposed to directory with many files)
        tar -cvf /$SCRIPT_DIR/exports/isaac-lab-$profile.tar isaac-lab-$profile.sif
        # open shared SSH connection
        ensure_ssh_master
        # make sure target directory exists
        ssh $SSH_OPTS $CLUSTER_LOGIN "mkdir -p $CLUSTER_SIF_PATH"
        # send image to cluster
        scp $SSH_OPTS $SCRIPT_DIR/exports/isaac-lab-$profile.tar $CLUSTER_LOGIN:$CLUSTER_SIF_PATH/isaac-lab-$profile.tar
        ;;
    repush)
        # source env file to get cluster login and path information
        source_cluster_env
        # open shared SSH connection
        ensure_ssh_master
        # make sure target directory exists
        ssh $SSH_OPTS $CLUSTER_LOGIN "mkdir -p $CLUSTER_SIF_PATH"
        # send image to cluster
        scp $SSH_OPTS $SCRIPT_DIR/exports/isaac-lab-$profile.tar $CLUSTER_LOGIN:$CLUSTER_SIF_PATH/isaac-lab-$profile.tar
        ;;
    job)
        if [ $# -ge 1 ]; then
            passed_profile=$1
            if [ -f ".env.$passed_profile" ]; then
                profile=$passed_profile
                shift
            fi
        fi
        job_args="$@"
        echo "[INFO] Executing job command"
        [ -n "$profile" ] && echo -e "\tUsing profile: $profile"
        [ -n "$job_args" ] && echo -e "\tJob arguments: $job_args"
        source_cluster_env
        # Get current date and time
        current_datetime=$(date +"%Y%m%d_%H%M%S")
        # Append current date and time to CLUSTER_ISAACLAB_DIR
        CLUSTER_ISAACLAB_DIR="${CLUSTER_ISAACLAB_DIR}_${current_datetime}"
        # open shared SSH connection
        ensure_ssh_master
        # Sync Isaac Lab code
        echo "[INFO] Syncing Isaac Lab code..."
        # Keep the source package .git dirs so W&B captures the commit + uncommitted diff
        rsync -rvh -e "ssh $SSH_OPTS" --rsync-path="mkdir -p $CLUSTER_ISAACLAB_DIR && rsync" --include="resources/IsaacLab/source/*/.git/***" --exclude="*.git*" --exclude "ilab/" --exclude "wandb/" --exclude "logs/" --exclude ".vscode/" --exclude "__pycache__" --filter=':- .dockerignore'  /$SCRIPT_DIR/../.. $CLUSTER_LOGIN:$CLUSTER_ISAACLAB_DIR
        # execute job script
        echo "[INFO] Executing job script..."
        # check whether the second argument is a profile or a job argument
        submit_job $job_args
        ;;
    develop)
        # Persistent dev-node management (start/status/attach/exec/sync/stop); cluster-agnostic.
        exec env CLUSTER="$CLUSTER" "$SCRIPT_DIR/cluster_dev/cluster_dev.sh" "$@"
        ;;
    *)
        echo "Error: Invalid command: $command" >&2
        help
        exit 1
        ;;
esac
