# Quickstart

## Prerequisites

1. Ensure that `$HOME/.local/bin` is on your `PATH`.

2. Install [just](https://just.systems/man/en/introduction.html), e.g. as a uv tool:
    ```bash
    uv tool install rust-just
    ```

## Usage

All commands below add bash aliases `ilab` and `manager` to your RC file of choice (can be configured in [justfile](justfile)).
- `manager`: enters the hcrl_isaac_manager directory and activates the manager venv.
- `ilab`: enters the hcrl_isaaclab extension directory and activates the Isaac Lab venv (if it exists).

### Local Installation

To install set up the repo for running Isaac Lab locally, use the command
```bash
just setup
```

This sets up the manager and Isaac Lab virtual environments.

### Cluster Deployment

If you are not planning on running Isaac Lab locally (e.g. if your machine does not meet [system requirements](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#system-requirements)), you can set up a minimal environment for deploying to either ray or HPC clusters.

#### Ray

To configure the environment for Ray cluster deployment, run
```bash
just ray
```
This sets up the manager and creates the appropriate Ray configuration files. You can then send jobs with
```bash
scripts/ray.sh job <train args here>
```

See the [Ray README](scripts/ray/README.md) for more details on the Ray interface.

#### HPC

To add an HPC cluster configuration, run
```bash
just add-cluster
```
This will create a folder `scripts/cluster/<name>_config` with the appropriate config files. You can then build and push a .sif image to the cluster with
```bash
just cluster
```
You only need to do this once, or whenever your dependencies change. You can then deploy a run to the cluster with
```bash
scripts/cluster.sh job <train args here>
```
See the [Cluster README](scripts/cluster/README.md) for more details.

# Asynchronous Video Logging

Environments that do not require cameras during training **can** be deployed to GPU clusters without RT cores, e.g. A100s and H100s. These runs will not support video recording synchronously during training. We instead provide a utility script for asynchronously logging training videos to W&B from any local machine that meets Isaac Sim's [GPU requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html#system-requirements).

To asynchronously log videos to W&B, start the listener with
```bash
scripts/video_listener.sh add --task <task_name>
```

Remove the listener with
```bash
scripts/video_listener.sh remove --task <task_name>
```

This sets up a cron job to run the script `hcrl_isaaclab/scripts/utils/log_videos_async.sh` every 30 minutes. Asynchronous videos for a project can also be created manually using the `hcrl_isaaclab/scripts/log_videos_async.py` script.

More script options can be viewed with `scripts/video_listener.sh --help`. When sending a job to the cluster, be sure to use the `--video` flag to enable async video logging.

# Justfile Targets

The project uses [just](https://just.systems/man/en/introduction.html) to manage packages and other utilities. Environment dependencies are managed using the **uv** package manager.

## Available Targets

### `deps`

- Installs system packages and tools (uv, gitman)
- Sets up manager uv environment
- Pulls Isaac Lab and extension subrepos as specified in `gitman.yml`
- Creates wandb env file, if necessary
- Adds bash aliases to RC file, if necessary

> **Note**: This is called internally by `setup`, `cluster`, and `ray` targets.

### `setup`

- Installs general dependencies (`just deps`)
- Installs Isaac Lab packages and sets up local uv environment

### `clean`

- Deletes Isaac Lab environment
- Removes bash aliases from RC file

### `docker`

- Installs and configures Docker if necessary
- Builds and starts the Isaac Lab Docker container

### `cluster [name]`

- Installs general dependencies (`just deps`)
- Installs nvidia-container-toolkit and Apptainer, if necessary
- Builds and starts the Isaac Lab Docker container (`just docker`)
- Builds and pushes Apptainer image to cluster

### `add-cluster`

- Creates cluster configuration files from template

### `ray`

- Installs general dependencies (`just deps`)
- Creates Ray configuration files from template

### `upload-artifacts [args]`

- Uploads managed large-file resources (robot assets, motion datasets, exported policies) to W&B as versioned artifacts
- `just upload-artifacts --list` shows the registry + local presence; `--all` uploads everything; or pass specific resource keys
- Dedups unchanged content by hash (cheap to re-run); reads W&B credentials from `scripts/.env.wandb`
- These artifacts are fetched back at runtime by the in-script resolver — see [Large-file resources](scripts/ray/README.md#large-file-resources)
