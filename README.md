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

#### LARG GPU Boxes

The UT LARG lab workstations are bare-metal multi-GPU boxes (no SLURM, no containers) reachable directly over SSH. Sync code and deploy to these clusters with:
```bash
scripts/larg/deploy.sh <host>                                      # one-time: rsync + build the ilab venv
scripts/larg/train.sh <host> <task> <run_name> [run_group] [num_envs]
```
See the [LARG README](scripts/larg/README.md) for the host list, all scripts, and env-var options.

## W&B Workspace Setup

We add two metrics to the W&B workspace that can be used as the x-axis in line graphs and media (set them in the workspace
settings, in the top left).)

- `local_step` is the training iteration, similar to the default `_step` metric. W&B forces `_step` to be monotonically
increasing, which means out-of-process/asynchronously logged videos cannot be logged at the step they are being recorded
for. `local_step` is a workaround for this, and we recommend using it instead of `_step`.
- `env_step` is the total number of environment steps. It is computed as `local_step * num_envs * num_steps_per_env`.
This can be useful for comparing training runs that use a different number of environments.

# Asynchronous Video Logging

Environments that do not require cameras during training **can** be deployed to GPU clusters without RT cores, e.g. A100s and H100s. These runs will not support video recording synchronously during training. Instead, the trainer logs rollout state to W&B and tags the run; a separate **async video logger** running on any local machine that meets Isaac Sim's [GPU requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html#system-requirements) then discovers the tagged run, pulls its checkpoints, renders the videos, and uploads them back to the same W&B run.

### 1. Tag the training run

When sending a cluster job, pass **both** `--video` and `--server` to `train.py`. `--server` tells the trainer it is on a headless (no-RT) box, so instead of spawning a local recorder it tags the W&B run with `log_videos_async` for the async logger to pick up (`--video` alone, on an RT-capable box, records in-process instead).

### 2. Run the async video logger (on an RT-capable device)

The unified logger is `hcrl_isaaclab/scripts/video_logger.py`; run it from the `ilab` venv.

```bash
python video_logger.py --mode async --task <task_name> --wandb_project <entity>/<project> [options]
```

Async mode scans `--wandb_project` for `log_videos_async`-tagged runs and records any checkpoints they haven't logged yet. Useful options:

| Arg | Default | Purpose |
| --- | --- | --- |
| `--task <id>` | -- | IsaacLab env id to rebuild for rendering (required). |
| `--wandb_project <entity>/<project>` | -- | Project to scan for tagged runs (required in async mode). |
| `--num_envs <N>` | 64 | Env count for the small render sim. |
| `--video_length <steps>` | 400 | Clip length in env steps (capped to one episode). |
| `--train_effective_envs <N>` | -- | Training run's `num_envs * world_size`, so the curriculum clock is restored correctly in the render sim (otherwise the curriculum looks fully ramped). |
| `--rerecord_from <iter>` / `--rerecord_all` | -- | Re-record checkpoints ≥ `iter` (or all), uploading alongside the existing videos. |
| `--max_runs_per_sweep <N>` | 0 (no limit) | Record `N` source runs then exit. Use `1` under an outer keeper loop so each process renders one run with a fresh sim. |
| `--stochastic` | off | Record the policy **sampling** actions (training-time exploration noise) instead of the deterministic mean. |

For a long-lived logger, wrap the `--max_runs_per_sweep 1` invocation in a loop that restarts it each pass and prevents the sim from hanging.

### Cron listener

To run the async logger automatically, register a cron listener (every 30 min):

```bash
scripts/video_listener.sh add --task <task_name> --wandb_project <entity>/<project>
```

Remove it with:

```bash
scripts/video_listener.sh remove --task <task_name> --wandb_project <entity>/<project>
```

This installs a cron job that runs `hcrl_isaaclab/scripts/utils/log_videos_async.sh`, which activates the `ilab` venv, sources your W&B credentials, skips the pass if GPU 0 is >50% busy, and invokes `video_logger.py --mode async`. See `scripts/video_listener.sh --help` for all options.

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
