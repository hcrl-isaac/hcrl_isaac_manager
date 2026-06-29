# hcrl_isaac_manager

Composes the hcrl Isaac Lab workspace — core `hcrl_isaaclab`, per-project `*_tasks` / `*_robots`,
and `robot_rl` — and deploys training to local GPUs, Ray, or HPC clusters. This README is the
quickstart; per-cluster READMEs ([Ray](scripts/ray/README.md), [Cluster](scripts/cluster/README.md),
[LARG](scripts/larg/README.md)) cover what each piece does and the full options.

## Prerequisites

- `$HOME/.local/bin` on your `PATH`.
- [just](https://just.systems/man/en/introduction.html): `uv tool install rust-just`.

## Configure the workspace

Pick which projects and IsaacLab mode to include in [`workspace.yaml`](workspace.yaml):

```yaml
projects: [ssti, umrl]   # the *_tasks repos to install (their robot deps are pulled automatically)
isaaclab:
  source: false          # false → IsaacLab from pip; true → clone IsaacLab source under resources/IsaacLab
```

## Install (local)

```bash
just setup
```

Resolves `workspace.yaml` into a flat, deduped set of repos under `resources/`, then builds the
manager and Isaac Lab venvs. Adds two bash aliases:

- `ilab` — the Isaac Lab extension dir + training venv.
- `manager` — this dir + the manager venv.

Scaffold a new project repo with `just new-tasks <name>` (registers under the `<name>/` namespace).

## Run

`--source` selects a project's namespace and is only needed when a task id is shared by two projects.

**Local:**
```bash
ilab
python scripts/train.py --task <task-id> [--source <ssti|umrl>]
```

**Ray:**
```bash
just ray                       # one-time: write Ray config files
just upload-artifacts --all    # one-time / when assets change: push large assets as W&B artifacts
manager
scripts/ray.sh job --task <task-id> [train args]
```
Large files (robot assets, motions, policies) are excluded from the job upload and fetched at runtime
as W&B artifacts, so `upload-artifacts` must run before the first job. See the
[Ray README](scripts/ray/README.md).

**HPC:**
```bash
just add-cluster               # one-time per cluster (CLUSTER=<name>)
just cluster                   # build + push the .sif (when deps change)
scripts/cluster.sh job --task <task-id> [train args]
```
See the [Cluster README](scripts/cluster/README.md).

**LARG GPU boxes:**
```bash
scripts/larg/deploy.sh <host>
scripts/larg/train.sh <host> <task> <run_name> [run_group] [num_envs]
```
See the [LARG README](scripts/larg/README.md).

## W&B custom x-axes

Add `local_step` and `env_step` as workspace metrics (workspace settings, top-left) to use as the
x-axis in line graphs / media:

- `local_step` — training iteration. Use it instead of `_step` (W&B forces `_step` monotonic, so
  out-of-process / async-logged videos can't be logged at the step they were recorded for).
- `env_step` — total env steps (`local_step * num_envs * num_steps_per_env`); compares runs that use
  different env counts.

## Asynchronous Video Logging

Environments that do not require cameras during training **can** be deployed to GPU clusters without
RT cores, e.g. A100s and H100s. These runs will not support video recording synchronously during
training. Instead, the trainer logs rollout state to W&B and tags the run; a separate **async video
logger** running on any local machine that meets Isaac Sim's
[GPU requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html#system-requirements)
then discovers the tagged run, pulls its checkpoints, renders the videos, and uploads them back to
the same W&B run.

### 1. Tag the training run

When sending a cluster job, pass **both** `--video` and `--server` to `train.py`. `--server` tells
the trainer it is on a headless (no-RT) box, so instead of spawning a local recorder it tags the W&B
run with `log_videos_async` for the async logger to pick up (`--video` alone, on an RT-capable box,
records in-process instead).

### 2. Run the async video logger (on an RT-capable device)

The unified logger is `hcrl_isaaclab/scripts/video_logger.py`; run it from the `ilab` venv.

```bash
python scripts/video_logger.py --mode async --task <task_name> --wandb_project <entity>/<project> [options]
```

Async mode scans `--wandb_project` for `log_videos_async`-tagged runs and records any checkpoints
they haven't logged yet. Useful options:

| Arg | Default | Purpose |
| --- | --- | --- |
| `--task <id>` | -- | IsaacLab env id to rebuild for rendering (required). |
| `--wandb_project <entity>/<project>` | -- | Project to scan for tagged runs (required in async mode). |
| `--num_envs <N>` | 64 | Env count for the small render sim. |
| `--video_length <steps>` | 400 | Clip length in env steps (capped to one episode). |
| `--train_effective_envs <N>` | -- | Training run's `num_envs * world_size`, so the curriculum clock is restored correctly in the render sim (otherwise the curriculum looks fully ramped). |
| `--rerecord_from <iter>` / `--rerecord_all` | -- | Re-record checkpoints >= `iter` (or all), uploading alongside the existing videos. |
| `--max_runs_per_sweep <N>` | 0 (no limit) | Record `N` source runs then exit. Use `1` under an outer keeper loop so each process renders one run with a fresh sim. |
| `--stochastic` | off | Record the policy **sampling** actions (training-time exploration noise) instead of the deterministic mean. |

For a long-lived logger, wrap the `--max_runs_per_sweep 1` invocation in a loop that restarts it
each pass and prevents the sim from hanging.

### Cron listener

To run the async logger automatically, register a cron listener (every 30 min):

```bash
scripts/video_listener.sh add --task <task_name> --wandb_project <entity>/<project>
```

Remove it with:

```bash
scripts/video_listener.sh remove --task <task_name> --wandb_project <entity>/<project>
```

This installs a cron job that runs `hcrl_isaaclab/scripts/utils/log_videos_async.sh`, which activates
the `ilab` venv, sources your W&B credentials, skips the pass if GPU 0 is >50% busy, and invokes
`video_logger.py --mode async`. See `scripts/video_listener.sh --help` for all options.

## Justfile Targets

`just --list` shows all targets. The project uses [just](https://just.systems/man/en/introduction.html)
to manage setup and deployment; environment dependencies are managed with the **uv** package manager.

### `deps`

- Installs system tools (uv, gitman) and the manager uv environment
- Creates the W&B env file, if necessary
- Adds the bash aliases (`ilab` / `manager`) to your RC file, if necessary

> **Note**: called internally by `setup`, `cluster`, and `ray`.

### `setup`

- Installs general dependencies (`just deps`)
- Resolves `workspace.yaml` and fetches the workspace repos (`just resolve`)
- Installs Isaac Lab + Isaac Sim and editable-installs every workspace package into the local uv env

### `resolve`

- Resolves `workspace.yaml` into a flat, deduped `gitman.yml`, then fetches all repos as siblings under `resources/`
- Re-run after editing `workspace.yaml` (which projects / IsaacLab mode to compose)

### `new-tasks <name>`

- Scaffolds a new `<name>_tasks` extension repo under `resources/`, registered under the `<name>/` namespace

### `clean`

- Deletes the Isaac Lab environment
- Removes the bash aliases from your RC file

### `docker`

- Installs and configures Docker if necessary
- Builds and starts the Isaac Lab Docker container

### `cluster [name]`

- Installs general dependencies (`just deps`)
- Installs nvidia-container-toolkit and Apptainer, if necessary
- Builds and starts the Isaac Lab Docker container (`just docker`)
- Builds and pushes the Apptainer image to the cluster

### `add-cluster`

- Creates cluster configuration files from template

### `ray`

- Installs general dependencies (`just deps`)
- Creates Ray configuration files from template

### `upload-artifacts [args]`

- Uploads managed large-file resources (robot assets, motion datasets, exported policies) to W&B as versioned artifacts
- `--list` shows the registry + local presence; `--all` uploads everything; or pass specific resource keys
- Dedups unchanged content by hash (cheap to re-run); reads W&B credentials from `scripts/.env.wandb`
- Fetched back at runtime by the in-script resolver — see [Large-file resources](scripts/ray/README.md#large-file-resources)
