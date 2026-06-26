# Using the Ray Servers

## Setup

Make accounts (or use existing accounts):

- [Tailscale](https://tailscale.com/) (use a non-utexas.edu email)
    - Follow the instructions on account creation to install Tailscale on your local machine
    - When prompted to add a second device, skip the tutorial
- [Github](https://github.com/signup)
- [wandb](https://wandb.ai/signup) (select Models under "What do you want to try first?", if prompted)

Add the server machine to your Tailnet (get link from Slack).

## Dependencies

The ray scripts assume that you will be developing on hcrl_isaaclab within manager-installed Isaac Lab.

> Note:
> - All paths below are relative to the `hcrl_isaac_manager` directory.
> - Below assumes you are using `uv` as your package manager. If you use conda, you will need to manually install
> dependencies listed in `pyproject.toml`.
> - Commands below are consistent for Unix-like systems, i.e. Linux and Mac. If running on Windows, some commands may
> be different.

Set up the Ray configuration with
```bash
just ray
```

If you have additional non-standard configurations (e.g. additional file mounts, env variables, etc.), you can edit the config files at `ray/.env.ray` and `ray/job_config.yaml`.

Activate the uv environment. You will need to have the environment activated anytime you want to use the ray commands.

```bash
manager  # or `source .venv/bin/activate`
```

Ensure connectivity to the server with

```bash
scripts/ray.sh list
```

This should display a blank table, like so:
```
                           Ray Jobs                                                                                                                             
┏━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ Job ID ┃ User ┃ Start Time ┃ End Time ┃ Status ┃ Entrypoint ┃
┡━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
└────────┴──────┴────────────┴──────────┴────────┴────────────┘
```

### Upload large assets (first time)

Robot assets, motion datasets, and exported policies are excluded from the per-job upload (they would
blow Ray's size limit) and fetched at runtime as W&B artifacts instead. Push them once before your
first job — and again whenever they change:

```bash
just upload-artifacts --all     # or --list to see the registry, or pass specific keys
```

See [Large-file resources](#large-file-resources) for how the runtime resolver fetches them and how to
register a new file.

## Ray Interface

You can access the Ray dashboard at `http://<server_ip>:8265`. The dashboard is view-only, i.e. you cannot cancel jobs
from this interface.

### `scripts/ray.sh job`

- Sends a job to the server.
    - You can modify the script that runs (e.g. between `train.py` and `play.py`) in the `python_script` field of `job_config.yaml`
- Can be followed by any arguments you'd like to pass to the script (e.g. `--task reach-v0`)

### `scripts/ray.sh stop <job_id>`

- Stop a running job
- Can provide additional arguments (see [`ray job stop` docs](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/cli.html#ray-job-stop))
- You can only stop jobs that have been created by you. If you really want to bypass this, comment out the check in
`scripts/ray/ray_interface.sh`.

### `scripts/ray.sh list`

- List your currently running jobs, ascending by start time
- View all users' runs with `--all_users`
- View the status of all runs with `--all_statuses`

### `scripts/ray.sh logs <job_id>`

- Download and print logs for a job
- Can provide additional arguments (see [`ray job logs` docs](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/cli.html#ray-job-logs))
- You can redirect the output to a file with the `>` operator: `./ray.sh logs <job_id> > <file_path>`
- Generally, you can use W&B to view your run logs and metrics. This function is mainly for when your job fails before it can deploy, or if you aren't using W&B

## How it works

1. Currently, each server node runs a modified Isaac Lab docker container with some additional dependencies (ray, git-lfs,
etc.).

2. When a job is sent, file mounts (configured in `scripts/ray/job_config.yaml` as `"/path/on/local/": "/path/on/server"`)
are copied from your local machine to the server. Dependencies are installed in each mount location with `pip install -e .`.
Large files (robot assets, motion datasets, exported policies) are *excluded* from this upload (`excludes` in
`job_config.yaml`) and fetched at runtime as W&B artifacts instead — see [Large-file resources](#large-file-resources).

3. User-specified initialization commands (`init_commands` in `job_config.yaml`), if any, are then executed.

4. The job script (`python_script` in `job_config.yaml`) is executed with any additional CLI args passed by the user. On
startup it fetches the large-file resources its task references, then runs as usual.

5. The script finishes or fails, after which file mounts are deleted (the artifact downloads persist in a cache).

## Large-file resources

Large files — robot assets (`resources/hcrl_robots`, `resources/ssti_robots`), motion datasets
(`resources/motion_datasets`), and exported policies (`tasks/.../policies/<name>`) — are too big to
upload with each job, so they are versioned as **W&B artifacts** and fetched at runtime by the
in-script resolver (`hcrl_isaaclab/utils/artifacts.py`). This is transparent: the resolver downloads
each artifact and symlinks it into the canonical path the configs already reference, so nothing in
the task configs changes. On a machine where the files already exist (local dev) it is a no-op.

Each task only fetches the artifacts it actually references (auto-detected by scanning the resolved
env cfg), so a pure-locomotion run does not pull the multi-GB motion datasets. Downloads land in
`HCRL_ARTIFACT_ROOT` (bind-mounted to a persistent host dir on the cluster), keyed by artifact
version, so they are fetched once and skipped thereafter; queued jobs pinning different versions
never collide.

### Adding / updating a large file

1. Register it in `REGISTRY` in `hcrl_isaaclab/utils/artifacts.py` (key, artifact type, canonical
   `dest`, the `marker` substring that signals a cfg uses it, and `tier`: `persistent` for static
   bulk, `cache` for LRU-prunable mid-size files).
2. Upload it from its local path. From the manager directory:
   ```bash
   just upload-artifacts <key>   # or --all / --list
   ```
   This sources W&B credentials from `scripts/.env.wandb` and runs the uploader in the `ilab` venv
   (equivalent to `source/hcrl_isaaclab/scripts/tools/upload_artifacts.py <key>`).
   Re-uploading dedupes unchanged content by hash, so it is cheap to re-run when a file changes.

### Cleanup

`HCRL_ARTIFACT_ROOT` grows over time. Bound the `cache`-tier footprint with
`artifacts.cache_cleanup(max_bytes)` (LRU eviction; `persistent` tier is never auto-pruned). Old
artifact *versions* in W&B can be deleted via the API/UI to reclaim quota.

## Async video logging

The Ray nodes (A100/H100) lack RT cores, so cameras can't run during training. To still get videos,
the trainer logs to W&B and tags the run; a separate **async video logger** on any RT-capable box
discovers the tagged run, pulls its checkpoints, renders the clips, and uploads them back to the same
run. (On an RT-capable box, `--video` *alone* records in-process instead.)

### 1. Tag the run

Pass **both** `--video` and `--async` to `train.py`. `--async` tells the trainer it's on a headless
no-RT box, so instead of spawning a local recorder it tags the W&B run `log_videos_async`.

### 2. Run the logger (on an RT-capable device, from the `ilab` venv)

```bash
python video_logger.py --mode async --task <task_name> --wandb_project <entity>/<project> [options]
```

Async mode scans `--wandb_project` for tagged runs and records any checkpoints they haven't logged yet.

| Arg | Default | Purpose |
| --- | --- | --- |
| `--task <id>` | — | env id to rebuild for rendering (required) |
| `--wandb_project <entity>/<project>` | — | project to scan (required) |
| `--num_envs <N>` | 64 | env count for the render sim |
| `--video_length <steps>` | 400 | clip length, capped to one episode |
| `--train_effective_envs <N>` | — | training `num_envs * world_size`, so the curriculum clock is restored (else the curriculum looks fully ramped) |
| `--rerecord_from <iter>` / `--rerecord_all` | — | re-record checkpoints ≥ `iter` (or all), uploaded alongside existing videos |
| `--max_runs_per_sweep <N>` | 0 (no limit) | record `N` runs then exit; use `1` under a keeper loop so each process renders one run with a fresh sim |
| `--stochastic` | off | record the policy **sampling** actions instead of the deterministic mean |

### Cron listener

To run it automatically every 30 min:

```bash
scripts/video_listener.sh add    --task <task_name> --wandb_project <entity>/<project>
scripts/video_listener.sh remove --task <task_name> --wandb_project <entity>/<project>
```

This installs a cron job running `hcrl_isaaclab/scripts/utils/log_videos_async.sh`, which activates
the `ilab` venv, sources W&B credentials, skips the pass if GPU 0 is >50% busy, and invokes
`video_logger.py --mode async`.

## Current Limitations

There are a couple of limitations on the server setup. They may be fixed over time, but if you need one immediately for your research, please let Emily know on Slack.

- **Isaac Lab only** *(plan to fix: no)*. Currently, the server is in an Isaac Lab docker container, so it can *only* run Isaac Lab jobs. If you need more general clusters, we recommend using TACC.
- **Fixed Isaac Lab** *(plan to fix: if demand)*. The runs assume that Isaac Lab itself is unmodified. If you edit Isaac Lab locally, those changes will not be reflected in the runs.
- **Coarse-grained resource allocation** *(plan to fix: yes)*: Currently, resources are allocated as a single GPU node per job (or multiple GPU nodes per job, if required). Some jobs may require fewer resources (e.g. 50% of the gpu, 20% of memory). We may eventually change this so that we can execute more jobs at once.

## Etiquette

Since we only have two server computers, this means that up to two jobs can run at a time. Please be considerate to other users — keep an eye on your jobs, and please cancel them if they aren't getting results. You can see if other users have queued jobs through the Ray dashboard or with `scripts/ray.sh list --all_users`.
