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

## Video on no-RT clusters

A100/H100 nodes lack RT cores, so cameras can't run during training. Pass `--video --async` to
`train.py` and the trainer tags the run instead of recording; a separate async logger on any
RT-capable box then renders + uploads from the run's checkpoints:

```bash
scripts/video_listener.sh add --task <task> --wandb_project <entity>/<project>   # 30-min cron, or run once:
ilab && python scripts/video_logger.py --mode async --task <task> --wandb_project <entity>/<project>
```

Full flow + options: [Ray README → Async video logging](scripts/ray/README.md#async-video-logging).

## Justfile targets

`just --list` shows all targets. The common ones: `setup`, `resolve` (re-resolve `workspace.yaml`),
`new-tasks <name>`, `ray`, `add-cluster` / `cluster`, `upload-artifacts`, `docker`, `clean`.
