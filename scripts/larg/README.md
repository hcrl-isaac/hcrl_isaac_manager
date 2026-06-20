# LARG GPU Boxes

Helpers for training on the UT LARG lab workstations — bare-metal multi-GPU boxes (no SLURM, no containers)
reachable directly over SSH. Training runs in a per-box `ilab` uv venv that mirrors the local one.

| Class | Hosts |
| --- | --- |
| A100 | `mckennie`, `hazard`, `debruyne`, `aaronson` |
| A40 | `pepi`, `pulisic`, `salah`, `pogba` |

All scripts live in `scripts/larg/` and share config — host list, SSH helpers, and remote paths — via
`common.sh`. A host may be given as a short name (`mckennie`) or a full SSH target. Override defaults with
env vars: `LARG_USER`, `LARG_DOMAIN`, `LARG_REMOTE_DIR` (remote path under `$HOME`), `LARG_LOCAL_DIR`,
`LARG_SCRATCH` (per-box scratch for the venv + uv cache; the NFS home is quota-limited).

## One-time setup

Rsync the tree to a box, then build the `ilab` venv + install Isaac Lab in the background (the long install
survives the SSH session):

```bash
scripts/larg/deploy.sh <host> [<host> ...]
scripts/larg/deploy.sh --log <host>          # poll the setup
```

`deploy.sh` calls `sync.sh` then runs `remote_setup.sh` on the box. The venv and uv cache are placed on
per-box scratch (`/var/local/$USER` by default), not the quota-limited NFS home.

## Sync code

Push later code changes (no rebuild) to one or more boxes:

```bash
scripts/larg/sync.sh <host> [<host> ...]
```

Excludes venvs, datasets, logs, docker images, and other large/rebuilt artifacts (see the `EXCLUDES` list in
`sync.sh`).

## Launch a training run

Single-node, multi-GPU `torchrun` job under `nohup`:

```bash
scripts/larg/train.sh <host> <task> <run_name> [run_group] [num_envs] [-- extra train.py args]
scripts/larg/train.sh --log <host> <task>    # tail the run log + GPU usage
```

Runs are sent with `--server`, so they tag the W&B run for **async video logging** (see below) rather than
rendering in-process. `run_group` defaults to `larg`; `num_envs` is optional.

| Env var | Purpose |
| --- | --- |
| `LARG_NPROC` | GPUs for the run (`torchrun --nproc_per_node`); default 4. |
| `CUDA_VISIBLE_DEVICES` | Pin the run to specific physical GPUs so several runs can share one box (also tags the log filename). |

Example — a 2-GPU run on physical GPUs 0,1 of `pepi`:

```bash
LARG_NPROC=2 CUDA_VISIBLE_DEVICES=0,1 scripts/larg/train.sh pepi <task> my-run my-group 8192
```

## Other helpers

- **`scripts/larg/bench.sh <host> <task> [-- extra]`** — single-GPU `num_envs` FPS sweep (`bench.py`) to pick
  the best per-GPU env count; `--log <host> <task>` to poll.
- **`python scripts/larg/pull_gpu_stats.py`** — print GPU utilization/memory across all LARG boxes. Use it to
  find free GPUs before launching, and be a good citizen on shared boxes.
- **`scripts/larg/video_logger.sh [--loop [secs]] <task> [<entity>/<project>]`** — run the async video logger
  for LARG runs on a *local* RT-core-capable box (one pass, or repeat every `secs`, default 1800). See
  [Asynchronous Video Logging](../../README.md#asynchronous-video-logging) for the full workflow and
  `video_logger.py --mode async` options.
