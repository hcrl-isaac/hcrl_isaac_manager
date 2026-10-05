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

Runs are sent with `--video async`, so they tag the W&B run for **async video logging** (see below) rather than
rendering in-process. `run_group` defaults to `larg`; `num_envs` is optional.

| Env var | Purpose |
| --- | --- |
| `LARG_NPROC` | GPUs for the run (`torchrun --nproc_per_node`); default 4. |
| `CUDA_VISIBLE_DEVICES` | Pin the run to specific physical GPUs so several runs can share one box (also tags the log filename). |

Example — a 2-GPU run on physical GPUs 0,1 of `pepi`:

```bash
LARG_NPROC=2 CUDA_VISIBLE_DEVICES=0,1 scripts/larg/train.sh pepi <task> my-run my-group 8192
```

## Rendering on LARG: the Vulkan clamp layer

Since the 2026-06-12 upgrade every box runs driver **595.71.05**, which reports `maxMemoryAllocationSize` as
UINT64_MAX. Isaac Sim 5.1's `rtx.scenedb` stores a value derived from it in 32 bits, overflows, and the RTX renderer
segfaults at startup (exit 139, backtrace through `createHydraEngine` -> `librtx.scenedb`). Headless training is
unaffected; every `enable_cameras` path died (`--video on`, `play.py --video`, `video_logger.py`).

`scripts/vulkan/` builds an implicit Vulkan layer that clamps the value to what driver 590 reports (4292870144)
and changes nothing on a driver that already reports a sane one. Install it once:

```bash
scripts/larg/install_vkclamp.sh <any LARG host>   # the home is one NFS share, so this covers every box
```

It lands in `~/.local/share/vkclamp` with its manifest in `~/.config/vulkan/implicit_layer.d/`, so every Isaac run
loads it with no launcher change; stderr shows `[vkclamp] ... max size 18446744073709551615 -> 4292870144`.
`VKCLAMP_DISABLE=1` turns it off for one run. Verified on pulisic (A40): the camera boot goes from exit 139 to a
clean render. The A100s have no RT cores, so render on the A40s.

## Other helpers

- **`scripts/larg/bench.sh <host> <task> [-- extra]`** — single-GPU `num_envs` FPS sweep (`bench.py`) to pick
  the best per-GPU env count; `--log <host> <task>` to poll.
- **`python scripts/larg/pull_gpu_stats.py`** — print GPU utilization/memory across all LARG boxes. Use it to
  find free GPUs before launching, and be a good citizen on shared boxes.
- **`scripts/larg/video_logger.sh [--loop [secs]] <task> [<entity>/<project>]`** — run the async video logger
  for LARG runs on a *local* box that can render — not a LARG box, see above (one pass, or repeat
  every `secs`, default 1800). See
  [Asynchronous Video Logging](../../README.md#asynchronous-video-logging) for the full workflow and
  `video_logger.py --mode async` options.
