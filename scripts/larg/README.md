# SSH GPU Boxes

Helpers for training on ssh GPU boxes: bare-metal multi-GPU workstations (no SLURM, no containers) reachable directly
over SSH with a key, such as a lab's shared machines. The scripts live in `scripts/larg/` (named for the lab whose boxes
they were first written for) and work on any such box. Training runs in a per-box `ilab` uv venv that mirrors the
local one, or in an Isaac Lab 3 install root (`pls run --root`).

The boxes are your **ssh pools** in the gitignored `scripts/cluster/res/compute.local.toml` (see
[A new ssh box](#a-new-ssh-box)); `pls res` probes them and `common.sh` reads them. A host may be given as a pool's
short name (`box1`, which gets the pool's `user` and `domain`), a full name, or a full `user@host` target. Override
defaults with env vars: `LARG_USER` (your login, when the pools name several or a host is in no pool),
`LARG_REMOTE_DIR` (the workspace on each box; default `/var/local/<login>/hcrl_isaac_manager`, off a quota'd NFS home),
`LARG_LOCAL_DIR`, `LARG_SCRATCH` (per-box scratch for run dirs and Kit caches; default `/var/local/<login>`).

## A new ssh box

1. **Key-based ssh** must work with no prompt: `ssh -o BatchMode=yes <login>@<host> nvidia-smi`. The box needs an
   NVIDIA driver for CUDA 12.8.
2. **Add the pool.** Each pool is a set of boxes sharing a login and domain:

   ```bash
   pls setup --add lab-a100 --kind ssh --hosts box1,box2 --user <login> --domain lab.example.edu
   pls setup --add lab-a40 --kind ssh --hosts box3 --user <login> --domain lab.example.edu --render --pin device
   ```

   `--render` marks RT-core cards (A40, RTX PRO, 5090: `train.sh` records video in-process there); `--pin device`
   makes card runs pick their GPU by device index, for boxes where Kit refuses a masked `CUDA_VISIBLE_DEVICES`;
   `--update` replaces a pool. It writes `scripts/cluster/res/compute.local.toml` (the main checkout's, which every
   worktree reads); editing that file by hand is equivalent:

   ```toml
   [compute.lab-a40]
   kind = "ssh"
   hosts = ["box3"]
   user = "<login>"
   domain = "lab.example.edu"
   render = true
   pin = "device"
   ```

3. **Check it:** `pls res status --pool lab-a40` lists every card as free or busy, not UNKNOWN.
4. **An Isaac Lab 3 install root** for `pls run --on <host>:<gpu> --root <dir>`: copy the minimal trees with
   `scripts/isaac3/root/deploy.sh <login>@<host> <dir>`, then build the root's venv on the box with
   `bash <dir>/resources/isaac3/root/remote_setup.sh <dir>` (`EXTRA_EXTRAS="isaacsim video"` for an RTX render box;
   usage in each script's header). Keep `<dir>` on local disk, e.g. `/var/local/<login>/isaac3`.
5. **Smoke it:** `pls run --on <host>:0 --root <dir> -- train --task <id> --num_envs 64 --max_iterations 2
   --logger tensorboard --video off`.

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

A detached (`setsid nohup`) run from the box's synced workspace: one GPU runs `train.py` directly, `LARG_NPROC` > 1
runs it under `torchrun` with `--distributed`. Each run gets a dir under `$LARG_SCRATCH/larg-runs/` holding its
`train.log` and `TMPDIR`; Kit caches are per GPU set (`$LARG_SCRATCH/kit-cache/gpu<N>`), so concurrent runs on one
box never share one.

```bash
scripts/larg/train.sh <host> <task> <run_name> [run_group] [num_envs] [-- extra train.py args]
scripts/larg/train.sh --log <host> <task>    # tail the run log + GPU usage
```

Runs on a `render = true` pool (RT-core cards, e.g. A40s) record their videos in-process (`--video on`; the Vulkan
clamp layer below lets them render). Other pools' cards (e.g. A100s, no RT cores) pass `--video async`, which tags the W&B run for the async recorder to render on any free
RT card (an A40, an RTX PRO 6000, a 5090). `run_group` defaults to `larg`; `num_envs` is optional.

| Env var | Purpose |
| --- | --- |
| `LARG_NPROC` | GPUs for the run; default 1 (> 1 uses `torchrun --nproc_per_node`). |
| `LARG_SCRATCH` | per-box scratch for run dirs and Kit caches; default `/var/local/<login>`. |
| `CUDA_VISIBLE_DEVICES` | Pin the run to specific physical GPUs so several runs can share one box (also tags the log filename). |

Example — a 2-GPU run on physical GPUs 0,1 of `box3`:

```bash
LARG_NPROC=2 CUDA_VISIBLE_DEVICES=0,1 scripts/larg/train.sh box3 <task> my-run my-group 8192
```

### Feature branches: staged code trees

A run on branches other than what the box's workspace has synced goes through a code tree, as on a cluster
(`pls stage`): named repos at named refs (or local worktrees) land read-only in `<workspace>/trees/<name>-<fingerprint>/`
on that box, every other repo links to the workspace, and the workspace itself is never touched.

```bash
scripts/larg/trees.sh stage box1 kick hhlm_tasks=feat/t1-kick-masked-ball hcrl_isaaclab=feat/train-agent-arg
CUDA_VISIBLE_DEVICES=0 LARG_HOLDER=<you> scripts/larg/train.sh --tree kick box1 hhlm/T1-Kick-v0 kick-s1
scripts/larg/trees.sh list box1
scripts/larg/trees.sh rm box1 kick-<fingerprint>    # refused while a run from it is alive
```

`--tree <name>` runs from the newest complete tree of that name: its staged repos lead `PYTHONPATH` over the
workspace's `ilab` venv, and the tree is marked in use for as long as the run lasts. Stop a tree run with the
`kill -- -<pid>` it prints (its process group). Trees are per box, so stage on the box you launch on.

## Rendering on LARG: the Vulkan clamp layer

Since the 2026-06-12 upgrade every box runs driver **595.71.05**, which reports `maxMemoryAllocationSize` as
UINT64_MAX. Isaac Sim 5.1's `rtx.scenedb` stores a value derived from it in 32 bits, overflows, and the RTX renderer
segfaults at startup (exit 139, backtrace through `createHydraEngine` -> `librtx.scenedb`). Headless training is
unaffected; every `enable_cameras` path died (`--video on`, `play.py --video`, `video_logger.py`).

`scripts/vulkan/` builds an implicit Vulkan layer that clamps the value to what driver 590 reports (4292870144)
and changes nothing on a driver that already reports a sane one. Install it once:

```bash
scripts/larg/install_vkclamp.sh <host>   # one install covers every box that shares this home (an NFS home)
```

It lands in `~/.local/share/vkclamp` with its manifest in `~/.config/vulkan/implicit_layer.d/`, so every Isaac run
loads it with no launcher change; stderr shows `[vkclamp] ... max size 18446744073709551615 -> 4292870144`.
`VKCLAMP_DISABLE=1` turns it off for one run. Verified on pulisic (A40): the camera boot goes from exit 139 to a
clean render. The A100s have no RT cores, so render on the A40s.

## Other helpers

- **`scripts/larg/bench.sh <host> <task> [-- extra]`** — single-GPU `num_envs` FPS sweep (`bench.py`) to pick
  the best per-GPU env count; `--log <host> <task>` to poll.
- **`python scripts/larg/pull_gpu_stats.py`** — print GPU utilization/memory across every host of your ssh pools. Use it to
  find free GPUs before launching, and be a good citizen on shared boxes.
- **Rendering a `--video async` run:** `pls run --on <host>:<gpu> --holder <you> -- video_recorder --mode async
  --wandb_project <entity>/<project> --only_run <id> --task <task>` on any free RT card, e.g. an A40. See [Asynchronous Video Logging](../../README.md#asynchronous-video-logging).
