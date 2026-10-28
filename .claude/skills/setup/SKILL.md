---
name: setup
description: First-time install of this workspace on a machine (or repair of a broken ilab venv) — prerequisites, project selection, `pls setup`, the post-install checks, the known failure modes, then an optional offer to add remote compute. Use when someone has just cloned the repo, `pls` is missing, or `ilab` imports fail.
---

# Setup

Get a working `ilab` venv and composed `resources/`, prove it works, then tell the user what compute they can add.
Ask before anything that installs system packages or needs sudo; everything else here is user-level.

## 1. Check the machine

- OS: Linux x86_64 or aarch64. `nvidia-smi` shows a GPU and a driver new enough for CUDA 12.8 (driver >= 570).
  No GPU is fine for browsing and editing (pick Isaac Lab mode `none`); training then runs on remote compute.
- `sh`, `curl` and `git` on PATH, and `$HOME/.local/bin` on PATH (uv installs there).
- Disk: ~25 GB free for the venv and the Isaac Lab clone, more for Isaac Sim (Kit RTX renders, ~20 GB). uv
  hardlinks from `~/.cache/uv`, so a rebuild does not free the old venv's space until `uv cache clean`.
- Credentials the user will need (ask, don't create): GitHub access to the workspace repos, a W&B account (API key)
  for logged runs, and access to any private data repos named in the task repos' `dependencies.yaml`.

## 2. Install

```bash
./bootstrap.sh          # fresh machine: installs uv if missing, then runs `pls setup`
pls setup               # later: reinstall or reconfigure (from the ilab venv)
```

`pls setup` opens a picker (arrow keys, space): **which projects** to compose and the **Isaac Lab mode**. Choose
`source` (Isaac Lab 3 has no pip release) or `none` (fetch repos only, no venv packages). On a non-interactive
shell it keeps the existing `workspace.yaml`; to write one by hand:

```yaml
projects: [hhlm]          # names from available_projects in workspace.defaults.yaml
isaaclab: {mode: source}
```

It then fetches every repo under `resources/` (`pls resolve`), builds `ilab` on Python 3.12, `uv sync`s the
locked Isaac Lab clone (which owns torch) and installs every workspace package editable. W&B credentials go in
`scripts/.env.wandb` (from `scripts/tools/.env.wandb.template`); `pls deps` asks for them.

## 3. Prove it works

```bash
source ~/.bashrc && ilab        # the alias pls setup added: activate ilab, cd to the manager
python -c "import torch, isaaclab; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
pls test hcrl_isaaclab          # CPU smoke; boots Isaac Sim, so it needs the GPU
pls run train --task <id> --num_envs 64 --max_iterations 2 --logger tensorboard --video off
```

Done when the arch list covers the GPU (e.g. `sm_120` for an RTX 50xx), the smoke passes, and the train log shows
`Learning iteration 1/2`. Exit code 0 alone proves nothing.

## 4. Known failures

| symptom | cause and fix |
| ------- | ------------- |
| `pls: command not found` | the venv isn't active: `ilab`, or `./bootstrap.sh deps` to add `pls` to an existing venv |
| `[install][ERROR] Isaac Lab 3 has no pip release` | `workspace.yaml` says `mode: pip`: set `source`, rerun `pls setup` |
| `needs a Python 3.12 venv` | an old 3.11 `ilab`: `pls clean && pls setup` (note any custom `workspace.yaml` entries first; clean deletes it) |
| `no kernel image is available` / sm_* missing | a torch build without your GPU's arch: never install torch by hand; `pls setup` restores the lock's |
| `.pt`/`.usd` files of ~130 bytes, unpickling errors | Git LFS pointers: `git lfs install`, then `pls resolve` (or `git lfs pull` in that repo) |
| Isaac Sim import exits on aarch64 | needs `LD_PRELOAD` of the system and torch libgomp (`pls run` sets it on SLURM) |
| `Failed to get DOF velocities from backend` | the GPU is still full from an earlier Isaac process: kill it and confirm the memory drained |
| `WARNING <repo>: no access to <url>; tasks that load it stop at load` | an `optional: true` data dep (`dependencies.yaml`) you can't read: setup carries on without it. Ask its owner for access (then `pls resolve`), use the copy the warning names, or skip the tasks that need it |
| a resolve fails on a required repo | no access to a code repo: ask its owner, or drop the project that needs it |

## 5. Offer compute (once, briefly)

Close with one short note, not a push: training runs on this machine by default, and `pls` can also use SSH
GPU boxes, SLURM clusters and Ray clusters the user has access to. Ask whether they want to add any now; if yes,
use the `add-compute` skill, otherwise stop. Never add compute the user didn't ask for.
