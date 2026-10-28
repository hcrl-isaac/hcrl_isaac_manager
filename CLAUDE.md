# CLAUDE.md — hcrl_isaac_manager

This repo composes an Isaac Lab 3 robot-learning workspace from sibling repos under `resources/` and runs training on
this machine or on any compute you add (SSH boxes, SLURM clusters, Ray). Every workspace command goes through `pls`.
The [README](README.md) is the human quickstart; this file is what a Claude session needs to work here.

Personal or machine-specific instructions (your own compute, how you like to work) go in a gitignored
`CLAUDE.local.md` next to this file, or in `~/.claude/CLAUDE.md`, never here.

## Skills

Project skills live in `.claude/skills/`. Use them rather than re-deriving the steps:

| skill | when |
| ----- | ---- |
| `setup` | first-time install, or a broken `ilab` venv |
| `add-compute` | adding an SSH box, SLURM cluster or Ray cluster, or checking which kinds you have |
| `launch-run` | any training run past a smoke test |
| `new-task` | adding a task (env + runner cfg + registration) |
| `convention-review` | reviewing a branch against the conventions below |

## Layout

- **`<manager>/`** (this repo): `workspace.defaults.yaml` (the project catalog, repo refs, Isaac Lab mode),
  `src/hcrl_cli/` (`pls`), `scripts/` (`cluster/`, `ray/`, `larg/`, `docker/`, `tools/`), `resources/`
  (the composed repos, gitignored).
- **`resources/hcrl_isaaclab/`**: core extension (tasks, assets, mdp, utils, `scripts/train.py` and friends).
- **`resources/robot_rl/`**: RL package (PPO, SAC, FB-CPR runners, models, storage).
- **`resources/<project>_tasks/`**: per-project task repos (e.g. `hhlm_tasks`, `ssti_tasks`).
- **`resources/<name>_robots/`, `hcrl_robots`, `motion_datasets`**: asset and data repos. Configs reach them through
  `hcrl_isaaclab/utils/paths.py:RESOURCES_DIR` (this workspace's `resources/`), never through symlinks.
- **`resources/IsaacLab/`**: the Isaac Lab 3 source clone, at the ref in `workspace.defaults.yaml`.
- Which repos are present: `workspace.defaults.yaml` (`always`, `available_projects`, `refs`) merged with the
  gitignored per-user `workspace.yaml`; `pls resolve` turns that into a flat `gitman.yaml` and fetches every repo.
  A repo without a `refs:` entry tracks `main`.

## Python environment

- One uv venv, **`ilab`**, at `<manager>/ilab` (Python 3.12). `pls setup` builds it: the manager's deps, then a
  `uv sync` of the uv-locked Isaac Lab 3 source clone (which owns torch), then every workspace package, editable.
  Isaac Lab 3 has no pip release, so `workspace.yaml` needs `isaaclab.mode: source`.
- The `ilab` shell alias activates it and cd's to the manager dir.
- Missing dependency? `uv pip install --python ilab/bin/python <pkg>`. Never install another torch over the lock's,
  never `uv sync --active` or `pip install -e` a worktree into `ilab` (that repoints the shared venv), and after
  anything that could touch torch, check `torch.cuda.get_arch_list()` covers your GPU (e.g. `sm_120` for Blackwell).

## Running code

- **Here:** `pls run <script> <args>` from anywhere, e.g. `pls run train --task <id> --num_envs 4096`,
  `pls run play --task <id> --checkpoint <path or W&B run URL>`. A bare name is an `hcrl_isaaclab/scripts/` script;
  `<repo>/<path>.py` runs any repo's script. Options go before `--`: `pls run --on <target> --wt <set> -- train ...`.
  `pls run --help` lists every form.
- **Elsewhere:** `pls run --on <pool | host:gpu | cluster | ray> -- ...`. `pls res status --free` shows which
  cards are free in every pool you have configured; `pls res claim`/`release` hold one.
- **Compute is per user.** The tracked `scripts/cluster/res/compute.toml` holds only this machine. Your SSH pools
  and your Ray dashboard go in the gitignored `compute.local.toml` (`pls setup --add <name> --kind ssh|ray ...`),
  SLURM clusters in `scripts/cluster/config/<name>/` (`pls setup --add <name>`), Ray credentials, job id and image in
  `scripts/ray/.env.ray` (`pls setup --on ray`). The `add-compute` skill walks through each kind.
- **SLURM clusters:** `pls connect --on <c>` opens the shared SSH connection (2FA once). Either queue a job
  (`pls run --on <c> --batch -- ...`) or hold a node (`pls res hold --on <c>`, release with `pls res drop`) and run
  on its cards (`pls run --on <c> -- ...`, `--gpus N` for one run across N cards). `pls jobs --on <c>` lists,
  follows and stops jobs; `pls shell --on <c>` opens a shell on the held node.
- **Runtimes:** an ssh box runs from an Isaac Lab 3 install root (`pls run --root <dir>`); SLURM and Ray run the
  shared image (`scripts/docker/`, as a `.sif` on SLURM), built and pushed per cluster by `pls setup --on <c>`.
  The workspace code is mounted into it at job start, so code changes need no rebuild; dependency changes do.
- **Code reaches remote compute only as staged trees** (`pls stage`, done by `pls run` for you): a snapshot of the
  workspace as on disk, or of named refs/worktrees. Runs must not write tracked files in place (trees share them by
  hardlink). `pls run --cmd -- <command>` starts in that machine's workspace; the staged code is in `$PLS_RESOURCES`.
- **Data ships only if committed.** A dataset or policy a task loads must be committed and pushed to its repo
  (motion data to `motion_datasets`, a Hugging Face dataset repo; exported policies to their task repo, via Git LFS),
  or a remote run cannot see it. A repo's `dependencies.yaml` names where each data repo is fetched from.
- **Worktrees:** `pls run --wt <set> -- ...` runs a set of repo worktrees (`scripts/worktree_env.py`). Keep repo
  worktrees under `resources/<repo>/worktrees/`, and pass absolute paths to `git worktree add`.

## Isaac Lab traps

- **Isaac keeps its VRAM after exit, crash or SIGTERM.** Stop whatever would restart it, SIGKILL it, then confirm
  the card drained. `Failed to get DOF velocities from backend` means the card is still full.
- **Parallel Kit processes each need their own `TMPDIR`, `XDG_CACHE_HOME` and `OMNI_CACHE_DIR`**; staggering the
  starts is not isolation.
- **Exit code 0 is not success.** Only a parsed `Learning iteration N` line proves training began.
- Failure text seen here: `Traceback`, `There was an error running python` (Hydra), `Could not override` (bad
  override key), `CUDA out of memory`, `No contact sensors added`, `Segmentation fault`, `Killed`, `srun: error`.
  Match all of them, not one.
- Isaac Lab 3 quaternions are **xyzw** (2.x was wxyz), and asset data reads go through `.torch`.

## Experiments

- Smoke, probe and repro runs: `--logger tensorboard --video off`. Every other run logs to W&B with video `on`
  (default) or `async`.
- `--run_name` is the change from the previous run (e.g. `resample=1-5s`), never a repeat or a status; `--run_group`
  is the dimension being explored. No one-run groups.
- Multi-GPU is iteration scaling: N ranks means budget/N. `agent.max_iterations` on resume is additive.
- Judge a policy by `Metrics/*`, termination rates, task measures and cumulative reward, never by single reward
  terms, and compare against a do-nothing policy. A metric pinned at a constant while reward rises is a bug.
- Before choosing any clip, clamp, floor or gate threshold, print the quantity's percentiles and put the bound in a gap.
- A launched run that dies looks like a healthy one. Watch every run for progress, failure *and* stall (log mtime
  older than ~15 min): `scripts/tools/watch_run.sh` does this; the `launch-run` skill covers it.

## Task and reward design

- Task ids: `<Robot>-<Task>-v0` is the canonical no-curriculum teacher, `-Depth-v0` the distilled student, `-PPO-v0`
  the PPO variant; CamelCase within a component (`CubePush`). Ablation axes are cfg values, not ids. No alias ids.
- No terminal charges (-100 on falling) and no `is_alive` terms. Keep the per-step sum positive, with penalties as
  bounded kernels. No explicit gait or phase rewards where a style expert exists.
- High-level policies observe no latent (including their own previous latent action).

## Code style

- **Push gate:** `VIRTUAL_ENV="" uvx pre-commit run --all-files` must pass (ruff check + format, codespell, yaml);
  CI runs it verbatim. ruff: line length 120, py311 syntax. The task repos' only build check is the CPU smoke:
  `pytest tests/test_smoke.py -m "not gpu"` (`pls test <repo>`).
- **Comments: one line ideal, two the hard max**, stating only the non-obvious why. Never reference milestones,
  dates, people, run ids or plan names, and never narrate the bug a change fixed (that goes in the commit).
- **Docstrings:** a one-line summary plus `Args:`/`Returns:` (entries of at most two lines), scoped to what the unit
  does. One constraint sentence of at most two lines is allowed when callers must know it.
- **Config classes** follow Isaac Lab (`@configclass`, `MISSING`, paired `XTerm`/`XTermCfg`). Generic names carry
  generic content: shared knobs in a base cfg, feature-specific fields in subclasses.
- **File references:** cite paths from the manager root (`resources/<repo>/...:<line>`) or absolute.
- No `HCRL_*` environment-variable switches for behavior; use a cfg field or a CLI flag.

## Git and PRs

- Work on a branch, never on `main`. Commit and push feature branches as you go.
- Branch names are `<type>/<short-kebab-description>` (e.g. `fix/rss-reorient-hcrl-env-cfg`); when two types fit,
  take the one a reviewer most needs to know (`fix/` over `refactor/`). Long-lived lines (`main`, `isaac-2`) keep
  their names. Types:
  - `feat/`: new behavior, task, option or tool
  - `fix/`: a bug, crash or wrong result
  - `refactor/`: restructuring with no change in behavior
  - `perf/`: speed or memory, results unchanged
  - `test/`: tests or test infrastructure only
  - `docs/`: README, CLAUDE.md, docstrings or comments only
  - `chore/`: dependencies, lockfiles, CI, cleanup and deletions
  - `sync/`: merging one long-lived branch into another
  - `revert/`: undoing an earlier merge
  - `exp/`: experiments and training runs not meant to merge (open as drafts, if at all)
- PRs follow each repo's `.github/PULL_REQUEST_TEMPLATE.md`: one framing sentence plus a bullet per change, related
  PRs as full URLs, and an honest checklist. Design docs go in the repo's gitignored `.claude/`, not the PR.
- Dependent PRs in one repo are stacked (`gh stack`). After each push, `git diff --name-only origin/<base>...HEAD`
  must list only your files.
