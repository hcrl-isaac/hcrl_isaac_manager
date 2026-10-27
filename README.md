# hcrl_isaac_manager

Composes the hcrl Isaac Lab workspace — core `hcrl_isaaclab`, per-project `*_tasks` / `*_robots`,
and `robot_rl` — and deploys training to local GPUs, Ray, or HPC clusters. This README is the
quickstart; per-cluster READMEs ([Ray](scripts/ray/README.md), [Cluster](scripts/cluster/README.md),
[LARG](scripts/larg/README.md)) cover what each piece does and the full options.

Working with Claude Code: [CLAUDE.md](CLAUDE.md) describes the workspace for a Claude session, and
`.claude/skills/` holds its procedures (`setup`, `add-compute`, `launch-run`, `new-task`, `convention-review`).

## Prerequisites

- `$HOME/.local/bin` on your `PATH`, and `sh` + `curl` (`./bootstrap.sh` installs uv if it is missing).

All workspace commands go through `pls` (below). On a fresh machine, `./bootstrap.sh` runs `pls setup` before `pls`
is installed; afterwards `pls` is on your PATH whenever the `ilab` venv is active.

## Configure the workspace

**Always run `pls setup`** — whether or not you want a local IsaacLab install. It opens a picker (arrow
keys, space to toggle) for **which projects** to install and how to handle **IsaacLab**, pre-filled with
your current selection, so re-running `pls setup` doubles as reconfiguration. The IsaacLab choice is:

- **source** — clone Isaac Lab 3 under `resources/IsaacLab` (at the ref in `workspace.defaults.yaml`) and install
  it from its uv lock (the usual local setup; Isaac Lab 3 has no pip release).
- **none** — don't install IsaacLab (or torch / the workspace packages); just fetch the project repos under
  `resources/`. Use this to browse/edit the projects locally and run training on a cluster or Ray instead.

The picker is part of `setup` (not a standalone command) because changing the selection changes which repos
are pulled and which packages installed, so it always flows through the full pull+install. On a
non-interactive shell (CI) it keeps your existing selection without prompting.

Your selection is written to a gitignored `workspace.yaml` (just the projects + IsaacLab mode); the
shared catalog of selectable projects and the defaults (org, refs, IsaacLab version) live in the
committed [`workspace.defaults.yaml`](workspace.defaults.yaml). To add a new selectable project, add it
to `available_projects` there. This replaces the old per-project git branches — selection is now local.

To pull in or update the workspace repos at any time, run **`pls resolve`** (see [`resolve`](#resolve)) —
after a repo adds a dependency in its `dependencies.yaml`, or to fetch each repo up to its pinned ref.

## Install (local)

```bash
./bootstrap.sh          # a fresh machine: installs uv, then runs `pls setup`
./bootstrap.sh deps     # a checkout that already has ilab: adds pls to it, removes nothing
pls setup               # afterwards (from the ilab venv), to reinstall or reconfigure
```

Resolves `workspace.yaml` into a flat, deduped set of repos under `resources/`, then builds a single
uv venv (`ilab`) with the manager + Isaac Lab / Isaac Sim stack (unless you picked IsaacLab **none**, which
stops after fetching the repos). Adds one bash alias:

- `ilab` — activate the `ilab` venv and cd to the manager dir. From there, run any extension script
  with `pls run <script> <args>` (no need to cd into the extension).

Scaffold a new project repo with `pls new <name>` (registers under the `<name>/` namespace).

## Run

Everything runs through `pls run`; `--on <where>` picks the place, and the script is the same everywhere.
`--source` selects a project's namespace and is only needed when a task id is shared by two projects.

**This machine:**
```bash
ilab                                                   # activate the ilab venv + cd to the manager dir
pls run train --task <task-id> [--source <ssti|hhlm>] # run any hcrl_isaaclab script from here
```
`pls run <script> <args>` runs any `hcrl_isaaclab/scripts/<script>.py` from the manager dir (no need
to cd into the extension): `pls run play --task <id> --checkpoint <path>`, `pls run video_logger …`,
`pls run export …`, etc.

**A GPU elsewhere** (a LARG box, or a card of a cluster's held node; see [Compute](#compute-pls-res)):
```bash
pls res status --free                                  # which cards are free, everywhere
pls run --on hazard:2 --holder <you> --note "<why>" -- train --task <task-id>
pls run --on larg-a40 --holder <you> -- video_logger --run <url>   # any free card of a pool
```

**Ray:**
```bash
pls setup --on ray             # one-time: write the Ray configs; then build + push the image (when deps change)
pls upload-artifacts --all     # large assets as W&B artifacts (when they change)
pls run --on ray -- train --task <task-id> [train args]
pls jobs --on ray [logs <job> | stop <job>]
```
Large files (robot assets, motions, policies) are excluded from the job upload and fetched at runtime
as W&B artifacts, which `pls upload-artifacts` publishes. See the [Ray README](scripts/ray/README.md).

**HPC (SLURM):**
```bash
pls setup --add <name>         # one-time per cluster: write its profile
pls setup --on <name>          # build + push the .sif (when deps change)
pls run --on <name> --batch -- train --task <task-id> [train args]   # a queued job of its own (any repo script)
pls res hold --on <name>       # or hold a node, then run on its cards: pls run --on <name> -- <script>
pls jobs --on <name> [logs <job> | stop <job>]
```
See the [Cluster README](scripts/cluster/README.md).

## W&B custom x-axes

Add `local_step` and `env_step` as workspace metrics (workspace settings, top-left) to use as the
x-axis in line graphs / media:

- `local_step` — training iteration. Use it instead of `_step` (W&B forces `_step` monotonic, so
  out-of-process / async-logged videos can't be logged at the step they were recorded for).
- `env_step` — total env steps (`local_step * num_envs * num_steps_per_env`); compares runs that use
  different env counts.

## Asynchronous Video Logging

A run on a card without RT cores (A100, H100, Horizon GB200), or one that leaves its card to training, records its
videos asynchronously: the trainer tags the W&B run, and a recorder on **any free RT-capable card** finds the tagged
run, pulls its checkpoints, renders the clips and uploads them back to the same run. Every RT card renders: the
LARG and Delta A40s, Stampede3's RTX PRO 6000s and the 5090s (driver 595/615 boxes through the Vulkan clamp layer,
see [LARG rendering](scripts/larg/README.md#rendering-on-larg-the-vulkan-clamp-layer)). Take whichever is free in
`pls res status --free` rather than waiting for one box.

### 1. Tag the training run

Pass `--video async` to `train.py` (`--eval async` for a task with an eval). It tags the W&B run with
`log_evals_async` instead of spawning a recorder next to the trainer. Runs tagged with the old `log_videos_async`
are still picked up.

### 2. Run the recorder on an RT card

`hcrl_isaaclab/scripts/video_recorder.py --mode async` sweeps a project for tagged runs and renders each pending
checkpoint, one `video_logger.py` worker per run:

```bash
pls run video_recorder --mode async --task <task_name> --wandb_project <entity>/<project> [options]   # this machine
pls run --on <host>:<gpu> --holder <you> -- video_recorder \
    --mode async --task <task_name> --wandb_project <entity>/<project> [options]                     # a leased card
```

| Arg | Default | Purpose |
| --- | --- | --- |
| `--task <id>` | -- | IsaacLab env id to rebuild for rendering. |
| `--wandb_project <entity>/<project>` | -- | Project to sweep for tagged runs (required in async mode). |
| `--only_run <id>` | -- | Restrict the sweep to one run. |
| `--num_envs <N>` | 64 | Env count for the small render sim. |
| `--video_length <steps>` | 400 | Clip length in env steps (capped to one episode). |
| `--train_effective_envs <N>` | -- | Training run's `num_envs * world_size`, so the curriculum clock is restored correctly in the render sim. |
| `--rerecord_from <iter>` / `--rerecord_all` | -- | Re-record checkpoints >= `iter` (or all). |
| `--stochastic` | off | Record the policy **sampling** actions instead of the deterministic mean. |

### Cron listener

To run the async logger automatically, register a cron listener (every 30 min):

```bash
scripts/video_listener.sh add --task <task_name> --wandb_project <entity>/<project>
```

Remove it with:

```bash
scripts/video_listener.sh remove --task <task_name> --wandb_project <entity>/<project>
```

This installs a cron job that runs `hcrl_isaaclab/scripts/utils/log_evals_async.sh`, which activates
the `ilab` venv, sources your W&B credentials, skips the pass if GPU 0 is >50% busy, and invokes
`video_logger.py --mode async`. See `scripts/video_listener.sh --help` for all options.

## pls

`pls` is the workspace command line (`src/hcrl_cli`), installed into the `ilab` venv by `pls deps`; `pls --help`
lists its verbs, grouped by what they do, and each verb has its own `--help`. On a fresh machine `./bootstrap.sh`
installs uv and runs `pls setup` before `pls` exists.

Every command that acts somewhere else names the place with `--on <where>`:

| `--on` | means |
| --- | --- |
| (none) | this machine |
| `host:gpu`, `<pool>`, `any`, `lease:<id>` | a GPU card: one card, any free card of a pool, or one you lease |
| `<cluster>` | a SLURM cluster profile (`scripts/cluster/config/<name>`) |
| `ray` | the Ray cluster |

```bash
pls run [--on W] [--wt SET] -- <script> [args]    # run a script anywhere (pls run --help)
pls jobs --on C [logs J | stop J]                 # the jobs on a cluster or Ray
pls res status [--free | --on C]                  # every GPU: who holds what
pls res claim | release | hold --on C | drop --on C <job>   # leases and held cluster nodes
pls connect --on C / pls disconnect --on C        # a cluster's SSH connection (2FA)
pls shell --on C                                  # a shell on the cluster's held node
pls stage --on C [list | <name> <repo>=<ref>]     # code trees on a cluster (pls run stages for you)
pls setup [--on C | --add NAME]                   # install here; build/push a cluster's image; add a cluster
```

The shared `ilab`'s `pls` runs the main checkout's `src/hcrl_cli`; to try `pls` changes from a worktree, run
`PYTHONPATH=<worktree>/src python -m hcrl_cli <verb> ...` from anywhere.

### Using pls yourself

- Commands connect on their own: the first one that needs a cluster opens its SSH connection and asks for your 2FA
  once. `pls connect --on <cluster>` does it ahead of time (say, before you leave a long launch queued).
- Code reaches a remote card or cluster only as a staged copy of your checkouts: `--wt <set>` runs a worktree set,
  and a dirty worktree is refused unless you pass `--allow-dirty`. Each run records the commits it ran under
  `pls_refs` in its W&B config.
- A card is yours only while you lease it: `pls run --on` leases it for the run, and `pls res claim` holds one longer.
  `pls res status` shows who holds what.

### Using pls from an agent session

Claude sessions use the same commands, with a few differences:

- **A session cannot answer 2FA.** When a cluster is not connected, a command fails at once with "ask the user to
  run `pls connect --on <cluster>`", rather than waiting on a prompt. The session asks the user (a push notification
  when they are away); one connection serves every session until `pls disconnect`.
- **Leases carry a holder and an objective:** `--holder "<session name>" --note "<what the run is for>"`, so
  `pls res status` tells the user and the other sessions what each card is doing. A card no run uses is released
  after 30 minutes idle.
- **Long runs detach and are watched:** `pls run ... --detach` prints the log path and a stop command, and
  `scripts/tools/watch_run.sh [--wandb <entity/project/run>] <label> <log> <target-iters>` exits on finished, failed,
  stalled or unreachable, so a background watcher wakes the session instead of the session polling.
- **Reproducible by default:** run repo scripts (`<repo>/<path>` or a script name) rather than local copies, so the
  run's W&B config records `pls_refs`, and the run can be replayed from the commits it ran.
- **When a command cannot do what the run needs, that is an infra gap:** report it (the command, what it lacks, the
  exact error) rather than writing a private launcher.

## pls verbs

`pls --help` lists every verb; environment dependencies are managed with the **uv** package manager. A verb's own
`--help` comes from the script it runs (`pls res --help`, `pls run --help`).

### `deps`

- Installs system tools (uv, gitman) and the manager uv environment
- Creates the W&B env file, if necessary
- Adds the `ilab` bash alias (activate the venv + cd to the manager dir) to your RC file, if necessary

> **Note**: called internally by `setup`.

### `setup`

- Installs general dependencies (`pls deps`)
- Opens the project + IsaacLab-mode (source/**none**) picker (arrow keys + space), pre-filled with your current selection, so re-running `setup` reconfigures; writes the gitignored `workspace.yaml` (keeps the existing selection on a non-interactive shell)
- Resolves the selection + defaults and fetches the workspace repos (`pls resolve`)
- Builds the Python 3.12 `ilab` venv, installs Isaac Lab 3 from its uv lock and editable-installs every workspace package into the local uv env — **skipped entirely for IsaacLab `none`** (repos are just fetched, for editing locally + running elsewhere)

### `resolve`

- Merges your selection (`workspace.yaml`) with the committed `workspace.defaults.yaml` and each repo's `dependencies.yaml` into a flat, deduped `gitman.yaml`, then fetches/updates all repos as siblings under `resources/` (via `gitman update`)
- This is how you **pull and update workspace dependencies**. Re-run it after a repo declares a new dependency in its `dependencies.yaml`, or to pull every repo up to its pinned ref. (To *change* which projects are installed, re-run `pls setup`, which also installs the new deps.)
- Afterwards every repo is **returned to the branch it was on** — updating fetches the pinned rev without silently moving your checkout
- Repos with **uncommitted local changes** get a per-repo `[y/N]` prompt to merge them onto the updated rev (stash → update → stash pop; conflicts are left in the tree with the stash preserved). Flags:
  - `--force` — merge all dirty repos without prompting
  - `--skip-changes` — leave all dirty repos untouched (they won't be updated)
  - `--checkout-pin` — end with every repo checked out at its **pinned rev** instead of restoring your branch. Dirty repos are stashed (no prompt) so the checkout can move; the stash is popped only if the repo was already on the pin, otherwise it is kept and the console prints the `git checkout <branch> && git stash pop` command to recover it

### `new <name>`

- Scaffolds a new `<name>_tasks` extension repo under `resources/`, registered under the `<name>/` namespace

### `run <script> [args]`

- Runs `hcrl_isaaclab/scripts/<script>.py` with the `ilab` venv from the manager dir (no need to cd
  into the extension), e.g. `pls run train --task <id>`, `pls run play …`, `pls run video_logger …`
- With options (`--on`, `--wt`, `--cmd`, `--batch`) the script follows `--`: see [pls](#pls) above and `pls run --help`
- Executes the script *file* directly so Isaac Sim's `AppLauncher` runs before the package import
  (the reason `python -m hcrl_isaaclab.scripts.<script>` can't work); auto-accepts the Isaac Sim EULA

### `clean`

- Deletes the Isaac Lab environment
- Removes the bash aliases from your RC file

### `docker`

- Installs and configures Docker if necessary
- Builds and starts the Isaac Lab Docker container

### `run`

- `pls run [--on W] [--wt SET | --tree T] [--batch] [--cmd] -- <script|command> [args]`: here, on a leased card, as a
  cluster batch job, or on Ray; `pls run --help` has the full grammar
- A batch job (`--on <cluster> --batch`) is a queued SLURM job of its own: any `<name>` or `<repo>/<path>` script,
  in place of the profile's `train.py` (any launcher before it, such as `torchrun`, is kept)

### `jobs`

- `pls jobs --on <cluster|ray>` lists queued and running jobs; `logs <job>` shows a job's output (`-f` follows),
  `stop <job>` cancels it, and `kill [--all | STEP...]` cancels run steps on a cluster's held node, keeping the node

### `res` <a id="compute-pls-res"></a>

- `pls res status [--free] [--on C]` probes every GPU (boxes, clusters, Ray) and shows each card's state and lease
- `claim`, `release`, `transfer`, `reserve` manage card leases; `pls res --help` lists them
- `hold --on <cluster>` submits a held node (a long-lived job whose cards `pls run --on` leases; it may queue for
  hours), `drop --on <cluster> <job>` cancels one, and `status --on <cluster>` shows the connection and held node first

### `connect`, `disconnect`

- `pls connect --on <cluster>` opens the cluster's shared SSH connection (2FA once); every command and session reuses
  it. `pls disconnect --on <cluster>` closes it for all of them

### `shell`

- `pls shell --on <cluster>` opens an interactive shell on the cluster's held node

### `stage`

- `pls stage --on <cluster>` stages the workspace as the code tree `default`; `<name> <repo>=<ref|path> ...` stages a
  named tree with those repos pinned; `list` shows the trees and `rm <tree>` removes one
- `pls run` stages what it runs by itself; `pls run --on <cluster> --tree <name> -- <script>` runs a staged tree

### `setup --on`, `setup --add`

- `pls setup --on <cluster>` builds the shared image and pushes it (the Apptainer `.sif`; `--build-only`,
  `--push-only`); `pls setup --on ray` sets up the Ray cluster
- `pls setup --add <name> [--update]` creates (or regenerates) a SLURM cluster profile
- `pls cluster ...` is retired: each old form prints the command that replaced it

### `upload-artifacts [args]`

- Uploads managed large-file resources (robot assets, motion datasets, exported policies) to W&B as versioned artifacts
- `--list` shows the registry + local presence; `--all` uploads everything; or pass specific resource keys
- Dedups unchanged content by hash (cheap to re-run); reads W&B credentials from `scripts/.env.wandb`
- Fetched back at runtime by the in-script resolver — see [Large-file resources](scripts/ray/README.md#large-file-resources)
