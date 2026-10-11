# Persistent cluster dev-node (`cluster_dev.sh`)

Turn an HPC compute node into a dev/training box you can SSH into all day, paying 2FA (where
the site requires it) **once**. Fully cluster-agnostic: all site specifics come from the
selected `scripts/cluster/config/<cluster>/.env.cluster` — nothing in the script is Delta-specific.

## Why this shape (the key constraint)
Interactive partitions are usually short-capped (e.g. NCSA Delta's is **1 hour**); only
**batch** jobs get the long walltime. Many sites also allow *"direct ssh to a compute node in
a running job"*. So we submit a long-lived **batch "sentinel"** job that just holds a node,
then ssh into it. Where SSH keys are disabled (password+2FA every login, e.g. Delta), a
persistent **ControlMaster** socket — opened once, kept warm — is what avoids re-authenticating.

## User command
```bash
# from the manager dir; the leading name selects scripts/cluster/config/<cluster>/
just cluster rtx-small develop start   # approve ONE 2FA prompt; rest is non-interactive (--no-stage: ship no code)
```
`develop` dispatches to `cluster_dev.sh` with `CLUSTER` set. `start` opens the SSH master (the
only 2FA prompt), stages the workspace as the code tree `default` (below), submits the sentinel job, and launches a
**background watcher** that tracks the (possibly multi-hour) queue wait. You can walk away.

## Tracking / using it (no credentials needed once the master is up)
```bash
just cluster rtx-small develop status      # job id / state / node / master+watcher health / live squeue
just cluster rtx-small develop attach      # interactive shell on the compute node (once RUNNING)
just cluster rtx-small develop exec -- <cmd>  # run <cmd> inside the Apptainer container, on the newest `default`
just cluster rtx-small develop stage       # ship local edits: the workspace as on disk -> a new `default`
just cluster rtx-small develop stop        # scancel the job + close the master
```
The watcher writes `~/.cluster_dev/<cluster>/state` (and `watch.log` next to it); when
`JOB_STATE=RUNNING` and `NODE` is set, the box is ready. A Claude session can poll `status` and
drive `exec`/`attach` over the live master with zero auth.

`exec -- <cmd> <args...>` reaches the container with its argv unchanged, so for shell syntax use
`exec -- bash -lc 'a; b && c'`. A single argument (`exec -- "python scripts/train.py --task X"`) is run as a
command string inside the container.

## Code trees (how code reaches a cluster)
```bash
just cluster delta develop stage                       # the workspace as on disk (uncommitted work too) -> `default`
just cluster delta develop stage push-foot hhlm_tasks=feat/push-foot-contact-penalty hcrl_isaaclab=main robot_rl=main
just cluster delta develop exec --tree push-foot -- python scripts/train.py --task ...   # (--detach as usual)
just cluster delta develop trees                       # list staged trees
just cluster delta develop trees rm push-foot-<fp>     # remove one (refused while a job that used it runs)
just cluster delta develop trees rm --partials         # clear interrupted stages older than an hour
```
`stage` uploads repos into a new `<CLUSTER_ISAACLAB_DIR>/trees/<name>-<fingerprint>/`. Nothing is mirrored onto a
shared workspace, so a run never sees code change under it:

- Bare `stage` is the whole workspace as on disk, named `default`: every git repo under `resources/` (IsaacLab
  included) except the shared data repo `motion_datasets`, and except the repos named in this cluster's
  `config/<cluster>/.rsync-exclude` (one repo name per line, also honoured from profiles with the same
  `CLUSTER_ISAACLAB_DIR`). It also adds `motion_datasets`' training files (`*.pt` and their
  `.manifest/.arena/.courts.json`) to the shared `resources/motion_datasets`, never deleting there; trees link it.
- `exec` without `--tree`, `start` and `pls run --on <name> --batch` (without `--tree`) use the newest `default`; each resolves
  the tree once, so a step or queued job keeps its code while newer trees are staged.
- A repo is given at a ref (fetched; `origin/<ref>` preferred) or as a local worktree top
  (`hcrl_isaaclab=./resources/hcrl_isaaclab/worktrees/wt`; `/`, `./` or `../`, relative to the manager dir under
  `just`), which carries its tracked and untracked non-ignored files.
- A named stage takes every repo it does not name from the newest `default` (code hardlinked, asset repos copied),
  or links the cluster's shared checkout where no `default` is staged yet; the MANIFEST says which.
- Files identical to one in a recent tree are hardlinked instead of re-sent, never to the shared checkout (which
  runs write to); a hardlink takes no inode, so a restaged `default` costs about one inode per directory plus the
  changed files. Staged files are read-only, so an in-place write fails instead of changing every tree that
  shares the file; the directories stay writable.
- An asset repo (`*_robots`) stages like any other (e.g. `hcrl_robots=fix/t1-shank-mass`), except that its files
  stay writable and are the tree's own copies, never hardlinked: the run's URDF->USD conversion writes beside the
  URDF, and that must change neither another tree nor the shared checkout.
- `MANIFEST` records each repo's ref or absolute path, commit and content hash (modes and symlink targets
  included). Staging the same content again reuses the tree, and a failed stage leaves no tree, partial or
  temporary checkout behind.
- `exec --tree <id>` takes the full `<name>-<fingerprint>` id, or a bare name for its newest tree. It runs
  the tree's own `node_exec.sh` and mounts staged repos writable, since the W&B artifact resolver re-links
  policies inside them. `hcrl_isaaclab/logs` goes to the shared logs dir, and the other `logs`/`outputs`/`wandb`
  dirs are node-local. Artifact links resolved in the shared checkout are carried into the tree, and the shared
  artifact root stays writable.
- Each run holds `.in-use/<job>.<step>` until it exits, and `trees rm` refuses while squeue still lists that step.
  A batch job holds `.in-use/<jobid>.nostep` while squeue lists it. `trees rm --partials` removes interrupted stages
  in which nothing changed for an hour. After each stage, trees of that name beyond the newest 5 are removed unless
  a run still holds them.
- `stage` and `exec` first check the free space where trees and run logs go (`CLUSTER_TREES_DIR`, `CLUSTER_LOGS_DIR`)
  and refuse below `CLUSTER_MIN_FREE_GB`: a full quota fails every checkpoint write while the run keeps going.
  On Lustre, `stage` also refuses once the metadata target holding the trees is over 98% of its inodes (`lfs df -i`),
  where every new file fails. `--no-space-check` skips both.
- `scripts/cluster/tests/test_stage.sh` checks the artifact resolver against a staged tree only where an
  hcrl_isaaclab checkout exists (locally, or with `HCRL_ISAACLAB_DIR`); CI skips that check.

## Config
The sentinel reuses the `#SBATCH` directives of the cluster's own `config/<cluster>/submit_job_slurm.sh`
(minus `--job-name`/`--output`), so the dev box asks for what a `job` submission asks for. `attach`/`exec`
pass the same `-p`, `-A`, `--gpus-per-node` and `--cpus-per-task` to their `srun --overlap` steps.

| directive | note |
|---|---|
| `#SBATCH -A` | allocation/charge code; required on TACC (`rtx-small`, `amd-rtx`), where the account has more than one project and both `sbatch` and every `srun --overlap` step refuse without it |
| `#SBATCH -p` | queue; omit for the cluster's default partition |
| `#SBATCH --gpus-per-node` | also sets the steps' `--gres=gpu:N`; omit for the queue default |
| `#SBATCH --cpus-per-task` | also given to each step, which otherwise binds to one cpu |
| `#SBATCH --time` | walltime to hold the node (steps default to `48:00:00` when unset) |

The rest comes from `config/<cluster>/.env.cluster`:

| var | default | note |
|---|---|---|
| `CLUSTER_LOGIN` | _(required)_ | `user@login-host` |
| `CLUSTER_ISAACLAB_DIR` | _(required)_ | the workspace's path on the cluster |
| `CLUSTER_LOGIN_HOST` | from `CLUSTER_LOGIN` | override to pin a specific login node |
| `CLUSTER_ATTACH_MODE` | `auto` | `auto` probes login→node ssh at job start; force with `ssh`/`srun` (or `attach --ssh/--srun`) |
| `CLUSTER_SRUN_EXTRA` | _(unset)_ | extra options for every `srun --overlap` step |
| `LOCAL_ISAACLAB_DIR` | the manager dir | the workspace a bare `stage` ships |

## Attach mode (auto-detected, no guess)
On job start the watcher probes `ssh login→node` with `BatchMode=yes` (fails fast instead of
prompting). If it works passwordlessly → `attach`/`exec` use **direct ssh** (cleanest GPU
access). If not (some sites, e.g. Delta, block this without re-auth) → they use **`srun
--overlap --jobid`** from the login node (needs no node ssh, no 2nd auth). Force with
`CLUSTER_ATTACH_MODE=ssh|srun` or `attach --ssh|--srun`.

## Notes / caveats
- The SIF must already be current on the cluster — **do not add dependencies** in
  `node_exec.sh` or training (editable `-e` code changes are fine, new third-party deps are
  not; rebuild/push the image instead).
- The first `default` uploads the whole workspace (assets included) and may be slow; later ones send changed files.
  The shared `<CLUSTER_ISAACLAB_DIR>/resources` is no longer written by anything but the data repo's additions: code
  left there by earlier syncs is only what a tree links when no `default` exists.
- Compute nodes need outbound internet for live W&B logging.

## Files
- `cluster_dev.sh` — control script (start/status/attach/exec/stage/trees/stop + internal watcher); `trees.sh` stages.
- `sentinel.sbatch` — node-holding job (envsubst template; resource directives come from the
  profile's `submit_job_slurm.sh`; does no heavy setup so a staging bug can't waste the allocation).
- `node_exec.sh` — runs on the node; stages SIF+caches+code once, then `apptainer exec`s
  (bind mounts mirror `scripts/cluster/run_singularity.sh`). Each tree carries its own copy.
