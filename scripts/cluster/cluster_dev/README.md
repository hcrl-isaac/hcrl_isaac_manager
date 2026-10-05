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
just cluster rtx-small develop start   # approve ONE 2FA prompt; rest is non-interactive
```
`develop` dispatches to `cluster_dev.sh` with `CLUSTER` set. `start` opens the SSH master (the
only 2FA prompt), mirrors the IsaacLab tree up, submits the sentinel job, and launches a
**background watcher** that tracks the (possibly multi-hour) queue wait. You can walk away.

## Tracking / using it (no credentials needed once the master is up)
```bash
just cluster rtx-small develop status      # job id / state / node / master+watcher health / live squeue
just cluster rtx-small develop attach      # interactive shell on the compute node (once RUNNING)
just cluster rtx-small develop exec -- <cmd>  # run <cmd> inside the Apptainer container on the node
just cluster rtx-small develop sync        # re-mirror local IsaacLab edits → cluster
just cluster rtx-small develop stop        # scancel the job + close the master
```
The watcher writes `~/.cluster_dev/<cluster>/state` (and `watch.log` next to it); when
`JOB_STATE=RUNNING` and `NODE` is set, the box is ready. A Claude session can poll `status` and
drive `exec`/`attach` over the live master with zero auth.

## Code trees (run specific branches without syncing)
```bash
just cluster delta develop stage push-foot hhlm_tasks=feat/push-foot-contact-penalty hcrl_isaaclab=main robot_rl=main
just cluster delta develop exec --tree push-foot -- python scripts/train.py --task ...   # (--detach as usual)
just cluster delta develop trees                       # list staged trees
just cluster delta develop trees rm push-foot-<fp>     # remove one (refused while a job that used it runs)
just cluster delta develop trees rm --partials         # clear interrupted stages older than an hour
```
`stage` uploads each named repo at a ref (fetched; `origin/<ref>` preferred) or a local worktree
(`hcrl_isaaclab=./resources/hcrl_isaaclab/worktrees/wt`: the worktree top, `/`, `./` or `../`, relative to the manager dir under
`just`, carrying tracked and untracked non-ignored files) into a new `<CLUSTER_ISAACLAB_DIR>/trees/<name>-<fingerprint>/`.
It never uses `--delete` and never touches the shared checkout. Every repo not named is a link to the shared one;
asset repos (`*_robots`, where URDF->USD conversion writes) and `IsaacLab` always are, and cannot be staged. Files
identical to one in a recent tree are hardlinked instead of re-sent (never to the shared checkout, which runs write
to). `MANIFEST` records each repo's ref or absolute path, commit and content hash (modes and symlink targets
included). Staging the same content again reuses the tree, and a failed stage leaves no tree, partial or temporary
checkout behind. `exec --tree <id>` takes the full `<name>-<fingerprint>` id, or a bare name when exactly one tree
has it. Staged repos are mounted read-only with their own `node_exec.sh`; `hcrl_isaaclab/logs` goes to the shared
logs dir and the other `logs`/`outputs`/`wandb` dirs are node-local. W&B artifact links resolved in the shared checkout are
carried into the tree, and the shared artifact root stays writable. Each run holds `.in-use/<job>.<step>` until it
exits, and `trees rm` refuses while squeue still lists that step. `trees rm --partials` removes interrupted stages
in which nothing changed for an hour. `develop sync` leaves `trees/` alone. Trees are not removed automatically.

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
| `LOCAL_ISAACLAB_DIR` | the manager dir | code mirrored up |

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
- `start`/`sync` mirror the local checkout (assets included; the first sync may be slow) with
  `--delete`, but never ship or delete worktrees or remote-only `resources/*` repos. Every config
  that syncs to the same `CLUSTER_ISAACLAB_DIR` shares that tree.
- Compute nodes need outbound internet for live W&B logging.

## Files
- `cluster_dev.sh` — control script (start/status/attach/exec/sync/stop + internal watcher).
- `sentinel.sbatch` — node-holding job (envsubst template; resource directives come from the
  profile's `submit_job_slurm.sh`; does no heavy setup so a staging bug can't waste the allocation).
- `node_exec.sh` — runs on the node; stages SIF+caches+code once, then `apptainer exec`s
  (bind mounts mirror `scripts/cluster/run_singularity.sh`). Reached via the synced
  `${CLUSTER_ISAACLAB_DIR}/scripts/cluster/cluster_dev/` copy.
