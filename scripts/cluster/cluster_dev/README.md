# Persistent cluster dev-node (`cluster_dev.sh`)

Turn an HPC compute node into a dev/training box you can SSH into all day, paying 2FA (where
the site requires it) **once**. Fully cluster-agnostic: all site specifics come from the
selected `<cluster>_config/.env.cluster` — nothing in the script is Delta-specific.

## Why this shape (the key constraint)
Interactive partitions are usually short-capped (e.g. NCSA Delta's is **1 hour**); only
**batch** jobs get the long walltime. Many sites also allow *"direct ssh to a compute node in
a running job"*. So we submit a long-lived **batch "sentinel"** job that just holds a node,
then ssh into it. Where SSH keys are disabled (password+2FA every login, e.g. Delta), a
persistent **ControlMaster** socket — opened once, kept warm — is what avoids re-authenticating.

## User command
```bash
# from ~/hcrl_isaac_manager — CLUSTER selects <cluster>_config/
CLUSTER=rtx-small scripts/cluster.sh develop start   # approve ONE 2FA prompt; rest is non-interactive
```
`develop` dispatches to `cluster_dev.sh` with `CLUSTER` set. `start` opens the SSH master (the
only 2FA prompt), mirrors the IsaacLab tree up, submits the sentinel job, and launches a
**background watcher** that tracks the (possibly multi-hour) queue wait. You can walk away.

## Tracking / using it (no credentials needed once the master is up)
```bash
CLUSTER=rtx-small scripts/cluster.sh develop status      # job id / state / node / master+watcher health / live squeue
CLUSTER=rtx-small scripts/cluster.sh develop attach      # interactive shell on the compute node (once RUNNING)
CLUSTER=rtx-small scripts/cluster.sh develop exec -- <cmd>  # run <cmd> inside the Apptainer container on the node
CLUSTER=rtx-small scripts/cluster.sh develop sync        # re-mirror local IsaacLab edits → cluster
CLUSTER=rtx-small scripts/cluster.sh develop stop        # scancel the job + close the master
```
The watcher writes `~/.cluster_dev/state` (and `~/.cluster_dev/watch.log`); when
`JOB_STATE=RUNNING` and `NODE` is set, the box is ready. A Claude session can poll `status` and
drive `exec`/`attach` over the live master with zero auth.

## Config knobs (set in `<cluster>_config/.env.cluster`)
The sentinel's resources are cluster-agnostic. **Any `CDEV_*` left unset emits no corresponding
`#SBATCH` directive**, so SLURM falls back to the queue default — set only what your queue needs.

| var | default | note |
|---|---|---|
| `CDEV_ACCOUNT` | _(unset)_ | `#SBATCH -A` allocation/charge code; omit where not required (e.g. rtx-small) |
| `CDEV_PARTITION` | _(unset)_ | `#SBATCH -p` queue; omit for the cluster's default partition |
| `CDEV_GPUS_PER_NODE` | _(unset)_ | unset → no `--gpus-per-node` (queue default); set `N` to force a count (also sets srun `--gres=gpu:N`) |
| `CDEV_CPUS` | _(unset)_ | `#SBATCH --cpus-per-task`; omit for queue default |
| `CDEV_MEM` | _(unset)_ | `#SBATCH --mem`; omit for queue default |
| `CDEV_EXCLUSIVE` | _(unset)_ | set `1`/`true`/`yes` to request a whole node (`--exclusive`) |
| `CDEV_TIME` | `48:00:00` | walltime to hold the node; cap to the queue max |
| `CDEV_ATTACH_MODE` | `auto` | `auto` probes login→node ssh at job start; force with `ssh`/`srun` (or `attach --ssh/--srun`) |
| `CDEV_LOGIN_HOST` | from `CLUSTER_LOGIN` | login host; override to pin a specific login node |
| `LOCAL_ISAACLAB_DIR` | resources/IsaacLab | code mirrored up |

Example configs shipped: `delta_config` / `multi-delta_config` (whole `gpuA40x4` node:
`CDEV_ACCOUNT=bggq-delta-gpu`, `CDEV_PARTITION=gpuA40x4`, `CDEV_GPUS_PER_NODE=4`, `CDEV_CPUS=64`,
`CDEV_MEM=200g`, `CDEV_EXCLUSIVE=1`); `rtx-small_config` (shared TACC queue: `CDEV_PARTITION=rtx-small`,
`CDEV_CPUS=14`, everything else inherited).

## Attach mode (auto-detected, no guess)
On job start the watcher probes `ssh login→node` with `BatchMode=yes` (fails fast instead of
prompting). If it works passwordlessly → `attach`/`exec` use **direct ssh** (cleanest GPU
access). If not (some sites, e.g. Delta, block this without re-auth) → they use **`srun
--overlap --jobid`** from the login node (needs no node ssh, no 2nd auth). Force with
`CDEV_ATTACH_MODE=ssh|srun` or `attach --ssh|--srun`.

## Notes / caveats
- The SIF must already be current on the cluster — **do not add dependencies** in
  `node_exec.sh` or training (editable `-e` code changes are fine, new third-party deps are
  not; rebuild/push the image instead).
- Assets ride the rsync (`resources/` is not excluded; the first sync may be slow if the
  cluster doesn't already have them).
- Compute nodes need outbound internet for live W&B logging.

## Files
- `cluster_dev.sh` — control script (start/status/attach/exec/sync/stop + internal watcher).
- `sentinel.sbatch` — node-holding job (envsubst template; resource directives filled in/omitted
  per `.env.cluster`; does no heavy setup so a staging bug can't waste the allocation).
- `node_exec.sh` — runs on the node; stages SIF+caches+code once, then `apptainer exec`s
  (bind mounts mirror `docker/cluster/run_singularity.sh`). Reached via the synced
  `${CLUSTER_ISAACLAB_DIR}/docker/cluster/cluster_dev/` copy.
