# Delta persistent dev-node (`cluster_dev.sh`)

Turn an NCSA Delta `gpuA40x4` node (4× A40) into a dev/training box you can SSH into all
day, paying Duo **once**.

## Why this shape (the key constraint)
Delta's *interactive* partition caps at **1 hour**; only **batch** jobs get the **48h**
(2-day) max. NCSA also enables *"direct ssh to a compute node in a running job"*. So we
submit a 48h **batch "sentinel"** job that just holds a node, then ssh into it. SSH keys
are **disabled** on Delta (password+Duo every login), so a persistent **ControlMaster**
socket — opened once, kept warm 48h — is what avoids re-authenticating.

## User command
```bash
cd ~/hcrl_isaac_manager/scripts/cluster/cluster_dev
./cluster_dev.sh start          # approve ONE Duo push; everything after is non-interactive
```
`start` opens the SSH master (the only Duo prompt), mirrors the IsaacLab tree up, submits
the sentinel job, and launches a **background watcher** that tracks the (possibly
multi-hour) queue wait. You can walk away.

## Tracking / using it (no credentials needed once the master is up)
```bash
./cluster_dev.sh status         # job id / state / node / master+watcher health / live squeue
./cluster_dev.sh attach         # interactive shell on the compute node (once RUNNING)
./cluster_dev.sh exec -- <cmd>  # run <cmd> inside the Apptainer container on the node
./cluster_dev.sh sync           # re-mirror local IsaacLab edits → Delta
./cluster_dev.sh stop           # scancel the job + close the master
```
The watcher writes `~/.cluster_dev/state` (and `~/.cluster_dev/watch.log`); when
`JOB_STATE=RUNNING` and `NODE` is set, the box is ready. A Claude session can poll
`status` and drive `exec`/`attach` over the live master with zero auth.

## Config knobs (top of `cluster_dev.sh`; sourced from `../delta_config/.env.cluster`)
| var | default | note |
|---|---|---|
| `DELTA_ACCOUNT` | `bggq-delta-gpu` | **[VERIFY q1]** active allocation w/ SUs |
| `DELTA_PARTITION` | `gpuA40x4` | **[VERIFY q3]** 48h allowed on this acct |
| `DELTA_GPUS_PER_NODE` | `4` | **[VERIFY q2]** whole node; set `1` to save SUs |
| `DELTA_TIME` | `48:00:00` | batch max is 2 days |
| `DELTA_LOGIN_HOST` | round-robin | **[VERIFY q9]** pin `dt-login01…` if desired |
| `LOCAL_ISAACLAB_DIR` | resources/IsaacLab | code mirrored up |

## Answered (2026-05-20) + auto-handled
- **q1/q3** ✅ `bggq-delta-gpu` active; 48h confirmed.
- **q2** ✅ whole node (`--gpus-per-node=4`). Use all 4 in one process (robot_rl supports
  `--distributed` / `torch.distributed.run --nproc_per_node=4`) **or** run up to 4 separate
  single-GPU runs (set `CUDA_VISIBLE_DEVICES` per `exec`). 4 parallel experiments is the
  win here.
- **q4** — *auto-detected, no guess.* On job start the watcher probes `ssh login→node` with
  `BatchMode=yes` (fails fast instead of prompting). If it works passwordlessly → `attach`/
  `exec` use **direct ssh** (cleanest GPU access). If not → they use **`srun --overlap
  --jobid`** from the login node (NCSA-documented, needs no node ssh, no 2nd auth). Either
  way it works tomorrow. Force with `DELTA_ATTACH_MODE=ssh|srun` or `attach --ssh|--srun`.
- **q5** ✅ apptainer loaded by default (current batch workflow works) — no `module load`.
- **q6** ✅ SIF current — **do NOT add dependencies** (no new pip installs in node_exec.sh
  or training; editable `-e` code changes are fine, new third-party deps are not).
- **q7** ✅ assets ride the rsync (`resources/` is NOT excluded; first sync may be slow if
  not already on Delta).
- **q8** ✅ password + phone push → `start` prompts password then push, once.
- **q10** ✅ compute nodes have internet → live W&B works (no offline mode needed).
- **q11** ✅ local master host stays up.
- **q9** — using round-robin `login.delta…` (fine: we always multiplex through one master).

## Files
- `cluster_dev.sh` — control script (start/status/attach/exec/sync/stop + internal watcher).
- `sentinel.sbatch` — 48h node-holding job (envsubst template; does no heavy setup so a
  staging bug can't waste the allocation).
- `node_exec.sh` — runs on the node; stages SIF+caches+code once, then `apptainer exec`s
  (bind mounts mirror `docker/cluster/run_singularity.sh`). Reached via the synced
  `${CLUSTER_ISAACLAB_DIR}/docker/cluster/cluster_dev/` copy.
