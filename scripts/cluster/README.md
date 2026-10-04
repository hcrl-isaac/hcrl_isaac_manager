# Deploying to HPC Clusters

Utility scripts for running the workspace on SLURM clusters (e.g. NCSA Delta, TACC Stampede3) inside the shared
Apptainer image, following the workflow of the [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html).

## Getting started

1. Create your cluster profile:
   ```bash
   just cluster add [name]
   ```
   The prompts write `scripts/cluster/config/<name>/.env.cluster` and `submit_job_slurm.sh`. Profiles are per-user
   and gitignored; every cluster command snapshots a changed profile into `config/<name>/.backup/`, restores a
   submit script that a checkout deleted from there, and refuses to run on a branch that still tracks the profiles
   (merge main into it first). Use a large-quota filesystem for the workspace directory (e.g. `$WORK` on TACC; every job copies
   the workspace there) and `$SCRATCH` for the `.sif` and Isaac Sim cache. For a `*.tacc.utexas.edu` login the
   profile also loads TACC's `tacc-apptainer` module.
2. Regenerate a profile after a template change, keeping your values as defaults (old files are backed up and the
   diff is printed):
   ```bash
   just cluster add --update <name>
   ```
3. Create `scripts/.env.wandb` from `scripts/tools/.env.wandb.template`; `just cluster job` refuses to run without it.
4. Build the `.sif` from the shared docker image and push it to the cluster (only needed when the image changes):
   ```bash
   just cluster <name> setup
   ```

## Commands

`CLUSTER=<name> just cluster <cmd>` and `just cluster <name> <cmd>` are equivalent.

| command | what it does |
|---|---|
| `add [--update] [name]` | create or regenerate your profile |
| `setup` | build the `.sif` and rsync it to `CLUSTER_SIF_PATH` |
| `build` | build the `.sif` only (into `scripts/cluster/exports/`) |
| `push` / `repush` | rsync an already built `.sif` (reuses the SSH master) |
| `job [args]` | copy the workspace to a timestamped dir under `CLUSTER_ISAACLAB_DIR`, then `sbatch` `scripts/train.py [args]` |
| `develop ...` | persistent dev node: see [cluster_dev/README.md](cluster_dev/README.md) |

Code changes ride each job's copy, so the `.sif` only needs a rebuild when dependencies change. The copy is removed
when the job ends (`REMOVE_CODE_COPY_AFTER_JOB`); training logs and the job's `slurm-<id>.out` end up in `CLUSTER_ISAACLAB_DIR/logs`, outside it.

## Profile settings (`.env.cluster`)

| var | note |
|---|---|
| `CLUSTER_LOGIN` | `user@login-host` |
| `CLUSTER_ISAACLAB_DIR` | workspace on the cluster (ends in `isaaclab`); job copies and logs |
| `CLUSTER_SIF_PATH`, `CLUSTER_ISAAC_SIM_CACHE_DIR` | scratch locations of the `.sif` and the Isaac Sim cache |
| `CLUSTER_PYTHON_EXECUTABLE` | script (plus fixed args) a `job` runs, e.g. `scripts/train.py` or a `torch.distributed.run` line |
| `CLUSTER_APPTAINER_FLAGS` | extra `apptainer exec` flags; default `--fakeroot`, which TACC needs to read the image's `/isaac-sim` |
| `CLUSTER_MODULE_LOAD` | Lmod module(s) providing apptainer on compute nodes (TACC: `tacc-apptainer`); blank if it is on `PATH` |
| `OMP_NUM_THREADS` | threads per process |

The job's resources (`-p`, `-A`, `-n`, `--cpus-per-task`, `--time`, mail) are the `#SBATCH` lines of
`submit_job_slurm.sh`; `develop` reuses them for its sentinel.
