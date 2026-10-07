# Deploying to HPC Clusters

Utility scripts for running the workspace on SLURM clusters (e.g. NCSA Delta, TACC Stampede3) inside the shared
Apptainer image, following the workflow of the [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html).

## Getting started

1. Create your cluster profile:
   ```bash
   just cluster add [name]
   ```
   The prompts write `scripts/cluster/config/<name>/.env.cluster` and `submit_job_slurm.sh`. Profiles are per-user
   and gitignored; every cluster command snapshots a changed profile into `config/<name>/.backup/` and restores a
   submit script that a checkout deleted from there. Branches that have not merged main still use the old tracked
   profiles and overwrite yours on checkout; your copies stay in `.backup/` and come back on the next cluster command
   after you return. Use a large-quota filesystem for the workspace directory (e.g. `$WORK` on TACC; every job copies
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
| `setup` | build the `.sif` and rsync it to `CLUSTER_SIF_PATH`; with `CLUSTER_ARCH=arm64`, build it on the cluster instead |
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
| `CLUSTER_LOGS_DIR` | run logs and checkpoints (default: inside `CLUSTER_ISAACLAB_DIR`); put it on project/scratch storage, not a quota'd home |
| `CLUSTER_TREES_DIR` | staged code trees (default `CLUSTER_ISAACLAB_DIR/trees`); same advice |
| `CLUSTER_MIN_FREE_GB` | `develop stage` / `exec` refuse below this much free space (default 10; quota-aware where `quota -s` prints a table) |
| `CLUSTER_ARCH` | `amd64` (default) or `arm64` for aarch64 nodes (TACC Horizon's Grace GB200s); see below |
| `CLUSTER_BUILD_TIME` | wall time of the `arm64` build job (default `02:00:00`) |

## arm64 clusters (`CLUSTER_ARCH=arm64`)

The x86 docker image cannot run on aarch64 nodes, so `setup` builds the `.sif` on the cluster: it uploads
`scripts/cluster/hcrl-isaac.def` (the Dockerfile's steps as an Apptainer recipe; `tests/test_sif_recipe.py` keeps the
two in step) and runs `apptainer build --fakeroot` in a batch job with the profile's partition and account, waiting
for it. The job pulls the Dockerfile's Isaac Sim base once (`build-hcrl-isaac/isaac-sim-base.sif` next to the
`.sif`), and the new `.sif` replaces the old one only once it built and its torch passed the CUDA check. `build` and
`push` refuse for such a profile. Two things differ from the x86 image: `usd-core` (no aarch64 wheel) is left out, as
Kit's own USD serves, and torch is Isaac Sim's CUDA build (PyPI's aarch64 torch is CPU-only). The log of each build
is `build-hcrl-isaac/build-<job>.log`.
| `CLUSTER_VKCLAMP_DIR` | where the Vulkan clamp layer is installed; default `${CLUSTER_SIF_PATH}/vkclamp` |

**Rendering on newer drivers.** Drivers 595.71 (Delta) and 615.71 (Stampede3 RTX nodes) make Isaac Sim's RTX
renderer segfault at startup, so every `enable_cameras` run (video, cameras) dies; headless training is fine.
`scripts/cluster/tools/install_vkclamp.sh <cluster>` builds the clamp layer from `scripts/vulkan/` into
`CLUSTER_VKCLAMP_DIR`. `node_exec.sh` and `run_singularity.sh` bind it at `/opt/vkclamp` when it is present and
point the container's `XDG_CONFIG_DIRS` at it, with no image rebuild. `VKCLAMP_DISABLE=1` turns it off.

The job's resources (`-p`, `-A`, `-n`, `--cpus-per-task`, `--time`, mail) are the `#SBATCH` lines of
`submit_job_slurm.sh`; `develop` reuses them for its sentinel.
