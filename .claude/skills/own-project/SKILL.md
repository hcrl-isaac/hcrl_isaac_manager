---
name: own-project
description: Use this workspace with your own task repo (one outside the hcrl-isaac org, or not in the catalog), or port its infra (the Docker image, Ray job submission, SLURM/TACC cluster runs, the res lease manager, W&B artifact data) into a different codebase. Use when the user says "my own repo" or "my own project", asks to "port this infra", or wants to "run my code on Ray / TACC / a cluster".
---

# Your own project

Two paths. **A** keeps this workspace and adds your task repo to it: the least work, and every `pls` command works.
**B** copies pieces of the infra into a codebase that does not use `hcrl_isaaclab` at all. Ask which one the user
means; when they have Isaac Lab tasks, A is almost always the answer.

## Prerequisites (either path)

Ask for these; never guess or create them:

- **Repos:** read access to `hcrl_isaac_manager` (public), `hcrl_isaaclab` (private, ask its owners), `robot_rl` and
  `hcrl_robots` (public). Clones use `git@github.com:hcrl-isaac/<repo>.git`, so GitHub needs the user's SSH key.
  The catalog projects (`ssti`, `hhlm`, `mbrl`) are private and are not needed.
- **W&B:** an account and API key in `scripts/.env.wandb` (`pls deps` asks).
- **SLURM (e.g. TACC):** the user's own allocation and account, a login that reaches the cluster, and their own
  cluster profile (`pls setup --add <name>`). Profiles are per user; nobody else's works for them.
- **Ray:** network access to the cluster's dashboard (port 8265) and the job credentials (`RAY_USER_ID`), both from
  the cluster's admin.
- **An image** (Ray and SLURM): Docker and a registry account (e.g. Docker Hub) to build and push the shared image,
  plus Apptainer for a SLURM `.sif`, on a machine with room for it. Never build images on a shared box whose disk is
  nearly full (e.g. hcrl2).

## A. This workspace with your own task repo

1. **The repo.** `pls new <name>` scaffolds `resources/<name>_tasks/` (package `<name>_tasks`, an example T1 task,
   CPU smoke tests, CI). An existing repo works if it has the same shape: a top-level package named `<name>_tasks`
   (`hcrl_isaaclab.tasks_registry` discovers task packages by that suffix), a `pyproject.toml`/`setup.py`, and a
   `dependencies.yaml` listing the workspace repos it needs. Push it wherever the user keeps code.
2. **Register it** in the gitignored `workspace.yaml`, next to any catalog names:
   ```yaml
   projects:
     - {name: <name>, git: https://github.com/<you>/<name>_tasks.git, ref: main}
   isaaclab: {mode: source}
   ```
   It is cloned as `resources/<name>_tasks` from that URL; its `dependencies.yaml` resolves as usual. `refs:` pins
   still win over `ref`. `pls setup`'s picker keeps the entry (it shows with its URL).
3. **Install:** `pls setup` (or `pls resolve` to only fetch). A default project the user cannot read is left out with
   a warning when `workspace.yaml` is first written; a project they select but cannot read fails the fetch.
4. **CPU smoke:** `pls test <name>_tasks` (every registered id registers and its cfgs build).
5. **A core task first,** to separate install problems from task problems. The core's tasks need only
   `hcrl_isaaclab`, `robot_rl`, `hcrl_robots` and Isaac Lab:
   ```bash
   pls run train --task T1-Velocity-v0 --num_envs 64 --max_iterations 2 --logger tensorboard --video off
   ```
   (`G1-Velocity-v0` works too; the core's `tasks/*/*/__init__.py` register the rest.) Then the same for one of the
   user's tasks. Passing means a `Learning iteration 1/2` line; exit code 0 alone proves nothing.
6. **Remotely:** add compute with the `add-compute` skill, then `pls run --on <host:gpu | cluster | ray> -- train ...`
   (the `launch-run` skill). The repo ships with every run: ssh boxes and Ray take every resolved `*_tasks` repo
   (`scripts/worktree_env.py:workspace_repos`), SLURM stages every git repo under `resources/`. What the user adds:
   - **Their image** (Ray, SLURM), with their repo's Python deps: `pls docker-deps` regenerates
     `scripts/docker/requirements.workspace.txt` from the repos they have cloned, own repos included (the change
     stays local). Then `pls setup --on ray` builds the image and pushes it to their `RAY_IMAGE`; for SLURM,
     `pls docker build`, then `pls setup --on <cluster>` turns the local image into the `.sif` and pushes it (it
     reuses an existing local image, so rebuild that first when deps change).
   - **An ssh box's install root** built with their repo's deps:
     `scripts/isaac3/root/deploy.sh <user>@<host> <dir> <name>_tasks`, then its `remote_setup.sh` on the host.
   - **Large files** their tasks load (assets, datasets, policies): committed to their repos, and for Ray uploaded as
     their own W&B artifacts (`pls upload-artifacts`).

## B. Porting the infra into another codebase

Copy and adapt; each piece names what it assumes about this workspace. Read the files rather than retyping them.

| piece | what it does | files | coupled to this workspace |
| ----- | ------------ | ----- | ------------------------- |
| Image | Ubuntu 24.04 + Isaac Lab 3 from its uv lock + Kit; workspace code is **not** baked in but bound at job start and put on `PYTHONPATH` | `scripts/docker/Dockerfile`, `entrypoint.sh`, `docker_interface.sh`, `collect_workspace_deps.py` (+ `requirements.workspace.txt`) | the Isaac Lab ref/extras from `workspace.defaults.yaml`; entrypoint expects repos at `/workspace/ext/<name>`; deps list built from the catalog |
| Ray jobs | renders a job config, ships the repos as file mounts (large files excluded), submits through the dashboard | `scripts/ray/ray_interface.sh`, `submit_job.py`, `job_args.py`, `build_file_mounts.py`, `preflight.py`, `tools/*.template.yaml`, `scripts/ray/README.md` | `.env.ray` (`RAY_USER_ID`, `RAY_IMAGE`); the dashboard address from the `ray` pool in `scripts/cluster/res/compute.local.toml`; repo list from `worktree_env.py`; entry is `hcrl_isaaclab/scripts/train.py` |
| SLURM | per-user cluster profiles; builds the `.sif` from the image and pushes it; queued batch jobs, or a held "sentinel" node whose cards runs lease (`cluster_dev`), each run entering the container through `node_exec.sh` | `scripts/cluster/add_cluster.sh`, `config/<name>/` (made by `pls setup --add`), `cluster_interface.sh` (`pls setup --on`), `run_singularity.sh`, `cluster_dev/{cluster_dev.sh,node_exec.sh,trees.sh,sentinel.sbatch}`, `scripts/cluster/README.md`, `cluster_dev/README.md` | staged trees of `resources/`; the binds RTX rendering needs in Apptainer: a generated Vulkan ICD json, the driver's NGX library (DLSS), and the optional Vulkan clamp layer (`tools/install_vkclamp.sh`, `scripts/vulkan/`) — copy these binds or renders fail or come out speckled |
| `res` leases | one live view of every GPU across pools (local, ssh, slurm, ray) and file-based leases that say who holds which card; no daemon | `scripts/cluster/res/` (`res.py`, `probe.py`, `leases.py`, `inventory.py`, `compute.toml`, `README.md`) | pools from `compute.toml` + `compute.local.toml` + cluster profiles; `pls run --on` (`src/hcrl_cli/launch.py`, `res/evaluate.py`) launches on leased cards |
| W&B data | large files (robot assets, datasets, exported policies) as W&B artifacts that carry their own placement, fetched at run start and linked at the path the cfg names | `resources/hcrl_isaaclab/hcrl_isaaclab/utils/artifacts.py`, `.../scripts/tools/upload_artifacts.py`, `scripts/ray/README.md` ("Large-file resources") | paths relative to the workspace `resources/` dir; entity from `WANDB_USERNAME` (each user's own), project `hcrl_resources` |

Order that pays off: the image first (prove one container run trains), then one launcher (Ray or SLURM batch), then
leases only if several people or sessions share cards. What does not port: `pls` itself (`src/hcrl_cli/`) assumes
the manager layout (`resources/`, the `ilab` venv, `hcrl_isaaclab/scripts/`), so a ported codebase calls the scripts
above directly or writes its own thin wrapper.
