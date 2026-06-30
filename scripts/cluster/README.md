# Deploying to HPC Clusters

This repo provides some utility scripts for deploying to HPC clusters, following a similar workflow as described in the [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html).

## Getting Started

To set up cluster configuration files, run
```bash
just add-cluster
```
and fill out the following prompts. This will create `.env.cluster` and `submit_job_slurm.sh` files in `scripts/cluster/<name>_config`. If no name is provided, it will be set to `default`.

You can then deploy the Isaac Lab image to the cluster with
```bash
just cluster <optional name>
```
If no name is specified, the `default` cluster will be used.

This performs the following steps:

1. Install necessary dependencies (Docker, Apptainer, nvidia-toolkit-container).
2. Build and start the Isaac Lab Docker container.
3. Compile the Docker container into an Apptainer image.
4. Tar and copy the image to the cluster.

## Cluster Interface

Cluster names can be specified with `CLUSTER=<name> just cluster <cmd>` (or `just cluster <name> <cmd>`). If no name is specified, it will be set to `default`.

### `just cluster job`

- Sends a training job to the cluster (runs `hcrl_isaaclab/scripts/train.py`).
- Can be followed by any arguments you'd like to pass to the script (e.g. `--task reach-v0`)
- Unlike with the Ray clusters, this copies *all* Isaac Lab code to the cluster, and will therefore include any changes made to Isaac Lab itself (not just the extensions)

### `just cluster push`

- Builds the Apptainer image from an existing Isaac Lab Docker image
    - This expects that the Docker image already exists. To create both the Docker and Apptainer image, use `just cluster`.
- Note that the Apptainer image only needs to be rebuilt if the Docker image changes (e.g. if updating top-level dependencies). Code changes are synced on job deployment.

### `just cluster repush`

- Pushes an *existing* tarred Apptainer image to the cluster.
- It can be used if the SSH request from the `push` command times out, or if you have an existing .sif image that you'd like to copy to a new cluster.
    - You can find the existing image in `resources/IsaacLab/docker/cluster/exports`.