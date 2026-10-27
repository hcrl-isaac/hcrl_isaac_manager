---
name: add-compute
description: Add a compute pool for `pls` (an SSH GPU box, a SLURM/HPC cluster, or a Ray cluster), or work out which kinds of compute the user has. Writes only the user's own gitignored config, then proves the pool probes with `pls res status`. Use when the user wants to train somewhere other than this machine.
---

# Add compute

`pls` sees compute as **pools**, each with a `kind` that picks how it is probed and launched on. Everything a user
adds is theirs: it goes in gitignored files, never in the tracked `compute.toml`. Ask for logins and hostnames;
never guess them, and never commit them.

## 1. Which kind is it?

Ask what the user has access to, then map each machine to a kind:

| kind | the user has | probed by | launched with |
| ---- | ------------ | --------- | ------------- |
| `local` | an NVIDIA GPU in this machine (always configured) | `nvidia-smi` here | `pls run -- ...` |
| `ssh` | GPU machines they can `ssh <user>@<host>` into with a key (lab boxes, a desktop) | `nvidia-smi` over ssh | `pls run --on <host>:<gpu> -- ...` |
| `slurm` | an HPC account: a login node with `sbatch`/`squeue`, an allocation, often 2FA | their jobs + `nvidia-smi` in an `srun` step | `pls run --on <cluster> --batch -- ...`, or `pls res hold` a node, then `pls run --on <cluster> -- ...` |
| `ray` | a Ray cluster's dashboard address (and a Docker image the workers pull) | the Ray dashboard API | `pls run --on ray -- ...` |

Quick tests when unsure: `ssh <host> nvidia-smi` works with no password → `ssh`. `ssh <login> sinfo` lists
partitions → `slurm`. A `http://<host>:8265` dashboard → `ray`. A machine reachable only through a scheduler is
`slurm`, even if you can ssh to its login node.

## 2. Add it

**ssh.** Add a pool to `scripts/cluster/res/compute.local.toml` (create it if missing; it is merged over
`compute.toml`):

```toml
[compute.<pool-name>]
kind = "ssh"
hosts = ["<host1>", "<host2>"]   # short names
domain = "<domain>"              # appended to each host; omit if the short name resolves
user = "<login>"
# pin = "device"                 # only if Kit refuses a masked CUDA_VISIBLE_DEVICES on these boxes
```

Each host needs key-based ssh (`ssh -o BatchMode=yes <user>@<host>.<domain> true` must succeed), a driver for
CUDA 12.8, and an Isaac Lab 3 install root for `pls run --root <dir>`: `scripts/isaac3/root/deploy.sh <host>
<dir>`, then `remote_setup.sh` on the host (usage in each script's header).

**slurm.** Run `pls setup --add <cluster-name>`. It asks for the login host, account, partition, GPU type and
scratch dirs and writes `scripts/cluster/config/<name>/` (gitignored). Then `pls connect --on <name>` (2FA once),
and `pls setup --on <name>` to build and push the cluster's `.sif` image. Details: `scripts/cluster/README.md`.

**ray.** Copy `scripts/ray/tools/.env.ray.template` to `scripts/ray/.env.ray` and fill it in, then
`pls setup --on ray`. The dashboard address goes in the pool, so `pls res` and `pls run --on ray` find it:

```toml
[compute.ray]
kind = "ray"
address = "http://<head>:8265"
```

Large assets reach Ray workers as W&B artifacts: `pls upload-artifacts --all` once, and again when they change.
Details: `scripts/ray/README.md`.

## 3. Prove it

```bash
pls res pools                      # the new pool is listed
pls res status --pool <pool-name>  # every card reads free/busy/held, not UNKNOWN
```

`UNKNOWN` means the probe failed: an ssh key or login problem, no SSH master for a SLURM cluster
(`pls connect --on <name>`), or a wrong address. Then run a 2-iteration smoke there before anything real:
`pls run --on <target> -- train --task <id> --num_envs 64 --max_iterations 2 --logger tensorboard --video off`.

To hide a pool without deleting it, set `enabled = false` on it in `compute.local.toml`.
