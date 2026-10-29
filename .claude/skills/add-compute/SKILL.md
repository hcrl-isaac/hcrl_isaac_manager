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

**ssh.** One pool per set of boxes sharing a login and domain:

```bash
pls setup --add <pool-name> --kind ssh --hosts <host1>,<host2> --user <login> [--domain <domain>] [--pin device] [--render]
  [--min-free-disk-gb N] [--max-hours H] [--desktop]
```

`--domain` is appended to each short host name (omit it if the short name resolves); `--pin device` only if Kit
refuses a masked CUDA_VISIBLE_DEVICES on these boxes; `--render` for RT-core cards (A40, RTX PRO, 5090), whose
`scripts/larg/train.sh` runs record video in-process; on a box its owner also uses (a desktop), `--min-free-disk-gb`,
`--max-hours` and `--desktop` set the guardrails in `scripts/cluster/res/README.md`; `--update` replaces an
existing pool. It writes the pool into
`scripts/cluster/res/compute.local.toml` (gitignored, merged over `compute.toml`; the main checkout's serves every
worktree), and editing that file by hand is equivalent:

```toml
[compute.<pool-name>]
kind = "ssh"
hosts = ["<host1>", "<host2>"]
domain = "<domain>"
user = "<login>"
```

Each host needs key-based ssh (`ssh -o BatchMode=yes <user>@<host>.<domain> true` must succeed), a driver for
CUDA 12.8, and an Isaac Lab 3 install root for `pls run --root <dir>`: `scripts/isaac3/root/deploy.sh
<user>@<host> <dir>` copies the minimal trees, then `bash <dir>/resources/isaac3/root/remote_setup.sh <dir>` on the
host builds its venv (usage in each script's header; `<dir>` on local disk, e.g. `/var/local/<login>/isaac3`).
`scripts/larg/README.md` ("A new ssh box") has the whole sequence.

**slurm.** Run `pls setup --add <cluster-name>`. It asks for the login host, account, partition, GPU type and
scratch dirs and writes `scripts/cluster/config/<name>/` (gitignored). Then `pls connect --on <name>` (2FA once),
and `pls setup --on <name>` to build and push the cluster's `.sif` image. Details: `scripts/cluster/README.md`.

**ray.** The dashboard address goes in the pool named `ray`, where `pls res`, `pls run --on ray` and
`pls jobs --on ray` all read it:

```bash
pls setup --add ray --kind ray --address http://<head>:8265
```

Then `pls setup --on ray` writes `scripts/ray/.env.ray` (asking for `RAY_USER_ID`, the id the user's jobs are tagged
with, and `RAY_IMAGE`, the `<registry user>/<name>` image the workers pull) and builds and pushes that image.

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
