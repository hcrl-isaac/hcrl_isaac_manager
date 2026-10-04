# Compute resources (`just res`)

One view of every GPU we can use, probed live from each pool rather than declared on a board.

```bash
just res                      # = just res status: every card on every pool
just res status --free        # only free cards
just res status --pool larg   # pools whose name starts with "larg" (repeatable)
just res status --json        # machine-readable
just res pools                # the configured pools
```

Card states: `free`, `busy` (a process is on it; owner, command and elapsed time are shown), `held` (memory in use
or utilization with no visible process, e.g. Isaac kept its VRAM after exit; idle cards read 0-545 MiB, so the
bound is 1 GiB) and `UNKNOWN` (the pool could not be probed, e.g. its
SSH master is down, or nvidia-smi failed or returned partial output). Unknown is never free; `status` exits 2 when
no pool could be probed.

## Pools

Every compute pool has a `kind`, which picks the backend that probes it:

| kind | probes | settings |
|---|---|---|
| `local` | `nvidia-smi` on this machine | |
| `ssh` | `nvidia-smi` on each host over ssh (keys, BatchMode) | `hosts`, `domain`, `user` |
| `slurm` | your jobs (`squeue`) and, per running job, `nvidia-smi` in an `srun --overlap` step over the cluster's SSH master | from the cluster profile |
| `ray` | the Ray dashboard API (nodes, GPUs, running jobs) | `address` |

`compute.toml` lists the lab's pools (`local` is the machine you run on). Put per-user settings (your LARG login,
extra pools, `enabled = false` to hide one) in `compute.local.toml` (gitignored), which is merged over it. The LARG
pools need a user (there or in `$LARG_USER`); without one they read UNKNOWN with a hint:

```toml
[compute.larg-a100]
user = "<your LARG login>"
```

SLURM pools come from your cluster profiles (`scripts/cluster/config/<name>/`); profiles that share a login are
probed once. Adding compute of an existing kind is a new `[compute.<name>]` entry; a new kind is a new probe function
in `probe.py`.
