# Compute resources (`just res`)

One view of every GPU we can use, probed live from each pool rather than declared on a board.

```bash
just res                      # = just res status: every card on every pool
just res status --free        # only free cards
just res status --pool larg   # pools whose name starts with "larg" (repeatable)
just res status --json        # machine-readable
just res pools                # the configured pools
just res claim / release / leases   # card leases, below
```

Card states:

- `free`; `leased` (free, but someone holds a lease on it)
- `busy`: a process is on it; owner, command and elapsed time are shown
- `held`: memory in use (>= 1 GiB; idle cards read 0-545 MiB) or utilization with no visible process, e.g. Isaac
  kept its VRAM after exit
- `UNKNOWN`: the pool could not be probed (SSH master down, nvidia-smi failed or returned partial output). Unknown
  is never free; `status` exits 2 when no pool could be probed.

## Leases

A lease says who is using a card. It needs no daemon and no upkeep: every `just res` call reconciles the leases
against the live probe, and a lease ends on its own.

```bash
just res claim mckennie:1 --holder "<session>" --note "T1 kick seed 3"   # name cards as host:gpu
just res claim --any --count 2 --min-free-gb 40 --pool larg --holder "<session>"
just res claim gpub065:2 --holder "<session>" --for 2h                   # time-boxed interactive work
just res leases                                                          # list (no probe)
just res release <id|host:gpu>
```

- A claim only succeeds on a card the probe sees as free and nobody holds.
- A new lease must show activity (memory or a process on the card) within 20 min, after which it is released
  once the card has been idle for 15 min. `--for` adds a hard end.
- Leases on a pool that cannot be probed are neither renewed nor released until it can be.
- The store is `~/.local/state/hcrl_res/leases.json` on the machine the sessions run on, written under a lock.

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
