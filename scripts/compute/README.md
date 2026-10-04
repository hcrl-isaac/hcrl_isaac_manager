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

A lease says who is using a card. There is no daemon: every `just res` call reconciles the leases with the live
probe, so a lease's state only changes when someone runs `res`.

```bash
just res claim mckennie:1 --holder "<session>" --note "T1 kick seed 3"   # host:gpu (host:job:gpu on SLURM)
just res claim --any --count 2 --min-free-gb 40 --pool larg --holder "<session>"
just res claim gpub065:2 --holder "<session>" --for 2h                   # time-boxed interactive work
just res leases                                                          # list (no probe)
just res release <id|host:gpu> --holder "<session>"                      # --force for someone else's
```

- A claim probes first and succeeds only on cards it sees as free that nobody holds; all named cards or none.
  `--any` fills partly used hosts first and skips Ray unless `--pool ray` is given (Ray schedules onto its cards).
- Activity renews a lease: a process of the holder's OS user on that pool, or a busy card whose process cannot be
  attributed (Ray). Another user's process does not renew it and shows `CONFLICT`; retained memory with no process
  and no utilization (a finished Isaac run) does not either and shows `held, no process`.
- Idle time runs from the first idle observation, so a lease is released only after two observations at least the
  window apart: `grace_min` before any activity was seen, `idle_min` after. Both default to 30 min (compute.toml
  `[leases]`), a design choice covering the ~15 min stall watchdog plus a relaunch and Kit boot. `--for` adds a
  hard end.
- A card that could not be read leaves its lease alone; a lease whose card is gone from a pool that probed cleanly
  (allocation ended, node removed) is released.
- The store is `~/.local/state/hcrl_res/leases.json`: one per OS user and machine (other users and other boxes see
  none of it), locked with `flock` (assumes a local filesystem; an NFS home may not honour it). A store that cannot
  be read is moved aside: `status` then shows no leases and `claim`/`release` refuse.

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
