# Compute resources (`just res`)

One view of every GPU we can use, probed live from each pool rather than declared on a board.

```bash
just res                      # = just res status: every card on every pool
just res status --free        # only free cards
just res status --pool larg   # pools whose name starts with "larg" (repeatable)
just res status --json        # machine-readable
just res pools                # the configured pools
just res claim / release / leases   # card leases, below
just res eval <script> ...    # run a one-off script on a leased card, below
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
just res release <id|host:gpu|host:job:gpu> --holder "<session>"         # --force for someone else's
just res transfer <id|host:gpu> --holder "<you>" --to "<session>"        # hand a lease over (--force: not yours)
just res claim gpub065:0 --adopt --run ni2cb9af --holder "<session>"     # take over a busy card's running run
```

- A handover moves the lease (`transfer`, which records the previous holder) instead of releasing and reclaiming:
  a released card that is still running reads `busy`, and a plain claim refuses it. `--adopt` is for a busy card
  whose run you take on when nobody leases it any more; it needs named cards and refuses leased or unknown ones. An
  adopted lease starts active, so its run renews it and the run's end releases it as usual. `--run` records the run
  a lease covers.

- A claim probes first and succeeds only on cards it sees as free that nobody holds; all named cards or none.
  `--any` fills partly used hosts first and skips Ray unless `--pool ray` is given (Ray schedules onto its cards).
- Activity renews a lease: a process of the holder's OS user on that pool, or a busy card whose processes cannot be
  attributed (Ray lists them as `?`). If every attributable process belongs to another user the lease is not renewed
  and shows `CONFLICT`, even when unattributed processes share the card; retained memory with no process and no
  utilization (a finished Isaac run) does not renew it either and shows `held, no process`.
- Idle time runs from the first idle observation, so a lease is released only after two observations at least the
  window apart: `grace_min` before any activity was seen, `idle_min` after. Both default to 30 min (compute.toml
  `[leases]`), a design choice covering the ~15 min stall watchdog plus a relaunch and Kit boot. `--for` adds a
  hard end.
- A card that could not be read leaves its lease alone, and so does a partial probe: a Ray node that is not ALIVE,
  an unparsable `squeue` line, or a SLURM job that is not RUNNING (e.g. COMPLETING) makes its cards unknown. A lease
  whose card is gone from a complete probe (allocation ended, node removed) is released only when it is still gone
  on a complete probe at least `idle_min` later.
- The store is `~/.local/state/hcrl_res/leases.json`, one per home directory: other OS users see none of it, and
  machines that share a home (the LARG boxes' NFS home) share it. It is locked with `flock`, which an NFS mount may
  not honour, so run `res` from one machine. A store that cannot be read or has wrong-typed fields is moved aside:
  `status` then shows no leases and `claim`/`release`/`leases` refuse.

## One-off scripts (`just res eval`)

Run an eval or analysis script on a leased card of a `local` or `ssh` pool, with its checkpoints brought along:

```bash
just res eval path/to/census.py --any --pool larg-a100 --holder "<session>" \
    --checkpoint REORIENT_CKPT=hcrl-ssti/Crab_Agile/d1tafrx7@5999 \
    --checkpoint TRAVERSE_CKPT=https://wandb.ai/hcrl-ssti/Crab_Agile/runs/d1tafrx7 \
    --env CENSUS_N=512 -- --script-flag value
just res eval probe.py --on hazard:2 --holder "<session>" --checkpoint ./model_200.pt   # exported as CHECKPOINT
just res eval probe.py --lease <id> --holder "<session>" --wt my-feature                 # your lease, a worktree set
```

- The card comes from `--on host:gpu`, `--any` or `--lease <id>`. A lease `eval` takes is released when the script
  ends, success or failure; a `--lease` you pass stays yours. `ray` and `slurm` pools are refused (use
  `just ray job` or a dev-node `develop exec`).
- `--checkpoint [NAME=]<ref>` (repeatable) exports the checkpoint's path on the target as `NAME` (default
  `CHECKPOINT`). A ref is a local path, a W&B run URL or `entity/project/run_id`, with `@<iter>` for
  `model_<iter>.pt` (default: the run's latest). W&B checkpoints download on this machine through the same code as
  `--load_run` (cached in `~/.cache/hcrl_res/checkpoints`), so a checkpoint is no longer tied to the box that
  trained it.
- The script runs this machine's code: the package repos (`hcrl_isaaclab`, `robot_rl`, `*_tasks`; a `--wt` worktree
  set where one exists) go on `PYTHONPATH`. A local pool imports the checkouts in place. On an ssh pool each repo
  becomes an immutable snapshot `<scratch>/res-eval/code/<repo>-<fingerprint>` (tracked and non-ignored files, written
  under a `.partial` name and renamed into place, hardlinked from the previous snapshot), and the run gets a fresh
  `<stage>/resources/` of links to those snapshots and to the target's own asset repos. A changed or deleted file
  makes a new snapshot, and nothing is ever synced into the target's shared checkout.
- The interpreter is the target workspace's `ilab` python, with a per-run `TMPDIR`, per-card `XDG_CACHE_HOME` /
  `OMNI_CACHE_DIR` and `HCRL_ARTIFACT_ROOT` under `<scratch>/res-eval/`. The card: pools with `pin = "cvd"`
  (default) mask `CUDA_VISIBLE_DEVICES` to it; `pin = "device"` (the A40 pool, where Kit has refused masked GPUs)
  leaves it unmasked. Either way `RES_EVAL_DEVICE` names the card the script should use (`cuda:0` when masked).
- `--env KEY=VALUE` (repeatable) and the W&B credentials from `scripts/.env.wandb` reach the script through a 0600 env
  file, never the command line.
- The run starts in its own process group. `--timeout` (default none) and `--stall` (no output for 15 min; `0` turns
  it off) kill the whole group and confirm it is gone; so does Ctrl-C. The exit status is the script's own, an exit 0
  after a Python traceback counts as a failure, and a killed run reports 124.
- Everything goes to `<stage>/log`, which starts with a MANIFEST of each repo's commit and dirty count. The stage is
  removed after success; a failed run keeps its log, MANIFEST and run.sh (its checkpoints and credentials are always
  removed). A lease `eval` took is released whatever happens. Old snapshots are not pruned yet.
- Pool settings: `workspace` (the manager checkout on the target, providing the venv and the asset repos; default
  `/var/local/<user>/hcrl_isaac_manager` on ssh pools, this machine's checkout locally), `scratch` (default
  `/var/local/<user>`, `~/tmp` locally) and `pin`. Keep `scratch` on local disk, never a quota'd NFS home: snapshots,
  checkpoints and Kit caches live there.

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
