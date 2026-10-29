# Compute resources (`pls res`)

One view of every GPU we can use, probed live from each pool rather than declared on a board.

```bash
pls res                      # = pls res status: every card on every pool
pls res status --free        # only free cards
pls res status --pool lab    # pools whose name starts with "lab" (repeatable)
pls res status --json        # machine-readable
pls res pools                # the configured pools
pls res claim / release / leases   # card leases, below
pls run --on <card> --holder <you> -- <script> ...   # a one-off script on a leased card, below
```

Card states:

- `free`; `leased` (free, but someone holds a lease on it)
- `busy`: a process is on it; owner, command and elapsed time are shown
- `held`: memory in use (>= 1 GiB; idle cards read 0-545 MiB) or utilization with no visible process, e.g. Isaac
  kept its VRAM after exit
- `UNKNOWN`: the pool could not be probed (SSH master down, nvidia-smi failed or returned partial output). Unknown
  is never free; `status` exits 2 when no pool could be probed.

## Leases

A lease says who is using a card. There is no daemon: every `pls res` call reconciles the leases with the live
probe, so a lease's state only changes when someone runs `res`.

```bash
pls res claim box1:1 --holder "<session>" --note "T1 kick seed 3"   # host:gpu (host:job:gpu on SLURM)
pls res claim --any --count 2 --min-free-gb 40 --pool lab --holder "<session>"
pls res claim gpub065:2 --holder "<session>" --for 2h                   # time-boxed interactive work
pls res leases                                                          # list (no probe)
pls res release <id|host:gpu|host:job:gpu> --holder "<session>"         # --force for someone else's
pls res transfer <id|host:gpu> --holder "<you>" --to "<session>"        # hand a lease over (--force: not yours)
pls res claim gpub065:0 --adopt --run ni2cb9af --holder "<session>"     # take over a busy card's running run
```

- A handover moves the lease (`transfer`, which records the previous holder) instead of releasing and reclaiming:
  a released card that is still running reads `busy`, and a plain claim refuses it. `--adopt` is for a busy card
  whose run you take on when nobody leases it any more; it needs named cards and refuses leased or unknown ones. An
  adopted lease starts active, so its run renews it and the run's end releases it as usual. `--run` records the run
  a lease covers.

- A claim probes first and succeeds only on cards it sees as free that nobody holds; all named cards or none.
  `--any` fills partly used hosts first and skips Ray unless `--pool ray` is given (Ray schedules onto its cards).
- `--wait [limit]` queues instead of refusing: while the cards are leased or busy (or too few `--any` matches are
  free), the claim probes again every minute and leases them once they free, giving up after the limit (e.g.
  `--wait 2h`; bare `--wait` has none). A card that does not exist is still refused at once.
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
  machines that share a home (e.g. boxes with one NFS home) share it. It is locked with `flock`, which an NFS mount may
  not honour, so run `res` from one machine. A store that cannot be read or has wrong-typed fields is moved aside:
  `status` then shows no leases and `claim`/`release`/`leases` refuse.

## Priority (reservations)

Priority is a reservation in `res`, not a promise to remember: it never leaves cards idle. Others may lease reserved
cards meanwhile; their leases *yield*, and move to the priority holder when it claims the cards.

```bash
pls res reserve --pool amd-rtx --count 8 --eta 10m-60m --holder "<session>" --note "BFM-Zero retrain (Emily)"
pls res reserve c571-003:0 c571-003:1 --eta 2h --holder "<session>"   # named cards instead of N of a pool
pls res reserve --pool amd-rtx --count 8 --eta now --holder "<session>"   # same holder and scope: updates the ETA
pls res claim --any --pool amd-rtx --count 8 --holder "<session>"       # takes free reserved cards, then yielding ones
pls res reserve --end [<id>] --holder "<session>"                          # end it early
```

- `--eta` is when the cards are needed: `now`, `30m`, or a window such as `10m-60m`. `--for` caps the priority
  (default 24h); it also ends once its holder has claimed cards under it and holds none of them again, i.e. the
  prioritized run is done and released (or went idle). `status` and `leases` show each reservation, the
  ETA, and the leases that yield to it.
- Reserving marks the leases already on those cards as yielding and prints their holders. **The priority session
  messages each of them with the ETA**, and again when it moves the ETA: `res` cannot message a session itself.
- A claim by anyone else on a reserved card succeeds, warns that it yields and when, and marks the lease. `--any`
  takes unreserved cards first. A `pls run --on` job whose lease yields prints `[res] YIELD ...` in its output, and
  `[res] PREEMPTED ...` once the card moved; a holder checkpoints when warned and stops its run when preempted.
- The priority holder's claim takes free reserved cards first, then yielding ones even while their runs are busy: the
  lease moves to it (`took over (from <holder>)`) and `res` prints whom to tell to stop now. `res` never stops another
  holder's processes; stopping is the holder's job. The old run usually still holds the card right after a
  takeover, so wait until `pls res status` shows it drained before launching on it.
- A pool reservation covers every card of that pool. On a SLURM login shared by several profiles (amd-rtx and
  rtx-small on Stampede3) the probe cannot tell which profile a job came from, so it covers every job on that login;
  name the cards when that matters.
- Reservations live in `reservations.json` beside the lease store, under the same lock.

## One-off scripts on a card (`pls run --on`)

`pls run --on <card>` runs an eval or analysis script on a leased card of a `local` or `ssh` pool, or of a held SLURM
dev sentinel, with its checkpoints brought along (`pls run --help` for the whole grammar, `--on any --help` for the
card options):

```bash
pls run --on lab-a100 --holder "<session>" \
    --checkpoint REORIENT_CKPT=<entity>/<project>/<run_id>@5999 \
    --checkpoint TRAVERSE_CKPT=https://wandb.ai/<entity>/<project>/runs/<run_id> \
    --env CENSUS_N=512 -- path/to/census.py --script-flag value                       # any free card of a pool
pls run --on box1:2 --holder "<session>" --checkpoint ./model_200.pt -- probe.py     # exported as CHECKPOINT
pls run --on lease:<id> --holder "<session>" --wt my-feature -- probe.py               # your lease, a worktree set
pls run --on box3:1 --wt legacy --holder "<session>" --detach \
    -- train --task hhlm/T1-Kick-v0 --headless --video on                              # train a branch on an ssh box
pls run --on local:0 --holder "<session>" -- probe.py                                  # this machine's card 0
pls run --on c571-003:3 --holder "<session>" --wt my-feature --detach -- census.py     # a sentinel's free card
                                                     # (c571-003:3557743:3 when the node runs several of your jobs)
pls run --on any --holder "<session>" --cmd -- nvidia-smi                              # a command, not a script
```

- A script is `<name>` (hcrl_isaaclab/scripts/<name>.py), `<repo>/<path>` or `<repo>:<path>` inside a shipped repo,
  which runs from its shipped copy so its own sibling imports resolve, or a local file (copied to the target). It runs
  in a writable working dir of its own, `<scratch>/res-eval/work/<stage id>`, where relative outputs such as `logs/`
  land and stay. `--cmd` runs a command line instead, from the box workspace (the work dir on SLURM).
- On a remote card every package repo ships: a `--wt` repo as it is on disk, any other at the main checkout's HEAD
  commit (its uncommitted edits stay here, and the target's own copy, which can be stale, is never used). The
  MANIFEST gives each one's commit. On an ssh card they are read-only snapshots under `<scratch>/res-eval/code/`;
  output a script writes inside one (`logs/`, `outputs/`, `wandb/`, e.g. video_logger's checkpoints and videos) lands
  in the box workspace's copy of that repo.
- A `--wt` worktree with uncommitted changes is refused before a lease, since its recorded commit would not restage
  the code that ran; `--allow-dirty` ships it anyway and says so in the MANIFEST. A run whose working dir is its own
  (a repo script, or any SLURM run) also gets a `config-defaults.yaml` there, which `wandb.init` loads, so its W&B
  config holds `pls_refs`: each repo's full commit and dirty count, the staged tree (`--tree` reruns it), `--wt`,
  `--root` and the MANIFEST lines.
- `--root DIR` runs the script with the venv of an Isaac Lab install root (`<DIR>/venv/bin/python`) instead of ilab,
  e.g. an Isaac Lab 3 root: `pls run --on <host>:<gpu> --root <isaac-lab-root> --wt <set> --holder <you> -- <script>`.
  The code ships as for any run (`--wt` picks the worktree set, so pick 3.x-ported branches), never the root's own
  `resources/` copies. On a box card PATH is `/usr/bin:/bin` behind the venv, since an unreadable dir on a shared
  default PATH breaks `torch.compile`. On a SLURM card the step runs the venv in a stock Debian bookworm image,
  `<DIR>/bookworm.sif` (pulled on first use; the 3.x wheels need glibc >= 2.35, newer than the clusters' hosts), with
  the same `/workspace/ext` mounts, stop file and heartbeat as any SLURM run, and the staged package repos on
  PYTHONPATH. Not with `--venv`.
- `--detach` starts the run in its own session and returns, printing how to follow its log and stop it. The run
  owns its stage dir, which its lease records, and the lease stays held until the card goes idle. `--timeout`
  and `--stall` do not apply to a detached run.
- `pls run --stop <lease> --holder <you>` stops a detached run: the process group (a SLURM run's `stop` file), then,
  once it is gone, removes the stage but its log and releases the lease. A run that may still be alive keeps the
  lease. A lease that records no run (one taken before `--stop` existed) is just released.

- The card comes from `--on host:gpu` (`local:<gpu>` for this machine; `host:job:gpu` when a SLURM node runs
  several jobs), `--on any`, `--on <pool>` or `--on lease:<id>`. A busy card is refused (`--wait [limit]` waits for
  it, or for an `--on any` match, to free, as `pls res claim --wait` does); `pls res claim --adopt` takes over the run
  on it. A lease the run takes is released when the script ends, success or failure; a `lease:<id>` you pass stays
  yours. `any` stays on the local and ssh pools; name a cluster pool to use its sentinel. Ray is its own target,
  `--on ray`.
- On a SLURM card (a running job of yours, normally a held dev sentinel) the script runs in the container through a
  step on that job (`cluster_dev.sh exec`), with the job's own partition, account and GPU request. The stage dir is
  `<cluster checkout>/artifacts/res-eval/<id>`, which the container sees as `/workspace/artifacts/res-eval/<id>`; the
  package repos are staged as a code tree (`pls stage`), as above; asset repos come from the cluster's
  newest `default` tree. The card is pinned by its GPU
  UUID, and a step that cannot see it fails with status 98 (on Delta a step holding the job's GPUs can starve
  another). The container's `/tmp` is the job's node-local dir, which every step of the sentinel shares, so the run
  gets its own `TMPDIR` there (removed when it ends) and per-card Kit caches beside it (gone with the job). A script
  runs from the per-run work dir `artifacts/res-eval/work/<id>`.
- The container hides a SLURM run's processes from the node, so it reports through files in its stage dir:
  `heartbeat` (touched every 2 s while it runs), `stop` (create it to end the run: TERM, then KILL after 10 s) and
  `status` (its exit status, written once the card's processes are gone or after 60 s; a Python traceback with exit
  0 counts as 1). `--detach` waits for the first heartbeat, prints the follow command over the SSH master and
  the `pls run --stop` command, and logs the step's wrapper to `<stage>/wrapper.log`. A detached run that does not start within 20 min
  (a job's first step copies the .sif to the node) is stopped and its credentials removed; a run whose `stop`
  exists before it starts exits 130 without launching. Each new SLURM stage prunes earlier ones: credentials of
  runs that never started after 30 min, and stages and work dirs older than 7 days whose run is not alive;
  `res-eval` code trees unused for 14 days (reuse counts) that no running step marks are removed when one is staged.
- From a manager worktree, `res`, `pls stage`, `pls shell` and `pls run --on <cluster>` use the main checkout's cluster profiles (a
  worktree has none of its own); `pls res hold`/`drop`, `pls disconnect` and `pls jobs ... kill` refuse there, since they change the shared sentinel
  state.
- `--checkpoint [NAME=]<ref>` (repeatable) exports the checkpoint's path on the target as `NAME` (default
  `CHECKPOINT`). A ref is a local path, a W&B run URL or `entity/project/run_id`, with `@<iter>` for
  `model_<iter>.pt` (default: the run's latest). W&B checkpoints download on this machine through the same code as
  `--load_run` (cached in `~/.cache/hcrl_res/checkpoints`), so a checkpoint is no longer tied to the box that
  trained it. A script that takes the path as an argument gets it through `{NAME}` in its arguments, e.g.
  `--checkpoint CKPT=T1_Balance/8ufa90cl -- --checkpoint '{CKPT}'`; braces naming no checkpoint stay as written.
- The script runs this machine's code: the package repos (`hcrl_isaaclab`, `robot_rl`, `*_tasks`; a `--wt` worktree
  set where one exists) go on `PYTHONPATH`. A set's `holosoma` worktree ships too (its main checkout never does:
  without one the target's own install is imported), imported from `src/holosoma_retargeting` rather than the repo
  root, so `$PLS_RESOURCES/holosoma/src/holosoma_retargeting/<script>.py` or `holosoma:src/...` runs that branch.
  A local pool imports the checkouts in place. On an ssh pool each repo
  becomes an immutable snapshot `<scratch>/res-eval/code/<repo>-<fingerprint>` (tracked and non-ignored files, written
  under a `.partial` name and renamed into place, hardlinked from the previous snapshot), and the run gets a fresh
  `<stage>/resources/` of links to those snapshots and to the target's own asset repos. A changed or deleted file
  makes a new snapshot, and nothing is ever synced into the target's shared checkout.
- The interpreter is the target workspace's `ilab` python, with a per-run `TMPDIR`, per-card `XDG_CACHE_HOME` /
  `OMNI_CACHE_DIR` and `HCRL_ARTIFACT_ROOT` under `<scratch>/res-eval/`. The card: pools with `pin = "cvd"`
  (default) mask `CUDA_VISIBLE_DEVICES` to it; `pin = "device"` (the A40 pool, where Kit has refused masked GPUs)
  leaves it unmasked. Either way `RES_EVAL_DEVICE` names the card the script should use (`cuda:0` when masked). Write
  `{DEVICE}` in the script's arguments (`--device {DEVICE}`, `agent.device={DEVICE}`) and `pls run` fills it in for
  that card. A hand-written `cuda:N` that is not the card is refused before anything is staged: on a `cvd` pool it is
  an invalid ordinal, and on a `device` pool it silently runs on a card you do not hold.
- `--env KEY=VALUE` (repeatable) and the W&B credentials from `scripts/.env.wandb` reach the script through a 0600 env
  file, never the command line.
- The run starts in its own process group. `--timeout` (default none) and `--stall` (no output for 15 min; `0` turns
  it off) kill the whole group and confirm it is gone; so does Ctrl-C. The exit status is the script's own, an exit 0
  after a Python traceback counts as a failure, and a killed run reports 124.
- Everything goes to `<stage>/log`, which starts with a MANIFEST of each repo's commit and dirty count. The stage is
  removed after success; a failed run keeps its log, MANIFEST and run.sh (its checkpoints and credentials are always
  removed). A lease the run took is released whatever happens. Old snapshots are not pruned yet.
- `--cpus N` runs on N cores of an ssh box and leases no card, for a job that needs no GPU (`--on <host>`):
  `pls run --on <host> --cpus 24 --root <root> --detach --cmd -- python -m holosoma_retargeting ...`.
  The run sees no CUDA device (`CUDA_VISIBLE_DEVICES=""`), is pinned with `taskset` to the highest-numbered cores
  your other CPU runs there do not hold, runs at `nice -n 10` so GPU jobs' feeder threads come first, and gets
  `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=N` (an `--env` value wins). Its lease's card is `<host>:cpu` with
  `cpus = N`: it never blocks a card claim, never yields to a reservation and never lapses for idling; it ends when
  the run does (the run's own release, `pls run --stop`, or `pls res status` finding its process group gone on the
  box). Your CPU runs on one box hold at most its budget: `cpu_budget = <cores>` under the ssh pool's
  `[compute.<name>]`, else half the box's `nproc`; a run that would exceed it is refused. Staging, `--wt`, `--root`,
  `--venv`, `--detach`, `--timeout`, `--stall` and the W&B refs work as on a card; `{DEVICE}` is refused, and SLURM
  (its own CPU allocation) and Ray refuse `--cpus`.
- Pool settings: `workspace` (the manager checkout on the target, providing the venv and the asset repos; default
  `/var/local/<user>/hcrl_isaac_manager` on ssh pools, this machine's checkout locally), `scratch` (default
  `/var/local/<user>`, `~/tmp` locally) and `pin`. Keep `scratch` on local disk, never a quota'd NFS home: snapshots,
  checkpoints and Kit caches live there.
- Guardrails, for a box someone else also works at (e.g. a desktop with a small disk), all optional:
  `min_free_gb = N` refuses to stage (and releases the lease) when the filesystem holding `scratch` has less than N
  GB free; `max_hours = H` refuses a `--timeout` over H and gives a run without one `--timeout H h`, and the runner
  itself then stops the run's process group at the limit (TERM, KILL 60 s on), so a `--detach` run or one whose
  session died is bounded too; `desktop = true` lets the login user's own processes under `desktop_mib` (default
  2048) that are not pls runs (none with `scratch` or `res-eval` in their command line) leave the card free, listed
  in its note as `desktop: ...`, and drops the utilization test for "held" (compositing keeps it up); memory nobody
  lists still holds the card at `desktop_mib` or above. `max_hours` and `min_free_gb` apply to `local` pools too.

## Pools

Every compute pool has a `kind`, which picks the backend that probes it:

| kind | probes | settings |
|---|---|---|
| `local` | `nvidia-smi` on this machine | |
| `ssh` | `nvidia-smi` on each host over ssh (keys, BatchMode) | `hosts`, `user`, `domain`, `pin`, `render`, `workspace`, `scratch`, `cpu_budget`, `min_free_gb`, `max_hours`, `desktop`, `desktop_mib` |
| `slurm` | your jobs (`squeue`) and, per running job, `nvidia-smi` in an `srun --overlap` step over the cluster's SSH master | from the cluster profile |
| `ray` | the Ray dashboard API (nodes, GPUs, running jobs) | `address` |

The tracked `compute.toml` holds only `local` (the machine you run on) and commented examples. Every other pool is
yours, in `compute.local.toml` (gitignored, merged over it; the main checkout's serves every worktree):

```bash
pls setup --add lab-a40 --kind ssh --hosts box3,box4 --user <login> --domain lab.example.edu [--pin device] [--render]
pls setup --add desk --kind ssh --hosts <box> --user <login> --min-free-gb 25 --max-hours 2 --desktop  # a shared desktop
pls setup --add ray --kind ray --address http://<head>:8265      # the pool `pls run --on ray` submits to
pls setup --add lab-a40 ... --update                              # replace a pool
```

or by hand, e.g. `enabled = false` under `[compute.<name>]` to hide a pool. An ssh pool without a `user` (and no
`$LARG_USER`) reads UNKNOWN with a hint. SLURM pools come from your cluster profiles
(`scripts/cluster/config/<name>/`, `pls setup --add <name>`); profiles that share a login are probed once. A new kind
is a new probe function in `probe.py`. `python3 scripts/cluster/res/inventory.py` answers the shell helpers' queries
(an ssh host's target, the Ray address).
