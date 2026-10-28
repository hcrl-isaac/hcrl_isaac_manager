---
name: launch-run
description: Launch a real training run the safe way — config check, smoke, launch with W&B and video, confirm it started, and arm a watchdog that catches crashes and stalls. Use for any run past a quick local smoke, on any compute (`pls run` here or `--on` a pool, cluster or Ray).
---

# Launch a run

A run that dies in its first minute looks exactly like a healthy one unless something watches it. This procedure
ends with a watchdog armed; after that, leave the run alone.

## 1. Check the config

- The overrides must exist on **this** task. Read a recent run's dumped `params/env.yaml` and `params/agent.yaml`
  for the task (or build the cfg once) and check every key you pass; a key that exists on a sibling task aborts at
  boot with `Could not override`.
- Know the budget: `--max_iterations` is literal. On multi-GPU (N ranks) the budget is divided by N; on a resume,
  `agent.max_iterations` is the remainder, not the total. Know the checkpoint save interval: a run that hits a wall
  between saves loses that work.
- On a cluster, know the wall: `scontrol show job <id>` prints start and end in the cluster's local time. If it won't
  fit, plan the resume from the last checkpoint.

## 2. Smoke it where it will run

```bash
pls run --on <target> -- train --task <id> --num_envs 64 --max_iterations 2 --logger tensorboard --video off <overrides>
```

Passing means the log shows `Learning iteration 1/2` (for distillation, also `Loading model checkpoint`). Exit code 0
proves nothing.

## 3. Launch

```bash
pls run --on <target> [--detach] -- train --task <id> --run_name <delta> --run_group <dimension> <overrides>
```

- W&B logging and video are on by default; use `--video async` on a card without RT cores (A100, H100, GB200), so
  a recorder on an RT card renders the clips.
- `--run_name` is the change from the previous run (e.g. `resample=1-5s`); `--run_group` is the dimension being
  explored (e.g. `teacher`). Never a repeat, a status or a one-run group.
- A job that needs no GPU runs on an ssh box's cores with no card leased: `pls run --on <host> --cpus N [--root DIR]
  --detach --cmd -- <command>` (pinned, nice 10, within your per-box CPU budget).
- On a SLURM cluster, `--batch` queues it as a job; on a held node it runs on its cards. Launch one run per command.
  When several Kit processes start at once, give each its own `TMPDIR`, `XDG_CACHE_HOME` and `OMNI_CACHE_DIR`, and
  stagger launches about 20 s apart (concurrent URDF imports race).

## 4. Confirm it started

A launcher returning a log path is not a start. Wait for the first `Learning iteration` line in that log (or the W&B
run's first logged iteration), then read the first few iterations for NaNs or a pinned metric.

## 5. Arm the watchdog

```bash
scripts/tools/watch_run.sh [--wandb <entity/project/run_id>] <label> <log> <target-iters> [stall_min] [report_every]
```

`<log>` is a local path, `user@host:/path` when the log is written on another host, or `ray:<job_id>`. It exits
FINISHED, STALLED, DEAD, FAILED or BLIND. Run it in the background so it wakes you when it exits, and treat every
exit except FINISHED as needing a look: check the log and the card (`pls res status`) before deciding it was noise.

## 6. When it ends

Confirm from the log that the final checkpoint was saved, not just that the process exited. Stop a run you own with
`pls run --stop <lease>` or `pls jobs --on <cluster> stop <job>`, then check the card drained (Isaac keeps its VRAM
after exit).
