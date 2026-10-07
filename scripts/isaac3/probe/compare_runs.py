"""Compare W&B training curves of several runs at matched learning iterations.

usage: compare_runs.py ENTITY/PROJECT LABEL=RUN_ID [LABEL=RUN_ID ...] [--at 50,100,150]
"""

import argparse

import wandb

KEYS = {
    "reward": "Train/mean_reward",
    "ep_len": "Train/mean_episode_length",
    "err_vxy": "Metrics/base_velocity/error_vel_xy",
    "fall": "Episode_Termination/body_height",
}
ap = argparse.ArgumentParser()
ap.add_argument("project")
ap.add_argument("runs", nargs="+")
ap.add_argument("--at", default="50,100,150,200,300,500,1000")
args = ap.parse_args()
its = [int(x) for x in args.at.split(",")]
api = wandb.Api(timeout=120)
print("| run | it | " + " | ".join(KEYS) + " |")
print("|---|---|" + "---|" * len(KEYS))
for spec in args.runs:
    label, rid = spec.split("=", 1)
    run = api.run(f"{args.project}/{rid}")
    # one row per logged step; the learning iteration is the row index of rows that carry the reward
    rows = [r for r in run.scan_history(keys=list(KEYS.values())) if r.get(KEYS["reward"]) is not None]
    for it in its:
        if it < len(rows):
            r = rows[it]
            print(f"| {label} | {it} | " + " | ".join(f"{r.get(k, float('nan')):.3f}" for k in KEYS.values()) + " |")
