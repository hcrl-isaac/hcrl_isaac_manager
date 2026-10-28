"""Build one of our tasks on Isaac Lab 3.0, step zero actions and report env steps/s.

usage: smoke_env.py TASK [hydra overrides, e.g. physics=newton_mjwarp env.scene.num_envs=64] [--steps N]
"""

import argparse
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument("--steps", type=int, default=50)
from isaaclab.app import add_launcher_args, launch_simulation

add_launcher_args(parser)
args, overrides = parser.parse_known_args()

import torch

import gymnasium as gym
import hcrl_isaaclab  # noqa: F401
import hhlm_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import register_task

env_cfg, _, rest = register_task(args.task, None, overrides=overrides)
assert not rest, f"unhandled overrides: {rest}"
print("physics:", type(env_cfg.sim.physics).__name__, "num_envs:", env_cfg.scene.num_envs, flush=True)
with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()
    print("obs groups:", {k: tuple(v.shape) for k, v in obs.items()}, flush=True)
    act = torch.zeros(
        env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device
    )
    for _ in range(5):
        env.step(act)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(args.steps):
        obs, rew, term, trunc, _ = env.step(act)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    n = env.unwrapped.num_envs
    print(
        f"SMOKE OK {args.task} {type(env_cfg.sim.physics).__name__} envs {n}: {n * args.steps / dt:,.0f} env-steps/s, "
        f"reward mean {rew.mean().item():.3f}, resets {int((term | trunc).sum())}",
        flush=True,
    )
    env.close()
sys.exit(0)
