"""Split one env step's wall time into physics vs MDP managers (CUDA-synchronized timers around each phase).

usage: profile_step.py TASK [physics=newton_mjwarp] [env.scene.num_envs=4096] [--steps 50]
"""

import argparse
import collections
import time

parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--fuse_actuators", action="store_true", help="drive the robot with one fused delayed-PD group")
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

add_launcher_args(parser)
args, overrides = parser.parse_known_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import hcrl_isaaclab  # noqa: E402,F401
from isaaclab_tasks.utils.hydra import register_task  # noqa: E402

TIMES: dict[str, float] = collections.defaultdict(float)


def timed(obj: object, name: str, label: str) -> None:
    """Replace ``obj.name`` with a wrapper that adds its synchronized wall time to ``TIMES[label]``."""
    fn = getattr(obj, name)

    def wrapper(*a: object, **k: object) -> object:
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        TIMES[label] += time.perf_counter() - t
        return out

    setattr(obj, name, wrapper)


env_cfg, _, rest = register_task(args.task, None, overrides=overrides)
assert not rest, rest
if args.fuse_actuators:
    from hcrl_isaaclab.mdp.actuators import fuse_actuators

    env_cfg.scene.robot.actuators = {"all": fuse_actuators(env_cfg.scene.robot.actuators)}
with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    for _ in range(10):
        env.step(act)
    timed(u.sim, "step", "physics step")
    timed(u.scene, "write_data_to_sim", "scene write (actuators)")
    timed(u.scene, "update", "scene update (data/sensors)")
    timed(u.action_manager, "apply_action", "action apply")
    timed(u.action_manager, "process_action", "action process")
    timed(u.observation_manager, "compute", "observations")
    timed(u.reward_manager, "compute", "rewards")
    timed(u.termination_manager, "compute", "terminations")
    timed(u.command_manager, "compute", "commands")
    timed(u.event_manager, "apply", "events")
    timed(u, "_reset_idx", "resets")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        env.step(act)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    print(
        f"PROFILE {args.task} {type(env_cfg.sim.physics).__name__}{' fused' if args.fuse_actuators else ''} envs {u.num_envs}: "
        f"{total / args.steps * 1e3:.1f} ms/env-step, {u.num_envs * args.steps / total:,.0f} env-steps/s"
    )
    for k, v in sorted(TIMES.items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s} {v / args.steps * 1e3:7.2f} ms  {100 * v / total:5.1f}%")
    print(f"  {'(unattributed)':30s} {(total - sum(TIMES.values())) / args.steps * 1e3:7.2f} ms")
    env.close()
