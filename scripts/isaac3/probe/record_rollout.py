"""Roll a trained T1 velocity policy through a fixed command schedule and save the robot trajectory.

usage: record_rollout.py CHECKPOINT OUT.npz [physics=newton_mjwarp] [--num_envs 4]
Saves root pos/quat (MuJoCo wxyz), joint positions by name, commands and falls, one row per policy step.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("out")
parser.add_argument("--task", default="hcrl/T1-Velocity-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--rtx_video", default=None, help="also render with Kit RTX into this dir (needs the isaacsim extra)"
)
parser.add_argument(
    "--gl_video", default=None, help="also render the USD scene with the headless Newton GL visualizer into this dir"
)
from isaaclab.app import add_launcher_args, launch_simulation

add_launcher_args(parser)
args, overrides = parser.parse_known_args()

# Kit has to load its own USD build before our task packages pull in the kit-less pxr, so start it first
if args.rtx_video:
    from isaaclab.app import AppLauncher

    _app = AppLauncher(headless=True, enable_cameras=True).app

import contextlib
import numpy as np
import torch

import gymnasium as gym
import hcrl_isaaclab  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import register_task
from robot_rl.runners import OnPolicyRunner

# (seconds, vx, vy, wz): walk, faster, sideways, turn, back, stand
SCHEDULE = [
    (3, 0.5, 0.0, 0.0),
    (3, 1.0, 0.0, 0.0),
    (3, 0.0, 0.4, 0.0),
    (3, 0.5, 0.0, 0.5),
    (3, -0.5, 0.0, 0.0),
    (2, 0.0, 0.0, 0.0),
]

env_cfg, agent_cfg, rest = register_task(args.task, "rsl_rl_cfg_entry_point", overrides=overrides)
assert not rest, rest
env_cfg.scene.num_envs = args.num_envs
env_cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
env_cfg.episode_length_s = sum(s for s, *_ in SCHEDULE) + 5.0
env_cfg.seed = 0
physics = type(env_cfg.sim.physics).__name__
if args.rtx_video:
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    steps = sum(round(s / (env_cfg.sim.dt * env_cfg.decimation)) for s, *_ in SCHEDULE)
    # camera follows env 0's robot root
    env_cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(
            headless=True,
            window_width=1280,
            window_height=720,
            origin_type="asset",
            origin_track_path="robot",
            eye=(2.6, -2.6, 0.15),
            lookat=(0.0, 0.0, -0.1),
            enable_markers=False,
            background_color=None,
        )
    ]
    # Isaac Lab 2.x's grey grid floor rather than 3.0's checker ground
    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    @configclass
    class GridGroundPlaneCfg(sim_utils.GroundPlaneCfg):
        usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"

    sim_utils.GroundPlaneCfg = GridGroundPlaneCfg
    # the T1 URDF's visual colours carry alpha 0.2-0.3, which Isaac Lab 3.0's importer honours (2.x ignored it):
    # render from a copy with opaque colours, meshes still read from the original asset dir
    import os
    import re
    import tempfile

    spawn = env_cfg.scene.robot.spawn
    if str(spawn.asset_path).endswith(".urdf"):
        with open(spawn.asset_path) as f:
            text = f.read()
        text = re.sub(
            r'rgba="([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"',
            lambda m: f'rgba="{m[1]} {m[2]} {m[3]} {1 if float(m[4]) > 0 else 0}"',
            text,
        )
        text = text.replace('filename="', f'filename="{os.path.dirname(os.path.abspath(spawn.asset_path))}/')
        opaque = os.path.join(tempfile.mkdtemp(), os.path.basename(spawn.asset_path))
        with open(opaque, "w") as f:
            f.write(text)
        spawn.asset_path = opaque
    # a clean shot: no command arrows or gait markers drawn into the scene
    for term in vars(env_cfg.commands).values():
        if hasattr(term, "debug_vis"):
            term.debug_vis = False
    env_cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=args.rtx_video, video_length=steps)]
if args.rtx_video:
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg

    steps = sum(round(s / (env_cfg.sim.dt * env_cfg.decimation)) for s, *_ in SCHEDULE)
    # camera follows env 0's robot root
    env_cfg.sim.visualizer_cfgs = [
        KitVisualizerCfg(
            headless=True,
            window_width=1280,
            window_height=720,
            origin_type="asset",
            origin_track_path="robot",
            eye=(2.6, -2.6, 0.15),
            lookat=(0.0, 0.0, -0.1),
            enable_markers=False,
            background_color=None,
        )
    ]
    env_cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=args.rtx_video, video_length=steps)]
if args.gl_video:
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    steps = sum(round(s / (env_cfg.sim.dt * env_cfg.decimation)) for s, *_ in SCHEDULE)
    env_cfg.sim.visualizer_cfgs = [
        # a fixed wide shot of env 0 over the whole schedule (Newton GL has no follow camera); record with --num_envs 1
        NewtonGLVisualizerCfg(
            headless=True,
            window_width=1280,
            window_height=720,
            eye=(2.5, -5.0, 1.6),
            lookat=(2.5, 0.5, 0.5),
            focal_length=24.0,
            visible_env_indices=[0],
        )
    ]
    env_cfg.video_recorders = [
        VideoRecorderCfg(source="visualizer:newton_gl", output_dir=args.gl_video, video_length=steps)
    ]
with contextlib.nullcontext() if args.rtx_video else launch_simulation(env_cfg, args):
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint, load_cfg={"actor": True, "memory": True, "style": False})
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    robot = base.scene["robot"]
    cmd_term = base.command_manager.get_term("base_velocity")
    obs = env.get_observations()
    rows = {
        "root_pos": [],
        "root_quat_wxyz": [],
        "joint_pos": [],
        "cmd": [],
        "lin_vel_b": [],
        "ang_vel_b": [],
        "done": [],
    }
    for secs, vx, vy, wz in SCHEDULE:
        for _ in range(round(secs / base.step_dt)):
            cmd_term.vel_command_b[:] = torch.tensor([vx, vy, wz], device=base.device)
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            q = robot.data.root_quat_w.torch.cpu().numpy()  # Isaac Lab 3.0 is xyzw
            rows["root_pos"].append(robot.data.root_pos_w.torch.cpu().numpy() - base.scene.env_origins.cpu().numpy())
            rows["root_quat_wxyz"].append(q[:, [3, 0, 1, 2]])
            rows["joint_pos"].append(robot.data.joint_pos.torch.cpu().numpy())
            rows["cmd"].append(np.tile([vx, vy, wz], (base.num_envs, 1)))
            rows["lin_vel_b"].append(robot.data.root_lin_vel_b.torch.cpu().numpy())
            rows["ang_vel_b"].append(robot.data.root_ang_vel_b.torch.cpu().numpy())
            rows["done"].append(dones.cpu().numpy())
    out = {k: np.stack(v, axis=1) for k, v in rows.items()}  # (envs, steps, ...)
    np.savez(args.out, joint_names=np.array(robot.joint_names), dt=base.step_dt, physics=physics, **out)
    falls = out["done"].any(axis=1).sum()
    err = np.abs(out["lin_vel_b"][..., :2] - out["cmd"][..., :2]).mean()
    print(
        f"RECORDED {args.out}: {physics}, {out['root_pos'].shape[1]} steps x {base.num_envs} envs, "
        f"envs with a reset {falls}, mean |v_xy - cmd| {err:.3f} m/s",
        flush=True,
    )
    env.close()
