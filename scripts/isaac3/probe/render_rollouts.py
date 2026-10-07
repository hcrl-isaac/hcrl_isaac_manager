"""Replay recorded Isaac rollouts kinematically in the T1 MuJoCo model, side by side, into one mp4.

usage: render_rollouts.py OUT.mp4 LABEL=ROLLOUT.npz [LABEL=ROLLOUT.npz ...] [--env 0] [--width 640]
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import imageio.v2 as imageio
import mujoco
from hcrl_sim2real.robots import get_robot
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("rollouts", nargs="+")
ap.add_argument("--env", type=int, default=0)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
args = ap.parse_args()

m = mujoco.MjModel.from_xml_path(get_robot("t1").default_model())
m.vis.global_.offwidth, m.vis.global_.offheight = (
    max(m.vis.global_.offwidth, args.width),
    max(m.vis.global_.offheight, args.height),
)
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, args.height, args.width)
free = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 135.0

panels = []
for spec in args.rollouts:
    label, path = spec.split("=", 1)
    r = np.load(path)
    adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in r["joint_names"]])
    panels.append((label, r, adr))
steps = min(p[1]["root_pos"].shape[1] for p in panels)
fps = round(1.0 / float(panels[0][1]["dt"]))


def frame(label: str, r: np.lib.npyio.NpzFile, adr: np.ndarray, k: int) -> np.ndarray:
    e = args.env
    qa = m.jnt_qposadr[free]
    d.qpos[qa : qa + 3] = r["root_pos"][e, k]
    d.qpos[qa + 3 : qa + 7] = r["root_quat_wxyz"][e, k]
    d.qpos[adr] = r["joint_pos"][e, k]
    mujoco.mj_forward(m, d)
    cam.lookat[:] = [d.qpos[qa], d.qpos[qa + 1], 0.6]
    renderer.update_scene(d, camera=cam)
    img = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(img)
    vx, vy, wz = r["cmd"][e, k]
    v = r["lin_vel_b"][e, k]
    draw.text((10, 8), label, fill=(255, 255, 255))
    draw.text((10, 24), f"cmd vx {vx:+.2f} vy {vy:+.2f} wz {wz:+.2f}", fill=(255, 255, 0))
    draw.text((10, 40), f"vel vx {v[0]:+.2f} vy {v[1]:+.2f}   t {k * float(r['dt']):5.1f}s", fill=(200, 255, 200))
    if r["done"][e, : k + 1].any():
        draw.text((10, 56), f"resets so far: {int(r['done'][e, : k + 1].sum())}", fill=(255, 120, 120))
    return np.asarray(img)


with imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8, macro_block_size=8) as w:
    for k in range(steps):
        w.append_data(np.concatenate([frame(lbl, r, adr, k) for lbl, r, adr in panels], axis=1))
print(f"wrote {args.out}: {steps} frames at {fps} fps, {len(panels)} panels")
