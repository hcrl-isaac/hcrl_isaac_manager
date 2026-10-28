"""Same T1 model, same PD torque law: CPU MuJoCo vs MuJoCo-Warp trajectories, standing with small leg sinusoids."""

import numpy as np
import sys

import mujoco
import mujoco_warp as mjw
import warp as wp
import yaml

xml, plant_path, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
amp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
with open(plant_path) as f:
    cfg = yaml.safe_load(f)
kp = np.asarray(cfg["common"]["stiffness"], float)
kd = np.asarray(cfg["common"]["damping"], float) * np.asarray(cfg["mech"]["firmware_kd_scale"], float)
tau_max = np.asarray(cfg["common"]["torque_limit"], float)
q0 = np.asarray(cfg["common"]["default_qpos"], float)

m = mujoco.MjModel.from_xml_path(xml)
m.opt.timestep = float(cfg["common"]["dt"])
# actuator order = joint order in this model; map SDK-order arrays through actuator joint ids
act_qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
act_dadr = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
legs = np.array([
    "Hip" in m.joint(m.actuator_trnid[i, 0]).name
    or "Knee" in m.joint(m.actuator_trnid[i, 0]).name
    or "Ankle" in m.joint(m.actuator_trnid[i, 0]).name
    for i in range(m.nu)
])


def init(d: mujoco.MjData) -> None:
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.72
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[act_qadr] = q0
    mujoco.mj_forward(m, d)


def target(t: float) -> np.ndarray:
    return q0 + legs * amp * np.sin(2 * np.pi * 1.0 * t + np.arange(m.nu))


def torque(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
    return np.clip(kp * (target(t) - q) - kd * dq, -tau_max, tau_max)


steps = int(seconds / m.opt.timestep)
# CPU MuJoCo
d = mujoco.MjData(m)
init(d)
cpu = np.zeros((steps, m.nq))
for k in range(steps):
    d.ctrl[:] = torque(d.qpos[act_qadr], d.qvel[act_dadr], d.time)
    mujoco.mj_step(m, d)
    cpu[k] = d.qpos
# MuJoCo-Warp, one world, same law evaluated on the host each step
wp.init()
d2 = mujoco.MjData(m)
init(d2)
mw = mjw.put_model(m)
dw = mjw.put_data(m, d2, nconmax=256, njmax=512)
gpu = np.zeros((steps, m.nq))
t = 0.0
for k in range(steps):
    qpos = dw.qpos.numpy()[0]
    qvel = dw.qvel.numpy()[0]
    ctrl = torque(qpos[act_qadr], qvel[act_dadr], t).astype(np.float32)
    wp.copy(dw.ctrl, wp.array(ctrl[None, :], dtype=wp.float32))
    mjw.step(mw, dw)
    t += m.opt.timestep
    gpu[k] = dw.qpos.numpy()[0]
for sec in (0.25, 0.5, 1.0, 2.0, seconds):
    k = min(int(sec / m.opt.timestep) - 1, steps - 1)
    jerr = np.abs(cpu[k, act_qadr] - gpu[k, act_qadr])
    print(
        f"t={sec:4.2f}s base z cpu {cpu[k, 2]:.4f} warp {gpu[k, 2]:.4f} | base xy diff {np.linalg.norm(cpu[k, :2] - gpu[k, :2]) * 100:.2f} cm"
        f" | joint |err| mean {jerr.mean():.4f} max {jerr.max():.4f} rad (leg max {jerr[legs].max():.4f})"
    )


def fell(x: np.ndarray) -> bool:
    return bool((x[:, 2] < 0.45).any())


print("fell: cpu", fell(cpu), "warp", fell(gpu))
