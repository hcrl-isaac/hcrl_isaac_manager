"""Write a T1 MJCF with the identified plant (armature, damping, frictionloss, floor friction) baked into the joints."""

import numpy as np
import os
import sys

import mujoco
import yaml
from hcrl_sim2real.robots import get_robot

plant_path, out_path = sys.argv[1], sys.argv[2]
robot = get_robot("t1")
src = robot.default_model()
with open(plant_path) as f:
    plant = yaml.safe_load(f)["sim"]
spec = mujoco.MjSpec.from_file(src)
# absolute mesh dir, so the baked file can live anywhere
spec.meshdir = os.path.join(os.path.dirname(src), spec.meshdir or "")
names = list(robot.joint_names)  # SDK order, the plant config's order
for key, attr in (("joint_armature", "armature"), ("joint_damping", "damping"), ("joint_frictionloss", "frictionloss")):
    vals = np.asarray(plant[key], float)
    assert vals.size == len(names), (key, vals.size)
    for n, v in zip(names, vals, strict=False):
        cur = getattr(spec.joint(n), attr)
        if np.ndim(cur):  # newer MuJoCo stores some joint fields as small vectors; the first entry is the classic value
            cur = np.array(cur, float).ravel()
            cur[0] = v
            setattr(spec.joint(n), attr, cur.reshape(-1, 1) if np.ndim(getattr(spec.joint(n), attr)) == 2 else cur)
        else:
            setattr(spec.joint(n), attr, float(v))
floor = float(np.atleast_1d(plant["floor_friction"])[0])
for g in spec.geoms:
    if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
        g.friction[0] = floor
m = spec.compile()
with open(out_path, "w") as f:
    f.write(spec.to_xml())
print(
    "wrote", out_path, "nq", m.nq, "njnt", m.njnt, "nu", m.nu, "ngeom", m.ngeom, f"mass {m.body_subtreemass[1]:.2f} kg"
)
for n in ("Left_Hip_Roll", "Left_Ankle_Pitch", "Left_Knee_Pitch"):
    d = m.joint(n).dofadr[0]
    print(
        f"  {n}: armature {m.dof_armature[d]:.4f} damping {m.dof_damping[d]:.4f} frictionloss {m.dof_frictionloss[d]:.4f}"
    )
