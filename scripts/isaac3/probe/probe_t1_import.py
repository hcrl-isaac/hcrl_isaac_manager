"""Spawn the identified T1 MJCF through Isaac Lab's Newton backend and compare the solver's MuJoCo model with the MJCF."""

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("xml")
args = parser.parse_args()

# kit-less: register the importer's PhysX schema fallback before any USD stage exists
import numpy as np

import isaaclab.sim as sim_utils
import isaacsim.asset.importer.utils.impl.physx_types  # noqa: F401
import mujoco
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

sim = SimulationContext(
    SimulationCfg(dt=0.002, device="cuda:0", physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=512, nconmax=256)))
)
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
robot = Articulation(
    ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.MjcfFileCfg(asset_path=args.xml, self_collision=True, fix_base=False),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.72)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )
)
sim.reset()
mjm = NewtonManager._solver.mj_model
ref = mujoco.MjModel.from_xml_path(args.xml)
print(
    f"solver model: nq {mjm.nq} nv {mjm.nv} nbody {mjm.nbody} ngeom {mjm.ngeom} nu {mjm.nu} | MJCF: nq {ref.nq} nv {ref.nv} nbody {ref.nbody} ngeom {ref.ngeom}"
)
print(
    f"total mass solver {mjm.body_mass.sum():.3f} kg vs MJCF {ref.body_mass.sum():.3f} kg | timestep {mjm.opt.timestep} cone {mjm.opt.cone} integrator {mjm.opt.integrator}"
)


def jname(m: mujoco.MjModel, j: int) -> str:
    n = m.joint(j).name
    return n.split("/")[-1]


ref_j = {jname(ref, j): j for j in range(ref.njnt) if ref.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
rows = []
for j in range(mjm.njnt):
    if mjm.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
        continue
    n = jname(mjm, j)
    k = next((ref_j[r] for r in ref_j if n.endswith(r) or r.endswith(n)), None)
    if k is None:
        rows.append((n, "NO MATCH"))
        continue
    a, b = mjm.jnt_dofadr[j], ref.jnt_dofadr[k]
    rows.append((
        n,
        mjm.dof_armature[a],
        ref.dof_armature[b],
        mjm.dof_damping[a],
        ref.dof_damping[b],
        mjm.dof_frictionloss[a],
        ref.dof_frictionloss[b],
    ))
bad = 0
for r in rows:
    if r[1] == "NO MATCH":
        print("  ", r[0], "NO MATCH")
        bad += 1
        continue
    n, a1, a2, d1, d2, f1, f2 = r
    ok = np.allclose([a1, d1, f1], [a2, d2, f2], atol=1e-4)
    bad += not ok
    if not ok or "Hip_Roll" in n or "Ankle_Pitch" in n:
        print(
            f"   {n:22s} armature {a1:.4f}/{a2:.4f} damping {d1:.4f}/{d2:.4f} frictionloss {f1:.4f}/{f2:.4f} {'OK' if ok else 'MISMATCH'}"
        )
print(f"joints compared {len(rows)}, mismatched {bad}")
fl = [mjm.geom_friction[g] for g in range(mjm.ngeom) if mjm.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
print(
    "solver floor friction:",
    fl[:1],
    "| MJCF floor friction:",
    [ref.geom_friction[g] for g in range(ref.ngeom) if ref.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE][:1],
)
print("solver geom types:", np.bincount(mjm.geom_type), "| MJCF geom types:", np.bincount(ref.geom_type))
sys.exit(0)
