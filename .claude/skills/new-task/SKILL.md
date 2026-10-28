---
name: new-task
description: Add a task to a workspace task repo the house way — where the code goes, the task id rules, registration, config-class structure, reward-design rules, and the smoke tests that prove it builds. Use when creating a new env/task or a new variant of one.
---

# New task

## 1. Where it goes

- A project's task lives in its `resources/<project>_tasks/` repo; shared MDP terms, assets and reference tasks go in
  `resources/hcrl_isaaclab/`. A new project repo: `pls new <name>` scaffolds `<name>_tasks`.
- Layout per task: `<project>_tasks/<task>/` holds the robot-agnostic env cfg (`<task>_env_cfg.py`) and an `mdp/`
  package for task-specific terms; each robot gets a subpackage (`<task>/<robot>/`) with its env cfg subclass,
  runner cfgs (`alg_cfg.py`) and the registrations (`__init__.py`). Import the robot subpackages from the task's
  `__init__.py`. Keep task cfgs robot-agnostic; robot specifics belong in the robot subpackage.

## 2. The id

- `<Robot>-<Task>-v0` is the canonical teacher, without curriculum; `-Depth-v0` the distilled student; `-PPO-v0` the
  PPO variant. CamelCase inside a component (`T1-CubePush-v0`, not `T1-Cube-Push-v0`).
- An ablation axis is a cfg value, overridden at launch, not a new id. No alias ids: every id resolves to its own env
  or runner cfg. Check the id is unused across every task repo before registering it.

## 3. Register it

```python
from hcrl_isaaclab.tasks_registry import register_task

register_task(
    id="<Robot>-<Task>-v0",
    entry_point="hcrl_isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.<task>_env_cfg:<Robot><Task>EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.alg_cfg:<Robot><Task><Algo>RunnerCfg",
    },
)
```

## 4. Config classes

- Isaac Lab patterns: `@configclass`, `MISSING` for required fields, a `<X>Term` class paired with its `<X>TermCfg`.
- A generic name carries only generic content: shared knobs in a base cfg, feature-specific fields in a subclass.
- One reward table per task: a finalized table replaces the base, rather than a subclass that nulls terms.
- Isaac Lab 3: quaternions are xyzw, asset data reads go through `.torch`, and backend-specific values (PhysX vs
  Newton) are physics presets, never `if` branches on the backend.

## 5. Rewards and terminations

- No terminal charges (e.g. -100 on a fall) and no `is_alive` terms: remove them, don't zero them.
- Keep the per-step reward sum positive, with penalties as bounded kernels (e.g. `exp(-err²/σ²)`), not unbounded
  linear costs. No approach or progress shaping (closing speed): use a bounded distance kernel, and only if shown
  necessary.
- No explicit gait or phase rewards where a style or traverse expert exists.
- A high-level policy observes no latent, including its own previous latent action.
- Before choosing a threshold (clip, clamp, gate), print the quantity's percentiles and put the bound in a gap.

## 6. Prove it builds and runs

```bash
pls test <project>_tasks       # CPU smoke: every registered id registers and its cfgs build
pls run train --task <id> --num_envs 64 --max_iterations 2 --logger tensorboard --video off
```

The smoke tests pick up every registered task automatically; add a test only for logic they don't exercise. Then
compare the trained policy against a do-nothing policy on the task's own success metric before calling it working.
