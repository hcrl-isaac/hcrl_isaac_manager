"""Apply mechanical Isaac Lab 2.x -> 3.0 renames across our repos (prints files changed per rule)."""

import pathlib
import re
import sys

R = pathlib.Path(__file__).resolve().parents[3] / "resources"  # manager resources/
REPOS = ["hcrl_isaaclab", "robot_rl", "hhlm_tasks", "ssti_tasks"]
RULES = [
    (r"\bAdditiveUniformNoiseCfg\b", "UniformNoiseCfg"),
    (r"\bAdditiveGaussianNoiseCfg\b", "GaussianNoiseCfg"),
    # SimulationCfg.render (RTX tuning) is gone in 3.0
    (r"^[ \t]*#?[ \t]*self\.sim\.render\.\w+ *=.*\n", ""),
    # 2.x PhysX contact forces were normal-only; 3.0 net_forces_w adds friction on Newton
    (r"\bnet_forces_w\b", "net_normal_forces_w"),
    (r"\bnet_forces_w_history\b", "net_normal_forces_w_history"),
    (r"\bforce_matrix_w\b", "normal_force_matrix_w"),
    (r"\bforce_matrix_w_history\b", "normal_force_matrix_w_history"),
]
only = set(sys.argv[1:])
for pat, rep in RULES:
    if only and pat not in only:
        continue
    rx = re.compile(pat, re.M)
    hits = []
    for repo in REPOS:
        for f in (R / repo).rglob("*.py"):
            if ".venv" in f.parts or "__pycache__" in f.parts:
                continue
            s = f.read_text()
            t = rx.sub(rep, s)
            if t != s:
                f.write_text(t)
                hits.append(str(f.relative_to(R)))
    print(f"{pat} -> {rep}: {len(hits)} files")
    for h in hits:
        print("   ", h)
