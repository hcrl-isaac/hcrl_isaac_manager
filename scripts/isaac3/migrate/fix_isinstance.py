"""isinstance against Isaac Lab 3.0 factory classes is always False: check the Base* class instead."""

import pathlib
import re

R = pathlib.Path(__file__).resolve().parents[3] / "resources"  # manager resources/
MAP = {
    "ContactSensor": ("isaaclab.sensors", "BaseContactSensor"),
    "FrameTransformer": ("isaaclab.sensors", "BaseFrameTransformer"),
    "Articulation": ("isaaclab.assets", "BaseArticulation"),
    "RigidObject": ("isaaclab.assets", "BaseRigidObject"),
}
for repo in ("hcrl_isaaclab", "hhlm_tasks", "ssti_tasks"):
    for f in (R / repo).rglob("*.py"):
        s = f.read_text()
        need = set()
        for cls, (mod, base) in MAP.items():
            rx = re.compile(rf"(isinstance\([^,()]+,\s*){cls}\)")
            if rx.search(s):
                s = rx.sub(rf"\g<1>{base})", s)
                need.add((mod, base))
        if not need:
            continue
        for mod, base in sorted(need):
            imp = re.search(rf"^from {re.escape(mod)} import ([^\n(]+)$", s, re.M)
            if imp and base not in imp.group(1):
                s = s[: imp.end(1)] + f", {base}" + s[imp.end(1) :]
            elif not imp:
                first = re.search(r"^(from|import) isaaclab", s, re.M)
                s = s[: first.start()] + f"from {mod} import {base}\n" + s[first.start() :]
        f.write_text(s)
        print(f.relative_to(R), sorted(b for _, b in need))
