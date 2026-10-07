"""Per-command-segment tracking error and resets of recorded rollouts, as a markdown table.

usage: rollout_metrics.py LABEL=ROLLOUT.npz [LABEL=ROLLOUT.npz ...]
"""

import numpy as np
import sys

rows = []
for spec in sys.argv[1:]:
    label, path = spec.split("=", 1)
    r = np.load(path)
    cmd, v, w, done = r["cmd"], r["lin_vel_b"], r["ang_vel_b"], r["done"]
    # segment boundaries where the (shared) command changes
    change = np.flatnonzero(np.any(np.diff(cmd[0], axis=0) != 0, axis=1)) + 1
    for seg in np.split(np.arange(cmd.shape[1]), change):
        c = cmd[0, seg[0]]
        settled = seg[len(seg) // 4 :]  # skip the first quarter of each segment (transient)
        exy = np.abs(v[:, settled, :2] - cmd[:, settled, :2]).mean()
        ewz = np.abs(w[:, settled, 2] - cmd[:, settled, 2]).mean()
        rows.append((label, f"vx {c[0]:+.1f} vy {c[1]:+.1f} wz {c[2]:+.1f}", exy, ewz, int(done[:, seg].sum())))
print("| policy | command | mean abs v_xy err (m/s) | mean abs w_z err (rad/s) | resets |")
print("|---|---|---|---|---|")
for label, c, exy, ewz, n in rows:
    print(f"| {label} | {c} | {exy:.3f} | {ewz:.3f} | {n} |")
