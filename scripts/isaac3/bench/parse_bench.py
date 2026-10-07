"""Summarize a train.py log: median collection/learning/iteration time after warm-up, and env-steps/s."""

import re
import statistics
import sys

with open(sys.argv[1]) as f:
    text = f.read()
envs = int(sys.argv[2])
grab = lambda key: [float(x) for x in re.findall(rf"{key} time: ([0-9.]+)s", text)]  # noqa: E731
col, learn, it = grab("Collection"), grab("Learning"), grab("Iteration")
steps = re.search(r"num_steps_per_env['\"]?\s*[:=]\s*(\d+)", text)
spe = int(steps.group(1)) if steps else 24
gpu = re.search(
    r"Device name\s*:\s*(.+)|NVIDIA [A-Za-z0-9 ]+(?:GPU|PRO|A100|A40|H100|H200|L40S?|GB200)[A-Za-z0-9 ]*", text
)
if len(col) < 8:
    print(f"FAILED ({len(col)} iterations logged)")
    sys.exit(0)
c, l, i = (statistics.median(x[5:]) for x in (col, learn, it))
print(
    f"iters {len(col)} collect {c:.3f}s learn {l:.3f}s iter {i:.3f}s env-steps/s {envs * spe / c:,.0f} (collection) {envs * spe / i:,.0f} (end-to-end)"
)
