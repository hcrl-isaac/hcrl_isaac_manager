"""Isaac Lab 3.0 data properties return ProxyArray: append ``.torch`` where our code reads them as tensors."""

import pathlib
import re
import sys

R = pathlib.Path(__file__).resolve().parents[3] / "resources"  # manager resources/
props = pathlib.Path(sys.argv[1]).read_text().split()
alt = "|".join(sorted(props, key=len, reverse=True))
# `<x>.data.<prop>` plus the usual aliases of an asset's data object
rx = re.compile(
    rf"((?:\.data|\b(?:d|data|robot_data|asset_data|obj_data|sensor_data))\.(?:{alt}))\b(?!\.torch|\.warp|\s*=[^=]|\()"
)
total = 0
for repo in sys.argv[2:]:
    for f in (R / repo).rglob("*.py"):
        if "__pycache__" in f.parts:
            continue
        s = f.read_text()
        t, n = rx.subn(r"\1.torch", s)
        if n:
            f.write_text(t)
            total += n
            print(f"{n:4d} {f.relative_to(R)}")
print("total", total)
