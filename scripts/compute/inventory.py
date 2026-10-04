"""Compute inventory: the pools `just res` knows about, from compute.toml plus per-user overrides."""

from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

COMPUTE_DIR = Path(__file__).resolve().parent
CLUSTER_CONFIG_DIR = COMPUTE_DIR.parent / "cluster" / "config"


@dataclass
class Pool:
    """One compute pool and the settings its backend needs.

    Args:
        name: Pool name shown by `just res`.
        kind: Backend name (local, ssh, slurm, ray).
        settings: Backend-specific settings from the inventory.
    """

    name: str
    kind: str
    settings: dict = field(default_factory=dict)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, val in over.items():
        out[key] = _merge(out[key], val) if isinstance(val, dict) and isinstance(out.get(key), dict) else val
    return out


def _read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text()) if path.is_file() else {}


def _cluster_login(env_file: Path) -> str:
    """CLUSTER_LOGIN as the shell sees it after sourcing the profile (values may reference variables)."""
    script = 'source "$1" >/dev/null 2>&1; printf %s "${CLUSTER_LOGIN:-}"'
    cmd = ["bash", "-c", script, "_", str(env_file)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout


def slurm_pools() -> list[Pool]:
    """One SLURM pool per per-user cluster profile under scripts/cluster/config/."""
    pools = []
    for env_file in sorted(CLUSTER_CONFIG_DIR.glob("*/.env.cluster")):
        login = _cluster_login(env_file)
        if login:
            pools.append(Pool(env_file.parent.name, "slurm", {"login": login}))
    return pools


def load_pools() -> list[Pool]:
    """All enabled pools: compute.toml merged with compute.local.toml, plus the cluster profiles."""
    cfg = _merge(_read_toml(COMPUTE_DIR / "compute.toml"), _read_toml(COMPUTE_DIR / "compute.local.toml"))
    entries = cfg.get("compute", {})
    pools = [Pool(name, s["kind"], s) for name, s in entries.items() if "kind" in s]
    known = {p.name for p in pools}
    for pool in slurm_pools():
        pool.settings = _merge(pool.settings, entries.get(pool.name, {}))
        if pool.name not in known:
            pools.append(pool)
    for pool in pools:
        if pool.kind == "ssh" and os.environ.get("LARG_USER"):
            pool.settings.setdefault("user", os.environ["LARG_USER"])
    return [p for p in pools if p.settings.get("enabled", True)]
