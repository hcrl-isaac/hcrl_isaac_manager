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


def _sourced_env(env_file: Path, keys: list[str]) -> dict:
    """Return `keys` as the shell sees them after sourcing `env_file` (values may reference variables)."""
    script = 'source "$1" >/dev/null 2>&1; shift; for k in "$@"; do printf "%s=%s\\n" "$k" "${!k}"; done'
    res = subprocess.run(["bash", "-c", script, "_", str(env_file), *keys], capture_output=True, text=True)
    return dict(line.split("=", 1) for line in res.stdout.splitlines() if "=" in line)


def _sbatch_values(job_file: Path) -> dict:
    vals = {}
    if job_file.is_file():
        for line in job_file.read_text().splitlines():
            parts = line.split(None, 2)
            if len(parts) >= 2 and parts[0] == "#SBATCH":
                flag, _, val = parts[1].partition("=")
                vals[flag] = val or (parts[2] if len(parts) > 2 else "")
    return vals


def slurm_pools() -> list[Pool]:
    """One SLURM pool per per-user cluster profile under scripts/cluster/config/."""
    pools = []
    for env_file in sorted(CLUSTER_CONFIG_DIR.glob("*/.env.cluster")):
        env = _sourced_env(env_file, ["CLUSTER_LOGIN", "CLUSTER_ISAACLAB_DIR"])
        if not env.get("CLUSTER_LOGIN"):
            continue
        sbatch = _sbatch_values(env_file.parent / "submit_job_slurm.sh")
        settings = {"login": env["CLUSTER_LOGIN"], "partition": sbatch.get("-p", ""), "account": sbatch.get("-A", "")}
        pools.append(Pool(env_file.parent.name, "slurm", settings))
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
    default_user = os.environ.get("LARG_USER") or os.environ.get("USER", "")
    for pool in pools:
        if pool.kind == "ssh":
            pool.settings.setdefault("user", default_user)
    return [p for p in pools if p.settings.get("enabled", True)]
