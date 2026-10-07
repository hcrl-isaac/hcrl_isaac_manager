"""Compute inventory: the pools `just res` knows about, from compute.toml plus per-user overrides."""

from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

COMPUTE_DIR = Path(__file__).resolve().parent


def _cluster_config_dir() -> Path:
    """This checkout's cluster profiles, or the main checkout's when this worktree has none (they are gitignored)."""
    own = COMPUTE_DIR.parent / "cluster" / "config"
    if any(own.glob("*/.env.cluster")):
        return own
    common = subprocess.run(
        ["git", "-C", str(COMPUTE_DIR), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    main = Path(common).parent / "scripts" / "cluster" / "config" if common else own
    return main if any(main.glob("*/.env.cluster")) else own


CLUSTER_CONFIG_DIR = _cluster_config_dir()


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


def profile_value(profile: str, key: str) -> str:
    """A cluster profile's variable as the shell sees it after sourcing the profile (values may reference others).

    Args:
        profile: Profile name (a directory under scripts/cluster/config/).
        key: Variable name, e.g. ``CLUSTER_ISAACLAB_DIR``.

    Returns:
        Its value, or "" when the profile or the variable is missing.
    """
    env_file = CLUSTER_CONFIG_DIR / profile / ".env.cluster"
    if not env_file.is_file():
        return ""
    script = f'source "$1" >/dev/null 2>&1; printf %s "${{{key}:-}}"'
    cmd = ["bash", "-c", script, "_", str(env_file)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout


def _cluster_login(env_file: Path) -> str:
    """CLUSTER_LOGIN as the shell sees it after sourcing the profile (values may reference variables)."""
    return profile_value(env_file.parent.name, "CLUSTER_LOGIN")


def slurm_pools() -> list[Pool]:
    """One SLURM pool per per-user cluster profile under scripts/cluster/config/."""
    pools = []
    for env_file in sorted(CLUSTER_CONFIG_DIR.glob("*/.env.cluster")):
        login = _cluster_login(env_file)
        if login:
            pools.append(Pool(env_file.parent.name, "slurm", {"login": login}))
    return pools


def load_config() -> dict:
    """compute.toml merged with compute.local.toml."""
    return _merge(_read_toml(COMPUTE_DIR / "compute.toml"), _read_toml(COMPUTE_DIR / "compute.local.toml"))


def load_pools() -> list[Pool]:
    """All enabled pools: compute.toml merged with compute.local.toml, plus the cluster profiles."""
    cfg = load_config()
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
