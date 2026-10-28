"""Compute inventory: the pools `pls res` knows about, from compute.toml plus per-user overrides.

Run as a script, it answers the shell helpers (scripts/larg/, scripts/ray/) and writes pools into the user's
compute.local.toml:
    inventory.py ssh-target <host>          the ssh target (user@host.domain) of a host
    inventory.py ssh-user                   the one login the ssh pools use
    inventory.py ssh-hosts                  every ssh pool's hosts, as "<pool> <host> <target>" lines
    inventory.py host-setting <host> <key>  a setting of the ssh pool that lists the host ("" when unset)
    inventory.py ray-address                the dashboard address of the `ray` pool
    inventory.py add <name> --kind ssh --hosts a,b --user u [--domain d] [--pin device] [--render] [--update]
    inventory.py add <name> --kind ray --address http://<head>:8265 [--update]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

RES_DIR = Path(__file__).resolve().parent
MANAGER_DIR = RES_DIR.parents[2]


def _main_checkout() -> Path:
    """The main checkout of this repo (this one, unless this is a worktree)."""
    common = subprocess.run(
        ["git", "-C", str(RES_DIR), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(common).parent if common else MANAGER_DIR


def _cluster_config_dir() -> Path:
    """This checkout's cluster profiles, or the main checkout's when this worktree has none (they are gitignored)."""
    own = MANAGER_DIR / "scripts" / "cluster" / "config"
    if any(own.glob("*/.env.cluster")):
        return own
    main = _main_checkout() / "scripts" / "cluster" / "config"
    return main if any(main.glob("*/.env.cluster")) else own


CLUSTER_CONFIG_DIR = _cluster_config_dir()


@dataclass
class Pool:
    """One compute pool and the settings its backend needs.

    Args:
        name: Pool name shown by `pls res`.
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


def local_config_path() -> Path:
    """The user's compute.local.toml: this checkout's, or the main checkout's in a worktree that has none (it is
    gitignored, so a worktree never has one and would lose the user's pools). A new file goes in the main checkout,
    where every worktree finds it."""
    local = RES_DIR / "compute.local.toml"
    return local if local.exists() else _main_checkout() / "scripts" / "cluster" / "res" / "compute.local.toml"


def load_config() -> dict:
    """compute.toml merged with the user's compute.local.toml (see local_config_path)."""
    return _merge(_read_toml(RES_DIR / "compute.toml"), _read_toml(local_config_path()))


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


def ssh_target(pool: Pool, host: str) -> str:
    """``user@fqdn`` for a host of an ssh pool: a host given as ``user@...`` as-is, the pool's domain appended to a
    short name."""
    if "@" in host:
        return host
    fqdn = host if "." in host or not pool.settings.get("domain") else f"{host}.{pool.settings['domain']}"
    return f"{pool.settings['user']}@{fqdn}"


def ssh_pool_of(host: str, pools: list[Pool]) -> Pool | None:
    """The ssh pool that lists ``host`` (a short name, or one with the pool's domain or a user@ prefix)."""
    bare = host.split("@")[-1]
    for pool in pools:
        if pool.kind != "ssh":
            continue
        domain = pool.settings.get("domain")
        if any(bare in (h, f"{h}.{domain}") for h in pool.settings.get("hosts", [])):
            return pool
    return None


def _no_user(where: str) -> str:
    return (
        f"no ssh user for {where}: set user under its [compute.<pool>] in compute.local.toml"
        " (`pls setup --add <pool> --kind ssh --hosts ... --user <login>`), or export LARG_USER"
    )


def resolve_ssh_target(host: str, pools: list[Pool]) -> str:
    """The ssh target of ``host``: its pool's user and domain, or $LARG_USER for a host no pool lists.

    Raises:
        ValueError: when no user is known for the host.
    """
    if "@" in host:
        return host
    pool = ssh_pool_of(host, pools)
    if pool is not None:
        if not pool.settings.get("user"):
            raise ValueError(_no_user(f"pool {pool.name}"))
        return ssh_target(pool, host)
    if not os.environ.get("LARG_USER"):
        raise ValueError(_no_user(f"host {host}, which no ssh pool lists"))
    return f"{os.environ['LARG_USER']}@{host}"


def ssh_user(pools: list[Pool]) -> str:
    """The login the ssh pools use ($LARG_USER first).

    Raises:
        ValueError: when the ssh pools name no user, or several.
    """
    if os.environ.get("LARG_USER"):
        return os.environ["LARG_USER"]
    users = sorted({p.settings["user"] for p in pools if p.kind == "ssh" and p.settings.get("user")})
    if not users:
        raise ValueError(_no_user("the ssh pools"))
    if len(users) > 1:
        raise ValueError(f"the ssh pools use several logins ({', '.join(users)}); export LARG_USER to pick one")
    return users[0]


def ray_address(pools: list[Pool]) -> str:
    """The dashboard address of the pool named ``ray`` (what `pls run --on ray` and `pls jobs --on ray` use).

    Raises:
        ValueError: when there is no such pool.
    """
    for pool in pools:
        if pool.name == "ray" and pool.kind == "ray" and pool.settings.get("address"):
            return str(pool.settings["address"]).rstrip("/")
    raise ValueError("no Ray pool: `pls setup --add ray --kind ray --address http://<head>:8265` adds one")


def _toml_value(val: object) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, list):
        return "[" + ", ".join(_toml_value(v) for v in val) + "]"
    return '"' + str(val).replace("\\", "\\\\").replace('"', '\\"') + '"'


def pool_table(name: str, settings: dict) -> str:
    """The TOML table of one pool."""
    return f"[compute.{name}]\n" + "".join(f"{k} = {_toml_value(v)}\n" for k, v in settings.items())


def write_pool(path: Path, name: str, settings: dict, update: bool = False) -> None:
    """Add (or with ``update``, replace) ``[compute.<name>]`` in the user's compute file, keeping the rest of it.

    Raises:
        ValueError: when the pool exists and ``update`` is false.
        tomllib.TOMLDecodeError: when the file would not parse (nothing is written).
    """
    text = path.read_text() if path.is_file() else ""
    found = re.search(rf"^\[compute\.{re.escape(name)}\][ \t]*$", text, re.M)
    if found and not update:
        raise ValueError(f"pool {name} is already in {path}; pass --update to replace it")
    if found:
        nxt = re.compile(r"^\[", re.M).search(text, found.end())
        tail = "\n" + text[nxt.start() :] if nxt else ""
        text = text[: found.start()] + pool_table(name, settings) + tail
    else:
        text = text.rstrip("\n") + ("\n\n" if text.strip() else "") + pool_table(name, settings)
    tomllib.loads(text)  # never write a file the inventory cannot read
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


ADD_USAGE = """pls setup --add <name> --kind ssh --hosts <a,b> --user <login> [--domain <d>] [--pin device] [--render]
                       [--update]
       pls setup --add <name> --kind ray --address http://<head>:8265 [--update]"""


def cmd_add(argv: list[str]) -> None:
    """``add``: write an ssh or ray pool into the user's compute.local.toml."""
    ap = argparse.ArgumentParser(
        prog="pls setup --add", usage=ADD_USAGE, description="Add a pool to your gitignored compute.local.toml."
    )
    ap.add_argument("name")
    ap.add_argument("--kind", choices=["ssh", "ray"], required=True)
    ap.add_argument("--hosts", help="ssh: comma-separated host names (short names get --domain)")
    ap.add_argument("--user", help="ssh: your login on the hosts")
    ap.add_argument("--domain", help="ssh: appended to each short host name")
    ap.add_argument("--pin", choices=["cvd", "device"], help="ssh: how card runs pick their GPU (default cvd)")
    ap.add_argument(
        "--render", action="store_true", help="ssh: RT-core cards; scripts/larg/train.sh records in-process"
    )
    ap.add_argument("--address", help="ray: the dashboard address, http://<head>:8265")
    ap.add_argument("--update", action="store_true", help="replace the pool if it exists")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        ap.error(f"pool name {args.name!r}: use letters, digits, - and _")
    if args.kind == "ssh":
        hosts = [h.strip() for h in (args.hosts or "").split(",") if h.strip()]
        if not hosts or not args.user:
            ap.error("--kind ssh needs --hosts and --user")
        if args.address:
            ap.error("--address is for --kind ray")
        settings: dict = {"kind": "ssh", "hosts": hosts, "user": args.user}
        settings.update({k: v for k, v in (("domain", args.domain), ("pin", args.pin)) if v})
        if args.render:
            settings["render"] = True
    else:
        if not args.address:
            ap.error("--kind ray needs --address")
        if args.hosts or args.user or args.domain or args.pin or args.render:
            ap.error("--hosts, --user, --domain, --pin and --render are for --kind ssh")
        settings = {"kind": "ray", "address": args.address}
    path = local_config_path()
    try:
        write_pool(path, args.name, settings, args.update)
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        sys.exit(f"[res] {exc}")
    print(f"[res] wrote pool {args.name} to {path} (per-user, not tracked by git)")
    print(f"[res] check it: pls res status --pool {args.name}")


QUERIES = {"ssh-target": 1, "ssh-user": 0, "ssh-hosts": 0, "host-setting": 2, "ray-address": 0}


def main(argv: list[str] | None = None) -> None:
    """The shell helpers' queries and ``add``; see the module docstring."""
    argv = sys.argv[1:] if argv is None else argv
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "add":
        return cmd_add(rest)
    if QUERIES.get(cmd) != len(rest):
        sys.exit(__doc__)
    pools = load_pools()
    try:
        if cmd == "ssh-target":
            print(resolve_ssh_target(rest[0], pools))
        elif cmd == "ssh-user":
            print(ssh_user(pools))
        elif cmd == "ssh-hosts":
            for pool in (p for p in pools if p.kind == "ssh"):
                for h in pool.settings.get("hosts", []):
                    print(pool.name, h, ssh_target(pool, h) if pool.settings.get("user") else "")
        elif cmd == "host-setting":
            pool = ssh_pool_of(rest[0], pools)
            val = pool.settings.get(rest[1], "") if pool else ""
            print(_toml_value(val) if isinstance(val, bool) else val)
        else:
            print(ray_address(pools))
    except ValueError as exc:
        sys.exit(f"[res] {exc}")


if __name__ == "__main__":
    main()
