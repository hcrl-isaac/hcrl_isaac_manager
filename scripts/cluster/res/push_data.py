#!/usr/bin/env python3
"""`pls stage --on <ssh host or pool> data`: add the data repos' new files to an ssh box's copy, as on a cluster.

A box run links each data repo from the box workspace's copy, else from its ``--root`` install's; the new files go
into whichever copy the box's runs read, never deleting anything there.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from evaluate import Target, local_workspace, make_target
from inventory import Pool, load_pools

DATA_REPOS = ("motion_datasets",)  # the cluster push's TREE_DATA_REPOS (cluster_dev/trees.sh)
# the training files a task loads (the cluster push's allowlist)
INCLUDE = ("*.pt", "*.arena.json", "*.courts.json", "*.manifest.json")
_LFS_POINTER = b"version https://git-lfs"


def lfs_pointers(src: str) -> list[str]:
    """Files of ``src`` (outside ``worktrees/``) that are Git LFS pointers, as rsync patterns rooted at ``src``."""
    found = []
    for root, dirs, files in os.walk(src):
        if root == src and "worktrees" in dirs:
            dirs.remove("worktrees")
        for name in files:
            path = os.path.join(root, name)
            if os.path.getsize(path) <= 1024:
                with open(path, "rb") as f:
                    if f.read(len(_LFS_POINTER)) == _LFS_POINTER:
                        found.append("/" + os.path.relpath(path, src))
    return found


def hosts_of(spec: str, pools: list[Pool]) -> list[tuple[Pool, str]]:
    """The ssh hosts ``spec`` names: one host of an ssh pool, or every host of an ssh pool (prefix match).

    Args:
        spec: A host name or an ssh pool name.
        pools: Configured pools.

    Returns:
        (pool, host) pairs.
    """
    ssh = [p for p in pools if p.kind == "ssh"]
    by_host = [(p, spec) for p in ssh if spec in p.settings.get("hosts", [])]
    if by_host:
        return by_host[:1]
    by_pool = [(p, h) for p in ssh if p.name.startswith(spec) for h in p.settings.get("hosts", [])]
    if not by_pool:
        names = sorted({p.name for p in ssh} | {h for p in ssh for h in p.settings.get("hosts", [])})
        sys.exit(f"[stage] --on {spec}: not an ssh host or pool ({', '.join(names) or 'none configured'})")
    return by_pool


def destination(t: Target, repo: str, root: str) -> str:
    """The box's copy of ``repo`` its runs read: the workspace's when it has one, else ``--root``'s, else the
    workspace's (created).

    Args:
        t: The host's target (ssh options and workspace).
        repo: Data repo name.
        root: An Isaac Lab install root on the box, or "".
    """
    q = shlex.quote
    homes = [t.workspace, *([root] if root else [])]
    probe = "; ".join(f"[ -d {q(h)}/resources/{q(repo)} ] && {{ echo {q(h)}; exit 0; }}" for h in homes)
    out = subprocess.run(["ssh", *t.ssh_opts, t.ssh, f"{probe}; exit 0"], capture_output=True, text=True)
    found = [line for line in out.stdout.splitlines() if line in homes]
    return f"{found[0] if found else t.workspace}/resources/{repo}"


def push(t: Target, src: str, dest: str) -> int:
    """Rsync ``src``'s training files (no LFS pointers, no worktrees) into ``dest`` on the box; never deletes."""
    skip = "\n".join(lfs_pointers(src))
    rules = [f"--include={p}" for p in INCLUDE]
    cmd = [
        "rsync", "-rlpt", "--info=progress2", "--exclude-from=-", "--exclude=/worktrees/", "--include=*/", *rules,
        "--exclude=*", "--prune-empty-dirs", "-e", "ssh " + " ".join(shlex.quote(o) for o in t.ssh_opts),
        f"--rsync-path=mkdir -p {shlex.quote(dest)} && rsync", f"{src}/", f"{t.ssh}:{dest}/",
    ]  # fmt: skip
    return subprocess.run(cmd, input=skip, text=True).returncode


def main() -> None:
    ap = argparse.ArgumentParser(prog="pls stage --on <ssh host or pool> data", description=__doc__)
    ap.add_argument("--on", required=True, help="an ssh host, or an ssh pool (every host of it)")
    ap.add_argument("--root", default="", help="the Isaac Lab install root its runs use (`pls run --root`)")
    args = ap.parse_args()
    resources = os.path.join(local_workspace(), "resources")
    failed = []
    for pool, host in hosts_of(args.on, load_pools()):
        t = make_target(pool, host, 0)
        for repo in DATA_REPOS:
            src = os.path.join(resources, repo)
            if not os.path.isdir(src):
                continue
            dest = destination(t, repo, args.root)
            print(f"[stage] adding new {repo} files -> {host}:{dest} (never deleted)", file=sys.stderr)
            if push(t, src, dest) != 0:
                failed.append(f"{host}:{repo}")
    if failed:
        sys.exit(f"[stage] could not add the data repos' new files on {', '.join(failed)}")


if __name__ == "__main__":
    main()
