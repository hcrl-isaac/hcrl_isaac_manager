#!/usr/bin/env python3
"""`pls stage --on <ssh host or pool> data`: add the data repos' new files to an ssh box's copy, as on a cluster.

A box run links each data repo from the box workspace's copy, else from its ``--root`` install's; the new files go
into whichever copy the box's runs read, never deleting anything there. With ``--wt SET`` they come from the set's
worktree of each data repo and only add files: one the box's copy already has is never replaced.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from evaluate import Target, local_workspace, make_target
from inventory import Pool, load_pools
from sshslot import SSH

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
    out = subprocess.run([SSH, *t.ssh_opts, t.ssh, f"{probe}; exit 0"], capture_output=True, text=True)
    found = [line for line in out.stdout.splitlines() if line in homes]
    return f"{found[0] if found else t.workspace}/resources/{repo}"


def _rsync(t: Target, src: str, dest: str, opts: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    """Rsync ``src``'s training files (no LFS pointers, no worktrees) into ``dest`` on the box with extra ``opts``."""
    skip = "\n".join(lfs_pointers(src))
    rules = [f"--include={p}" for p in INCLUDE]
    cmd = [
        "rsync", "-rlpt", *opts, "--exclude-from=-", "--exclude=/worktrees/", "--include=*/", *rules,
        "--exclude=*", "--prune-empty-dirs", "-e", SSH + " " + " ".join(shlex.quote(o) for o in t.ssh_opts),
        f"--rsync-path=mkdir -p {shlex.quote(dest)} && rsync", f"{src}/", f"{t.ssh}:{dest}/",
    ]  # fmt: skip
    return subprocess.run(cmd, input=skip, text=True, capture_output=capture)


def push(t: Target, src: str, dest: str, only_new: bool = False) -> int:
    """Rsync ``src``'s training files into ``dest`` on the box; never deletes, and with ``only_new`` never replaces."""
    return _rsync(t, src, dest, ["--info=progress2", *(["--ignore-existing"] if only_new else [])]).returncode


def changed(t: Target, src: str, dest: str) -> list[str]:
    """``src``'s training files that the box's ``dest`` holds under the same name with different content."""
    out = _rsync(t, src, dest, ["-n", "-i", "--checksum"], capture=True)
    if out.returncode != 0:
        sys.exit(f"[stage] could not compare {src} with {t.ssh}:{dest}: {out.stderr.strip()[-300:]}")
    # a send is "<f" (">f" locally); a new file's flags are all "+", so the rest are files the box has, changed
    return [line.split(None, 1)[1] for line in out.stdout.splitlines() if line[:2] in ("<f", ">f") and line[2] != "+"]


def source(resources: str, repo: str, wt: str) -> str:
    """The local copy of ``repo`` to push: the set's worktree with ``wt`` (or "" when it has none), else the checkout."""
    path = os.path.join(resources, repo, "worktrees", wt) if wt else os.path.join(resources, repo)
    return path if os.path.isdir(path) else ""


def main() -> None:
    ap = argparse.ArgumentParser(prog="pls stage --on <ssh host or pool> data", description=__doc__)
    ap.add_argument("--on", required=True, help="an ssh host, or an ssh pool (every host of it)")
    ap.add_argument("--root", default="", help="the Isaac Lab install root its runs use (`pls run --root`)")
    ap.add_argument("--wt", default="", help="add new files from this worktree set's data worktrees")
    ap.add_argument(
        "--keep-existing",
        action="store_true",
        help="with --wt: add the new files although some of the set's differ from the box's copy (which stays)",
    )
    args = ap.parse_args()
    if args.keep_existing and not args.wt:
        sys.exit("[stage] --keep-existing goes with --wt SET")
    resources = os.path.join(local_workspace(), "resources")
    srcs = {repo: source(resources, repo, args.wt) for repo in DATA_REPOS}
    if args.wt and not any(srcs.values()):
        sys.exit(f"[stage] no data repo has worktrees/{args.wt} ({', '.join(DATA_REPOS)})")
    plan = []
    for pool, host in hosts_of(args.on, load_pools()):
        t = make_target(pool, host, 0)
        plan += [(host, t, repo, src, destination(t, repo, args.root)) for repo, src in srcs.items() if src]
    # every host is checked before any is pushed, so a refusal leaves the pool as it was
    diffs = {(host, repo): changed(t, src, dest) for host, t, repo, src, dest in plan} if args.wt else {}
    if (bad := {k: v for k, v in diffs.items() if v}) and not args.keep_existing:
        lines = "; ".join(f"{h}:{r} {', '.join(v)}" for (h, r), v in bad.items())
        sys.exit(
            f"[stage] files of {args.wt} differ from the box's copy, which --wt never replaces ({lines}); nothing"
            " pushed. Give them new names, or pass --keep-existing"
        )
    for (h, r), v in bad.items():
        for f in v:
            print(f"[stage] WARN: {h}:{r}/{f} differs; it keeps its copy (--keep-existing)", file=sys.stderr)
    failed = []
    for host, t, repo, src, dest in plan:
        print(f"[stage] adding new {repo} files from {src} -> {host}:{dest} (never deleted)", file=sys.stderr)
        if push(t, src, dest, only_new=bool(args.wt)) != 0:
            failed.append(f"{host}:{repo}")
    if failed:
        sys.exit(f"[stage] could not add the data repos' new files on {', '.join(failed)}")


if __name__ == "__main__":
    main()
