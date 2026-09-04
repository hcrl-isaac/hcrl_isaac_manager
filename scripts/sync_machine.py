"""Push this machine's workspace + Claude state to a peer box (``just sync <host>``).

The peer is authoritative for nothing: this is a one-way push from here to ``<host>``. Run it from the
machine you are leaving. It ships, in order:

1. **Code, git-aware.** Every checkout (the manager, each ``resources/<repo>``, each
   ``resources/<repo>/worktrees/<name>``) is brought to the same branch + commit on the peer -- commits
   are pushed straight into the peer's repo (``refs/sync/<branch>``), so unpushed work travels -- then
   only the dirty files (modified/untracked, deletions too) are rsynced on top. Gitignored extras that
   the code needs (per-repo ``.claude/``, ``.env.*``) ride along. ``workspace.yaml`` never does: it is
   per-machine by design.
2. **Claude state, path-rewritten.** Session transcripts, memory, ``SESSIONS.md``, scratchpads and
   ``~/.cluster_dev`` are copied with every absolute path translated to the peer's layout (home,
   manager dir, ``/tmp/claude-<uid>/<project-slug>``), and only where this side is newer, so a session
   continued over there is not clobbered by a stale copy from here.
3. **Deps, incrementally.** The peer's ``ilab`` (and ``booster-deploy/.venv``) are re-synced with uv --
   fast when nothing changed, a full ``just setup`` only when the venv is missing.

Safety: if the peer has TRACKED modifications in files this side is NOT also changing, the code step
refuses (``--force`` overrides); its untracked files are never touched, so they are not guarded.
``--dry-run`` prints every action without touching the peer.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

MANAGER_DIR = Path(__file__).resolve().parent.parent
HOME = Path.home()
# gitignored paths the code needs on the other side (relative to each checkout root)
REPO_EXTRAS = [".claude"]
MANAGER_EXTRAS = [
    "CLAUDE.md",  # the manager gitignores its own (/CLAUDE.md), so it is a plain file, not a tracked one
    ".claude",
    "scripts/.env.wandb",
    "scripts/ray/.env.ray",
    "scripts/cluster/config",
]
# sibling repos under $HOME synced the same way, with the gitignored dirs their code needs
EXTRA_REPOS = {"booster-deploy": ["models", "configs"]}
# never shipped: per-machine or rebuilt on the peer
SKIP_TOP = {"ilab", ".venv", "logs", "wandb", "artifacts", "__pycache__"}
TEXT_SUFFIXES = {
    ".sh",
    ".py",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".cfg",
    ".env",
}
REWRITE_MAX_BYTES = 4 << 20
# one multiplexed connection for the whole run: ~30 checkouts x 3 round-trips at ~350 ms each is
# otherwise the dominant cost, with almost no data moving
SSH_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "ControlMaster=auto",
    "-o",
    f"ControlPath={HOME}/.ssh/cm/sync-%C",
    "-o",
    "ControlPersist=120",
]
SSH_CMD = "ssh " + " ".join(shlex.quote(o) for o in SSH_OPTS)


def sh(cmd: list[str] | str, *, check: bool = True, capture: bool = True, dry: bool = False, env=None) -> str:
    """Run a local command; with ``dry`` only print it."""
    if dry:
        print(
            "  $",
            cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd),
        )
        return ""
    r = subprocess.run(cmd, shell=isinstance(cmd, str), text=True, capture_output=capture, env=env)
    if check and r.returncode:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{r.stderr or r.stdout}")
    return (r.stdout or "").rstrip("\n")


def remote(host: str, script: str, *, check: bool = True, dry: bool = False) -> str:
    """Run a bash snippet on the peer with the user-local tool dir on PATH."""
    full = f"export PATH=$HOME/.local/bin:$PATH; {script}"
    if dry:
        print(f"  ssh {host}: {script.strip().splitlines()[0][:100]}...")
        return ""
    return sh(
        ["ssh", *SSH_OPTS, host, full],
        check=check,
    )


def slug(path: str) -> str:
    """Claude's project-dir name for a working directory (every non-alnum char becomes ``-``)."""
    return re.sub(r"[^A-Za-z0-9]", "-", path)


@dataclass
class Peer:
    host: str
    home: str
    uid: str
    manager: str

    @property
    def scratch_root(self) -> str:
        return f"/tmp/claude-{self.uid}/{slug(self.manager)}"

    @property
    def projects_dir(self) -> str:
        return f"{self.home}/.claude/projects/{slug(self.manager)}"


@dataclass
class Local:
    home: str = str(HOME)
    uid: str = str(os.getuid())
    manager: str = str(MANAGER_DIR)

    @property
    def scratch_root(self) -> str:
        return f"/tmp/claude-{self.uid}/{slug(self.manager)}"

    @property
    def projects_dir(self) -> str:
        return f"{self.home}/.claude/projects/{slug(self.manager)}"


def path_map(local: Local, peer: Peer) -> list[tuple[str, str]]:
    """Longest-first substitutions that translate this machine's absolute paths into the peer's."""
    pairs = [
        (local.scratch_root, peer.scratch_root),
        (local.manager, peer.manager),
        (local.home, peer.home),
    ]
    return sorted({p for p in pairs if p[0] != p[1]}, key=lambda p: -len(p[0]))


def rewrite(text: str, pmap: list[tuple[str, str]]) -> str:
    for src, dst in pmap:
        text = text.replace(src, dst)
    return text


# ----------------------------------------------------------------------------------------------- code


@dataclass
class Checkout:
    rel: str  # path relative to the manager dir ("" for the manager itself)
    repo_rel: str  # the main checkout that owns the object store (== rel unless a worktree)
    branch: str
    sha: str
    dirty: list[str] = field(default_factory=list)  # modified/added/untracked, relative to the checkout
    deleted: list[str] = field(default_factory=list)
    root: Path = MANAGER_DIR  # local dir the checkout lives under (an extra repo lives under $HOME)
    extras: list[str] = field(default_factory=list)  # gitignored paths shipped alongside the dirty files

    @property
    def path(self) -> Path:
        return self.root / self.rel if self.rel else self.root


def git(path: Path, *args: str) -> str:
    return sh(["git", "-C", str(path), *args])


def scan_checkout(
    rel: str, repo_rel: str, root: Path = MANAGER_DIR, extras: list[str] | None = None
) -> Checkout | None:
    path = root / rel if rel else root
    if not (path / ".git").exists():
        return None
    branch = git(path, "rev-parse", "--abbrev-ref", "HEAD")
    sha = git(path, "rev-parse", "HEAD")
    co = Checkout(rel, repo_rel, branch, sha, root=root, extras=REPO_EXTRAS if extras is None else extras)
    for line in git(path, "status", "--porcelain", "--untracked-files=all").splitlines():
        code, name = line[:2], line[3:]
        if " -> " in name:
            name = name.split(" -> ", 1)[1]
        top = name.split("/", 1)[0]
        # nested checkouts (resources/, worktrees/) are scanned on their own; per-machine dirs stay home
        if top in SKIP_TOP or top in ("resources", "worktrees") or name == "workspace.yaml":
            continue
        # build metadata that `pip install -e` rewrites per machine is not a change worth carrying
        if "__pycache__" in name or name.endswith(".pyc") or ".egg-info/" in name:
            continue
        (co.deleted if "D" in code else co.dirty).append(name)
    return co


def scan_all() -> list[Checkout]:
    cos = [c for c in [scan_checkout("", "", extras=MANAGER_EXTRAS)] if c]
    for name, extras in EXTRA_REPOS.items():
        if (c := scan_checkout("", "", root=HOME / name, extras=REPO_EXTRAS + extras)) is not None:
            cos.append(c)
    for repo in sorted((MANAGER_DIR / "resources").glob("*/")):
        rel = f"resources/{repo.name}"
        if (c := scan_checkout(rel, rel)) is None:
            continue
        cos.append(c)
        for wt in sorted((repo / "worktrees").glob("*/")):
            if wt.is_symlink():  # sibling links (hcrl_robots -> ../../hcrl_robots), not checkouts
                continue
            if (wc := scan_checkout(f"{rel}/worktrees/{wt.name}", rel)) is not None:
                cos.append(wc)
    return cos


def sync_checkout(co: Checkout, peer: Peer, pmap: list[tuple[str, str]], *, force: bool, dry: bool) -> None:
    rroot = rewrite(str(co.root), pmap)
    rpath = f"{rroot}/{co.rel}" if co.rel else rroot
    rrepo = f"{rroot}/{co.repo_rel}" if co.repo_rel else rroot
    label = co.rel or f"<{co.root.name}>"
    probe = remote(
        peer.host,
        f"test -d {shlex.quote(rrepo)}/.git || {{ echo missing; exit 0; }}; "
        f"cd {shlex.quote(rpath)} 2>/dev/null && git rev-parse HEAD || echo absent",
        dry=dry,
    )
    if not dry and probe == "missing":
        print(f"[sync] {label}: repo missing on peer (run `just setup` there first) -- skipped")
        return
    print(f"[sync] {label}: {co.branch}@{co.sha[:8]}  dirty={len(co.dirty)} deleted={len(co.deleted)}")
    # 1) commits: push HEAD into the peer's object store under a sync ref -- unless it already has it.
    # --no-verify skips git-lfs's pre-push hook: LFS objects come to the peer from origin, not from here.
    if dry or probe != co.sha:
        sh(
            [
                "git",
                "-C",
                str(co.path),
                "push",
                "-q",
                "-f",
                "--no-verify",
                f"{peer.host}:{rrepo}",
                f"HEAD:refs/sync/{co.branch}",
            ],
            dry=dry,
            env={**os.environ, "GIT_SSH_COMMAND": SSH_CMD},
        )
    # 2) checkout: create the worktree if needed, then land on the branch at exactly this sha
    is_wt = co.rel != co.repo_rel
    ours = "\\n".join(co.dirty + co.deleted)
    guard = (
        ""
        if force
        else f"""
        theirs=$(git status --porcelain | grep -v '^??' | cut -c4- | sed 's/.* -> //' \\
            | grep -v -E '^(resources/|worktrees/|ilab/|\\.venv/|logs/|wandb/|artifacts/|workspace\\.yaml$)|__pycache__|\\.pyc$|\\.egg-info/' || true)
        extra=$(printf '%s\\n' "$theirs" | grep -vxF -f <(printf '%b\\n' {shlex.quote(ours)}) | grep -v '^$' || true)
        if [ -n "$extra" ]; then echo "REFUSE: peer has its own uncommitted changes in {label}:"; echo "$extra"; exit 3; fi"""
    )
    script = f"""
        set -e
        if [ ! -d {shlex.quote(rpath)} ]; then
            {"git -C " + shlex.quote(rrepo) + " worktree add -q " + shlex.quote(rpath) + " -B " + shlex.quote(co.branch) + " refs/sync/" + shlex.quote(co.branch) if is_wt else "false"}
        fi
        cd {shlex.quote(rpath)}
        {guard}
        if [ "$(git rev-parse HEAD)" != "{co.sha}" ] || [ "$(git rev-parse --abbrev-ref HEAD)" != {shlex.quote(co.branch)} ]; then
            git checkout -q -f -B {shlex.quote(co.branch)} {co.sha}
        fi
        {" ".join("rm -f " + shlex.quote(p) + ";" for p in co.deleted)}
    """
    if dry:
        sh(["ssh", peer.host, script], dry=True)
    else:
        r = subprocess.run(
            ["ssh", *SSH_OPTS, peer.host, f"export PATH=$HOME/.local/bin:$PATH; {script}"],
            text=True,
            capture_output=True,
        )
        if "REFUSE:" in r.stdout:
            print(r.stdout.rstrip())
            raise SystemExit("[sync] aborting; re-run with --force to overwrite the peer's changes")
        if r.returncode:
            raise SystemExit(f"[sync] {label}: peer checkout failed ({r.returncode}):\n{r.stderr or r.stdout}")
    # 3) dirty files + gitignored extras
    files = list(co.dirty) + [e for e in co.extras if (co.path / e).exists()]
    if files:
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("\n".join(files) + "\n")
        sh(
            [
                "rsync",
                "-a",
                "-r",
                "--relative",
                "--files-from",
                f.name,
                "-e",
                SSH_CMD,
                f"{co.path}/",
                f"{peer.host}:{rpath}/",
            ],
            dry=dry,
        )
        os.unlink(f.name)
        retarget_symlinks(co.path, rpath, files, peer, pmap, dry=dry)


def retarget_symlinks(local_root: Path, remote_root: str, subpaths: list[str], peer: Peer, pmap, *, dry: bool) -> None:
    """Absolute symlinks that point into a mapped tree are re-pointed at the peer's copy of that tree."""
    cmds = []
    for sub in subpaths:
        base = local_root / sub
        links = [base] if base.is_symlink() else [p for p in base.rglob("*") if p.is_symlink()] if base.is_dir() else []
        for link in links:
            target = os.readlink(link)
            if os.path.isabs(target) and (mapped := rewrite(target, pmap)) != target:
                dst = f"{remote_root}/{link.relative_to(local_root)}"
                cmds.append(f"ln -sfn {shlex.quote(mapped)} {shlex.quote(dst)}")
    if cmds:
        print(f"[sync]   retargeting {len(cmds)} symlink(s)")
        remote(peer.host, "; ".join(cmds), dry=dry)


# ------------------------------------------------------------------------------------------- claude


def is_text(p: Path) -> bool:
    return p.suffix in TEXT_SUFFIXES or p.suffix == ".jsonl" or not p.suffix


def rewrite_tree(src: Path, pmap: list[tuple[str, str]], staging: Path) -> Path:
    """Stage rewritten copies of ``src``'s text files only (mtimes preserved); binaries are not touched."""
    dst = staging / src.name
    for p in src.rglob("*"):
        if p.is_dir() or p.is_symlink() or not is_text(p):
            continue
        data = p.read_bytes()
        if p.suffix == ".jsonl":  # transcripts exceed the cap; they are line-oriented, so stream them
            data = b"".join(rewrite(line.decode(), pmap).encode() for line in data.splitlines(keepends=True))
        elif len(data) <= REWRITE_MAX_BYTES:
            try:
                data = rewrite(data.decode(), pmap).encode()
            except UnicodeDecodeError:
                pass
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        st = p.stat()
        os.utime(out, (st.st_atime, st.st_mtime))
    return dst


def push_tree(local_dir: Path, remote_dir: str, peer: Peer, pmap, staging: Path, *, dry: bool) -> None:
    """Two passes so multi-GB scratch binaries stream straight from source and never hit a staging copy."""
    if not local_dir.exists():
        return
    print(f"[sync] claude: {local_dir} -> {peer.host}:{remote_dir}")
    remote(peer.host, f"mkdir -p {shlex.quote(remote_dir)}", dry=dry)
    text_globs = [f"--exclude=*{s}" for s in TEXT_SUFFIXES | {".jsonl"}]
    sh(["rsync", "-a", "--update", "-e", SSH_CMD, *text_globs, f"{local_dir}/", f"{peer.host}:{remote_dir}/"], dry=dry)
    if not dry:
        staged = rewrite_tree(local_dir, pmap, staging)
        if staged.exists():
            sh(["rsync", "-a", "--update", "-e", SSH_CMD, f"{staged}/", f"{peer.host}:{remote_dir}/"])


def sync_claude(local: Local, peer: Peer, pmap, *, dry: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="sync-claude-") as tmp:
        staging = Path(tmp)
        proj = Path(local.projects_dir)
        # transcripts + memory (rewritten); tool-results and file-history are session-local noise
        if proj.exists():
            sel = staging / "proj"
            sel.mkdir()
            for p in proj.glob("*.jsonl"):
                (sel / p.name).write_bytes(p.read_bytes())
                os.utime(sel / p.name, (p.stat().st_atime, p.stat().st_mtime))
            if (proj / "memory").is_dir():
                sh(["cp", "-a", str(proj / "memory"), str(sel / "memory")])
            push_tree(sel, peer.projects_dir, peer, pmap, staging / "s1", dry=dry)
        for name in ("SESSIONS.md", "settings.json", "history.jsonl"):
            f = HOME / ".claude" / name
            if f.exists():
                sel = staging / f"f-{name}"
                sel.mkdir()
                sh(["cp", "-a", str(f), str(sel / name)])
                push_tree(
                    sel,
                    f"{peer.home}/.claude",
                    peer,
                    pmap,
                    staging / f"s-{name}",
                    dry=dry,
                )
        push_tree(
            HOME / ".cluster_dev",
            f"{peer.home}/.cluster_dev",
            peer,
            pmap,
            staging / "s2",
            dry=dry,
        )
        scratch = Path(local.scratch_root)
        if scratch.is_dir():
            push_tree(scratch, peer.scratch_root, peer, pmap, staging / "s3", dry=dry)


# --------------------------------------------------------------------------------------------- deps


def ensure_deps(peer: Peer, *, dry: bool) -> None:
    """Bring the peer's venvs up to date without a rebuild: uv is a no-op when nothing changed."""
    m = shlex.quote(peer.manager)
    script = f"""
        set -e
        command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
        command -v just >/dev/null || uv tool install -q rust-just
        export PATH=$HOME/.local/bin:$PATH VIRTUAL_ENV=""
        cd {m}
        if [ ! -x ilab/bin/python ]; then
            echo "[deps] no ilab venv on peer -> full 'just setup' (this is the slow path)"
            OMNI_KIT_ACCEPT_EULA=YES just setup < /dev/null
            exit 0
        fi
        echo "[deps] manager deps (uv sync)"; uv sync -q
        echo "[deps] workspace packages (editable re-install, fast when unchanged)"
        PY=ilab/bin/python
        uv pip install -q --python $PY --torch-backend cu128 --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match -e "resources/hcrl_isaaclab[isaacsim]"
        for d in resources/robot_rl resources/*_tasks resources/*_robots resources/holosoma/src/holosoma_retargeting; do
            if [ -d "$d" ] && {{ [ -f "$d/setup.py" ] || [ -f "$d/pyproject.toml" ]; }}; then
                uv pip install -q --python $PY --torch-backend cu128 --extra-index-url https://pypi.nvidia.com -e "$d"
            fi
        done
        if [ -f $HOME/booster-deploy/pyproject.toml ]; then
            echo "[deps] booster-deploy venv"
            cd $HOME/booster-deploy
            [ -x .venv/bin/python ] || uv venv -q --python 3.11 .venv
            uv pip install -q --python .venv/bin/python --torch-backend cpu -e ".[test,sim,motions]"
        fi
        echo "[deps] done"
    """
    print(f"[sync] deps on {peer.host}")
    out = sh(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            peer.host,
            f"export PATH=$HOME/.local/bin:$PATH; {script}",
        ],
        check=False,
        dry=dry,
    )
    if out:
        print("\n".join("  " + line for line in out.splitlines()[-12:]))


# --------------------------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host", help="ssh host of the peer (e.g. hcrl2, ebuntu)")
    ap.add_argument("--dry-run", action="store_true", help="print what would happen; touch nothing")
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite the peer's own uncommitted changes",
    )
    ap.add_argument("--no-code", action="store_true")
    ap.add_argument("--no-claude", action="store_true")
    ap.add_argument("--no-deps", action="store_true")
    ap.add_argument(
        "--artifacts",
        action="store_true",
        help="also rsync artifacts/ (large exported policies)",
    )
    args = ap.parse_args()
    dry = args.dry_run

    local = Local()
    if args.host in (os.uname().nodename, "localhost"):
        raise SystemExit("[sync] refusing to sync a machine to itself")
    probe = sh(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            args.host,
            'printf "%s\\n%s\\n" "$HOME" "$(id -u)"',
        ]
    )
    rhome, ruid = probe.splitlines()[:2]
    peer = Peer(
        args.host,
        rhome,
        ruid,
        os.environ.get("SYNC_REMOTE_MANAGER", f"{rhome}/{MANAGER_DIR.name}"),
    )
    pmap = path_map(local, peer)
    print(f"[sync] {local.manager} -> {peer.host}:{peer.manager}" + ("  (DRY RUN)" if dry else ""))
    for src, dst in pmap:
        print(f"[sync]   rewrite {src} -> {dst}")

    if not args.no_code:
        for co in scan_all():
            sync_checkout(co, peer, pmap, force=args.force, dry=dry)
        if args.artifacts and (MANAGER_DIR / "artifacts").is_dir():
            sh(
                [
                    "rsync",
                    "-a",
                    "--update",
                    f"{MANAGER_DIR}/artifacts/",
                    f"{peer.host}:{peer.manager}/artifacts/",
                ],
                dry=dry,
            )
    if not args.no_claude:
        sync_claude(local, peer, pmap, dry=dry)
    if not args.no_deps:
        ensure_deps(peer, dry=dry)
    print("[sync] done")


if __name__ == "__main__":
    main()
