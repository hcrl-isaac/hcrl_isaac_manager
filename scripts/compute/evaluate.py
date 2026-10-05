"""`just res eval`: run a one-off script on a leased GPU of a local or ssh pool, with its checkpoints and code."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import checkpoints as ck
import leases as ls
from inventory import COMPUTE_DIR, Pool
from probe import SSH_OPTS

MANAGER_DIR = COMPUTE_DIR.parents[1]
SUPPORTED = ("local", "ssh")
CKPT_CACHE = Path.home() / ".cache" / "hcrl_res" / "checkpoints"
TRACEBACK = "Traceback (most recent call last)"
TIMED_OUT = 124
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class Target:
    """Where the script runs: one card of a local or ssh pool.

    Args:
        pool: Pool name.
        kind: ``local`` or ``ssh``.
        host: Host name as the probe reports it.
        gpu: Physical GPU index.
        ssh: ``user@fqdn`` for ssh pools, "" for local.
        workspace: Manager checkout on the target (holds ilab/ and the asset repos).
        scratch: Per-box scratch dir that stage dirs and code snapshots go under.
        pin: ``cvd`` masks CUDA_VISIBLE_DEVICES to the card; ``device`` leaves it unmasked and the script
            selects the card from RES_EVAL_DEVICE.
    """

    pool: str
    kind: str
    host: str
    gpu: int
    ssh: str = ""
    workspace: str = ""
    scratch: str = ""
    pin: str = "cvd"


def local_workspace() -> str:
    """The manager checkout holding resources/: this tree, or the main checkout when run from a worktree."""
    if (MANAGER_DIR / "resources").is_dir():
        return str(MANAGER_DIR)
    out = subprocess.run(
        ["git", "-C", str(MANAGER_DIR), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return str(Path(out).parent) if out else str(MANAGER_DIR)


def make_target(pool: Pool, host: str, gpu: int) -> Target:
    """Target for one card; refuses pools whose backend `eval` cannot run on yet.

    Args:
        pool: The card's pool.
        host: Host name from the probe.
        gpu: Physical GPU index.

    Returns:
        The target, with workspace, scratch and pin from the pool settings or their defaults.
    """
    if pool.kind not in SUPPORTED:
        hint = {"ray": "use `just ray job`", "slurm": "use a dev-node step (`just cluster <c> develop exec`)"}
        sys.exit(f"[res] eval does not run on {pool.kind} pools yet; {hint.get(pool.kind, 'pick a local/ssh card')}")
    s = pool.settings
    pin = s.get("pin", "cvd")
    if pin not in ("cvd", "device"):
        sys.exit(f"[res] pool {pool.name}: pin must be 'cvd' or 'device', not {pin!r}")
    if pool.kind == "local":
        ws = s.get("workspace") or local_workspace()
        scratch = s.get("scratch", f"{Path.home()}/tmp")
        return Target(pool.name, "local", host, gpu, workspace=ws, scratch=scratch, pin=pin)
    user = s.get("user") or os.environ.get("LARG_USER", "")
    if not user:
        sys.exit(f"[res] no ssh user for pool {pool.name}: set user under [compute.{pool.name}] in compute.local.toml")
    fqdn = host if "." in host or not s.get("domain") else f"{host}.{s['domain']}"
    return Target(
        pool.name,
        "ssh",
        host,
        gpu,
        ssh=f"{user}@{fqdn}",
        workspace=s.get("workspace", f"/var/local/{user}/hcrl_isaac_manager"),
        scratch=s.get("scratch", f"/var/local/{user}"),
        pin=pin,
    )


def parse_env(pairs: list[str]) -> dict[str, str]:
    """``KEY=VAL`` pairs as a dict; exits on a malformed one before anything is claimed."""
    out = {}
    for pair in pairs:
        key, sep, val = pair.partition("=")
        if not sep or not _ENV_KEY.match(key):
            sys.exit(f"[res] --env {pair!r}: expected KEY=VALUE")
        out[key] = val
    return out


def parse_checkpoints(specs: list[str]) -> list[ck.CheckpointRef]:
    """Parsed ``--checkpoint`` specs; exits on a bad or duplicate name before anything is claimed."""
    refs = []
    for spec in specs:
        try:
            refs.append(ck.parse(spec))
        except ValueError as exc:
            sys.exit(f"[res] --checkpoint {exc}")
    names = [r.name for r in refs]
    if len(set(names)) != len(names):
        sys.exit(f"[res] --checkpoint names must differ (got {', '.join(names)}); use NAME=<ref>")
    return refs


def wandb_env() -> dict[str, str]:
    """W&B credentials from scripts/.env.wandb (KEY=VAL lines), falling back to the caller's environment."""
    env = {k: os.environ[k] for k in ("WANDB_API_KEY", "WANDB_USERNAME", "WANDB_ENTITY") if os.environ.get(k)}
    path = Path(local_workspace()) / "scripts" / ".env.wandb"
    if path.is_file():
        for line in path.read_text().splitlines():
            key, sep, val = line.strip().removeprefix("export ").partition("=")
            if sep and _ENV_KEY.match(key):
                env[key] = val.strip().strip("'\"")
    return env


def fetch_checkpoint(ref: ck.CheckpointRef) -> str:
    """Local path of a checkpoint: the given path, or a W&B download into the manager box's cache."""
    if ref.local:
        return ref.local
    ws = local_workspace()
    py = os.path.join(ws, "ilab", "bin", "python")
    cmd = [py, str(COMPUTE_DIR / "checkpoints.py"), os.path.join(ws, "resources", "hcrl_isaaclab"), str(CKPT_CACHE)]
    print(f"[res] fetching {ref.name} from W&B run {ref.run_path} {ref.model or '(latest)'}", file=sys.stderr)
    res = subprocess.run(
        [*cmd, ref.run_path, ref.model], capture_output=True, text=True, env={**os.environ, **wandb_env()}
    )
    lines = res.stdout.strip().splitlines()
    if res.returncode != 0 or not lines or not os.path.isfile(lines[-1]):
        sys.exit(f"[res] could not fetch {ref.name} ({ref.run_path}): {(res.stderr.strip().splitlines() or ['?'])[-1]}")
    return lines[-1]


def local_code(workspace: str, wt: str) -> dict[str, str]:
    """The package repos a run imports (core, RL package, ``*_tasks``), as worktree-set or main checkout paths.

    Args:
        workspace: Manager checkout whose resources/ holds the repos.
        wt: Worktree-set name, or "" for the main checkouts.

    Returns:
        Repo name -> path; asset repos (``*_robots``) are left to the target's own copies.
    """
    sys.path.insert(0, str(COMPUTE_DIR.parent))
    from worktree_env import workspace_repos

    resources = os.path.join(workspace, "resources")
    code, overridden = {}, False
    for repo in workspace_repos(resources):
        main = os.path.join(resources, repo)
        if repo.endswith("_robots") or not os.path.isdir(main):
            continue
        wdir = os.path.join(main, "worktrees", wt) if wt else ""
        overridden |= bool(wdir) and os.path.isdir(wdir)
        code[repo] = wdir if wdir and os.path.isdir(wdir) else main
    if wt and not overridden:
        sys.exit(f"[res] --wt {wt}: no repo under {resources} has worktrees/{wt}")
    return code


def _git(src: str, *args: str) -> bytes:
    return subprocess.run(["git", "-C", src, *args], capture_output=True, check=True).stdout


def code_files(src: str) -> list[bytes]:
    """Tracked and untracked non-ignored files of a checkout that exist on disk (relative, as git lists them)."""
    files = _git(src, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")
    return [f for f in files if f and os.path.lexists(os.path.join(src, f.decode()))]


def _changed_paths(status: bytes) -> list[bytes]:
    """Paths in ``git status --porcelain=v1 -z`` output; a rename or copy counts once (its source entry is skipped)."""
    entries, paths, skip = status.split(b"\0"), [], False
    for entry in entries:
        if skip:
            skip = False
            continue
        if len(entry) > 3:
            paths.append(entry[3:])
            skip = entry[:1] in (b"R", b"C")
    return paths


def code_fingerprint(src: str) -> tuple[str, str]:
    """Content fingerprint of a checkout, and its commit and dirty count for the stage MANIFEST.

    Args:
        src: A git checkout or worktree.

    Returns:
        The fingerprint (HEAD tree, status, and the content, mode or link target of every changed path) and a
        ``<commit> dirty=<n>`` description.
    """
    tree = _git(src, "rev-parse", "HEAD^{tree}").strip()
    status = _git(src, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    h = hashlib.sha256(tree + b"\0" + status)
    changed = _changed_paths(status)
    for rel in sorted(changed):
        path = os.path.join(src, rel.decode())
        if os.path.islink(path):
            h.update(os.readlink(path).encode())
        elif os.path.isfile(path):
            h.update(oct(os.stat(path).st_mode).encode() + Path(path).read_bytes())
    commit = _git(src, "rev-parse", "--short", "HEAD").decode().strip()
    return h.hexdigest()[:12], f"{commit} dirty={len(changed)}"


def _describe(src: str) -> str:
    """A checkout's commit and dirty count, or a note when it is not a git checkout."""
    try:
        return code_fingerprint(src)[1]
    except (subprocess.CalledProcessError, OSError):
        return "(not a git checkout)"


def runner_script(
    t: Target, stage: str, script: str, env_names: list[str], pythonpath: list[str], cwd: str = ""
) -> str:
    """The bash that runs on the target: a stage log, isolated caches, the given PYTHONPATH, the script's status.

    Args:
        t: The target.
        stage: The stage dir on the target.
        script: The script's path on the target.
        env_names: Names of the exported variables, for the log.
        pythonpath: PYTHONPATH entries on the target.
        cwd: Directory to run the script from (default: the target workspace).

    Returns:
        The run.sh text.
    """
    q = shlex.quote
    cache = f"{t.scratch}/res-eval/cache/{t.host}-gpu{t.gpu}"  # one lease per card, so per-card caches never race
    pin = (
        f"export CUDA_VISIBLE_DEVICES={t.gpu} RES_EVAL_DEVICE=cuda:0"
        if t.pin == "cvd"
        else f"unset CUDA_VISIBLE_DEVICES; export RES_EVAL_DEVICE=cuda:{t.gpu}"
    )
    return f"""#!/usr/bin/env bash
set -u
exec > >(tee -a {q(stage + "/log")}) 2>&1
cd {q(t.workspace)} || {{ echo "[res] no workspace {t.workspace} on {t.host}"; exit 97; }}
set -a; source {q(stage + "/env")}; set +a
rm -f {q(stage + "/env")}  # credentials stay in this process's environment only
{pin}
export TMPDIR={q(stage + "/tmp")} XDG_CACHE_HOME={q(cache + "/xdg")} OMNI_CACHE_DIR={q(cache + "/omni")}
export HCRL_ARTIFACT_ROOT={q(t.scratch + "/res-eval/artifacts")}
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$OMNI_CACHE_DIR" "$HCRL_ARTIFACT_ROOT"
gomp="$(ls ilab/lib/python3.11/site-packages/torch/lib/libgomp-*.so.1 2>/dev/null | head -1)"
[ -n "$gomp" ] && export LD_PRELOAD="$gomp${{LD_PRELOAD:+:$LD_PRELOAD}}"
export PYTHONPATH={q(":".join(pythonpath))}"${{PYTHONPATH:+:$PYTHONPATH}}"
cat {q(stage + "/MANIFEST")}
py="$PWD/ilab/bin/python"
echo "[res] {t.host}:gpu{t.gpu} ({t.pin}) python=$py env: {" ".join(env_names) or "-"}"
cd {q(cwd or t.workspace)} || exit 97
"$py" {q(script)} "$@"
"""


class Stage:
    """A fresh per-invocation directory on the target, plus the immutable code snapshots it links to."""

    def __init__(self, t: Target) -> None:
        self.t = t
        self.dir = f"{t.scratch}/res-eval/{uuid.uuid4().hex[:8]}"
        self.proc: subprocess.Popen | None = None
        self.dead = False

    def _ssh(self, cmd: str, **kw: object) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh", *SSH_OPTS, self.t.ssh, cmd], **kw)

    def sh(self, cmd: str) -> int:
        """Run a shell command on the target (here, for a local pool) and return its status."""
        if self.t.kind == "local":
            return subprocess.run(["bash", "-c", cmd]).returncode
        return self._ssh(cmd).returncode

    def make(self) -> None:
        """Create the stage dir (0700)."""
        q = shlex.quote
        if self.sh(f"mkdir -p -m 700 {q(self.dir)} && mkdir -p {q(self.dir)}/ckpt {q(self.dir)}/resources") != 0:
            sys.exit(f"[res] cannot create {self.dir} on {self.t.host}")

    def put(self, src: str, rel: str, mode: int = 0o600) -> str:
        """Copy a local file (symlinks resolved) to ``<stage>/<rel>`` and check that it landed whole.

        Args:
            src: Local file.
            rel: Destination relative to the stage dir.
            mode: File mode on the target.

        Returns:
            The destination path on the target.
        """
        dest = f"{self.dir}/{rel}"
        real = os.path.realpath(src)
        if self.t.kind == "local":
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(real, dest)
            os.chmod(dest, mode)
        else:
            self._ssh(f"mkdir -p {shlex.quote(os.path.dirname(dest))}")
            ssh = "ssh " + " ".join(shlex.quote(o) for o in SSH_OPTS)
            cmd = ["rsync", "-L", "-s", f"--chmod=F{mode:o}", "-e", ssh, real, f"{self.t.ssh}:{dest}"]
            if subprocess.run(cmd).returncode != 0:
                sys.exit(f"[res] could not copy {src} to {self.t.host}:{dest}")
        if self.sh(f'[ "$(stat -c %s {shlex.quote(dest)})" = {os.path.getsize(real)} ]') != 0:
            sys.exit(f"[res] {src} did not land whole at {self.t.host}:{dest}")
        return dest

    def write(self, text: str, rel: str, mode: int = 0o600) -> str:
        """Write ``text`` to ``<stage>/<rel>`` through a local file that is 0600 from creation.

        Args:
            text: File content.
            rel: Destination relative to the stage dir.
            mode: File mode on the target.

        Returns:
            The destination path on the target.
        """
        tmp = CKPT_CACHE.parent / f"stage-{uuid.uuid4().hex}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            f.write(text)
        try:
            return self.put(str(tmp), rel, mode=mode)
        finally:
            tmp.unlink()

    def link_checkpoint(self, src: str, name: str) -> str:
        """A checkpoint on the target: the cached local file itself on a local pool, else a verified copy."""
        if self.t.kind == "local":
            dest = f"{self.dir}/ckpt/{name}/{os.path.basename(src)}"
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.symlink(os.path.realpath(src), dest)
            if not os.path.isfile(dest):
                sys.exit(f"[res] checkpoint {src} is not a file")
            return dest
        return self.put(src, f"ckpt/{name}/{os.path.basename(src)}")

    def snapshot(self, repo: str, src: str) -> tuple[str, str]:
        """Bring one package repo to an ssh target as a read-only ``<scratch>/res-eval/code/<repo>-<fingerprint>``.

        Args:
            repo: Repo name.
            src: Local checkout or worktree.

        Returns:
            The snapshot path and the repo's line for the stage MANIFEST.
        """
        q = shlex.quote
        fp, desc = code_fingerprint(src)
        root = f"{self.t.scratch}/res-eval/code"
        snap = f"{root}/{repo}-{fp}"
        line = f"{repo} {src} {desc} -> {snap}"
        if self._ssh(f"[ -f {q(snap)}/.complete ]").returncode == 0:
            return snap, line
        part = f"{snap}.partial.{uuid.uuid4().hex[:6]}"
        prev = self._ssh(
            f"ls -1dt {q(root)}/{repo}-*/.complete 2>/dev/null | head -1", capture_output=True, text=True
        ).stdout.strip()
        links = [f"--link-dest={os.path.dirname(prev)}/"] if prev else []
        print(f"[res] uploading {repo} ({desc}) -> {self.t.host}:{snap}", file=sys.stderr)
        drop = f"chmod -R u+w {q(part)} 2>/dev/null; rm -rf {q(part)}"
        try:
            self._ssh(f"mkdir -p {q(part)}")
            ssh = "ssh " + " ".join(q(o) for o in SSH_OPTS)
            cmd = ["rsync", "-rlp", "--checksum", "-s", "--from0", "--files-from=-", *links, "-e", ssh, f"{src}/"]
            if subprocess.run([*cmd, f"{self.t.ssh}:{part}/"], input=b"\0".join(code_files(src))).returncode != 0:
                sys.exit(f"[res] could not upload {repo} to {self.t.host}")
            finish = f"touch {q(part)}/.complete && chmod -R a-w {q(part)} && mv -T {q(part)} {q(snap)} 2>/dev/null"
            if self._ssh(finish).returncode == 0:
                part = ""
        finally:
            if part:  # failed, interrupted, or another stage finished the same snapshot first
                self._ssh(drop)
        if self._ssh(f"[ -f {q(snap)}/.complete ]").returncode != 0:
            sys.exit(f"[res] could not finish the {repo} snapshot on {self.t.host}")
        return snap, line

    def sync_code(self, code: dict[str, str]) -> tuple[list[str], list[str]]:
        """PYTHONPATH entries that import ``code`` on the target, and the MANIFEST lines describing them.

        A local target imports the checkouts in place. An ssh target gets a fresh ``<stage>/resources/`` of links:
        package repos to their snapshots, every other repo to the target workspace's copy (so RESOURCES_DIR
        resolves); nothing is ever synced through a link.

        Args:
            code: Repo name -> local checkout path.

        Returns:
            The PYTHONPATH entries and the MANIFEST lines.
        """
        if self.t.kind == "local":
            return list(code.values()), [f"{repo} {src} {_describe(src)}" for repo, src in code.items()]
        q = shlex.quote
        res = f"{self.dir}/resources"
        snaps, lines = {}, []
        for repo, src in code.items():
            snaps[repo], line = self.snapshot(repo, src)
            lines.append(line)
        links = " ".join(f"ln -s {q(s)} {q(res)}/{repo};" for repo, s in snaps.items())
        assets = f'for d in {q(self.t.workspace)}/resources/*/; do n=$(basename "$d"); '
        assets += f'[ -e {q(res)}/"$n" ] || ln -s "${{d%/}}" {q(res)}/"$n"; done'
        if self._ssh(f"{links} {assets}").returncode != 0:
            sys.exit(f"[res] could not link the stage's resources on {self.t.host}")
        return [f"{res}/{repo}" for repo in code], lines

    def start(self, argv: list[str]) -> subprocess.Popen:
        """Start the runner in its own process group (pid in ``<stage>/pid``), stdout and stderr merged.

        Args:
            argv: The script's arguments.

        Returns:
            The local process streaming the run (the runner itself, or its ssh).
        """
        q = shlex.quote
        if self.t.kind == "local":
            self.proc = subprocess.Popen(
                ["bash", f"{self.dir}/run.sh", *argv],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            Path(f"{self.dir}/pid").write_text(str(self.proc.pid))
        else:
            args = " ".join(q(a) for a in argv)
            remote = f"cd {q(self.dir)} && {{ setsid bash run.sh {args} < /dev/null & echo $! > pid; wait $!; }}"
            self.proc = subprocess.Popen(
                ["ssh", *SSH_OPTS, self.t.ssh, remote],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        return self.proc

    def start_detached(self, argv: list[str]) -> None:
        """Start the runner in its own session, detached from this process (pid in ``<stage>/pid``).

        Args:
            argv: The script's arguments.
        """
        q = shlex.quote
        if self.t.kind == "local":
            proc = subprocess.Popen(
                ["bash", f"{self.dir}/run.sh", *argv],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            Path(f"{self.dir}/pid").write_text(str(proc.pid))
            return
        args = " ".join(q(a) for a in argv)
        cmd = f"cd {q(self.dir)} && {{ setsid nohup bash run.sh {args} < /dev/null > /dev/null 2>&1 & echo $! > pid; }}"
        if self._ssh(cmd).returncode != 0:
            sys.exit(f"[res] could not start the run on {self.t.host}")

    def kill(self) -> bool:
        """Kill the runner's process group on the target (TERM, then KILL) and report whether it is gone."""
        if self.t.kind == "local" and self.proc is not None:
            self.dead = _kill_local_group(self.proc)
            return self.dead
        q = shlex.quote
        script = (
            f"p=$(cat {q(self.dir)}/pid 2>/dev/null) || exit 0; kill -TERM -- -$p 2>/dev/null; "
            "for _ in $(seq 20); do kill -0 -- -$p 2>/dev/null || exit 0; sleep 0.5; done; "
            "kill -KILL -- -$p 2>/dev/null; sleep 1; ! kill -0 -- -$p 2>/dev/null"
        )
        self.dead = self.sh(script) == 0
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        return self.dead

    def remove(self, keep_log: bool) -> None:
        """Delete the stage dir; a failed run keeps only its log, MANIFEST and run.sh."""
        q = shlex.quote
        if keep_log:
            self.sh(f"cd {q(self.dir)} 2>/dev/null && rm -rf ckpt env tmp resources")
        else:
            self.sh(f"rm -rf {q(self.dir)}")


def _kill_local_group(proc: subprocess.Popen) -> bool:
    """TERM then KILL a local runner's process group, reaping the leader so its zombie does not count as alive."""
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, sig)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=wait)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.2)
    return False


def run(stage: Stage, argv: list[str], timeout: float, stall: float) -> int:
    """Stream the run; exit 0 after a Python traceback is a failure, and a timeout or stall kills it.

    Args:
        stage: The prepared stage.
        argv: The script's arguments.
        timeout: Seconds before the run is killed (0 = none).
        stall: Seconds without output before the run is killed (0 = none).

    Returns:
        The script's exit status, 1 for a traceback with status 0, or 124 when it was killed.
    """
    proc = stage.start(argv)
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    start = last = time.monotonic()
    saw_traceback = False
    while True:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            line = ""
        if line is None:
            break
        now = time.monotonic()
        if line:
            last = now
            sys.stdout.write(line)
            sys.stdout.flush()
            saw_traceback |= TRACEBACK in line
        why = "timed out" if timeout and now - start > timeout else ""
        why = why or ("stalled (no output)" if stall and now - last > stall else "")
        if why:
            gone = stage.kill()
            state = "gone" if gone else "STILL RUNNING, check the card"
            print(f"[res] {why}; killed the run ({state})", file=sys.stderr)
            return TIMED_OUT
    rc = proc.wait()
    stage.dead = stage.t.kind == "local" or rc != 255  # ssh exits 255 when the connection, not the run, ended
    if rc == 0 and saw_traceback:
        print("[res] the script exited 0 but printed a Python traceback; reporting failure", file=sys.stderr)
        return 1
    return rc


def _lease_card(lease_id: str, holder: str, pools: list[Pool]) -> tuple[Pool, str, int]:
    """Pool, host and gpu of a lease the caller already holds.

    Args:
        lease_id: The lease id.
        holder: The caller, who must hold it.
        pools: Configured pools.

    Returns:
        The lease's pool, host and GPU index; exits when the lease is missing, foreign or on a SLURM job.
    """
    try:
        with ls.locked_store() as leases:
            found = [x for x in leases if x.id == lease_id]
    except ls.LeaseStoreError as exc:
        sys.exit(f"[res] {exc}")
    if not found:
        sys.exit(f"[res] no lease {lease_id} (see: just res leases)")
    lease = found[0]
    if lease.holder != holder:
        sys.exit(f"[res] lease {lease_id} is held by {lease.holder}, not {holder}")
    host, _, gpu = lease.card.split(" ")[0].partition(":")
    name = lease.report.split("/")[0]
    pool = next((p for p in pools if p.name == name), None)
    if pool is None or lease.job:
        sys.exit(f"[res] lease {lease_id} is on {lease.report}, which eval cannot run on")
    return pool, host, int(gpu)


def _touch_lease(lease_id: str) -> None:
    """Mark a lease active, so a long staging does not let it lapse before the script starts."""
    with ls.locked_store() as leases:
        for lease in leases:
            if lease.id == lease_id:
                lease.last_active, lease.idle_since = time.time(), 0.0


def _check_on(spec: str, pools: list[Pool]) -> None:
    """Refuse ``--on`` a host no local or ssh pool lists, naming the backends eval does not run on."""
    host = spec.split(":")[0]
    known = {os.uname().nodename.split(".")[0]} if any(p.kind == "local" for p in pools) else set()
    known |= {h for p in pools if p.kind == "ssh" for h in p.settings.get("hosts", [])}
    if host not in known:
        other = "/".join(sorted({p.kind for p in pools if p.kind not in SUPPORTED})) or "other"
        sys.exit(f"[res] --on {spec}: not a host of a local or ssh pool; eval does not run on {other} pools yet")


def _interrupt(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(signal.Signals(signum).name)


def _repo_script(spec: str, wt: str) -> tuple[str, str]:
    """``(repo, path)`` for a ``<repo>:<path>`` script inside a shipped package repo, else ``("", "")``.

    Args:
        spec: The script argument.
        wt: The worktree set, which decides where the file is checked.

    Returns:
        The repo and the path within it; exits when the repo is not shipped or the file is missing.
    """
    if os.path.exists(spec) or ":" not in spec:
        return "", ""
    repo, _, rel = spec.partition(":")
    code = local_code(local_workspace(), wt)
    if repo not in code:
        sys.exit(f"[res] {spec}: {repo} is not a shipped repo ({', '.join(code)})")
    if not os.path.isfile(os.path.join(code[repo], rel)):
        sys.exit(f"[res] {spec}: no {rel} in {code[repo]}")
    return repo, rel


def _print_detached(stage: Stage, taken: ls.Lease | None) -> None:
    """How to follow, stop and release a detached run."""
    q = shlex.quote
    on = (lambda c: f"ssh {stage.t.ssh} {q(c)}") if stage.t.kind == "ssh" else (lambda c: c)
    print(f"[res] started detached on {stage.t.host}:gpu{stage.t.gpu} (stage {stage.dir})", file=sys.stderr)
    print(f"[res] follow: {on('tail -f ' + q(stage.dir + '/log'))}", file=sys.stderr)
    print(f"[res] stop:   {on('kill -TERM -- -$(cat ' + q(stage.dir + '/pid') + ')')}", file=sys.stderr)
    if taken is not None:
        print(
            f"[res] lease {taken.id} stays held; it releases once the card idles, or: just res release {taken.id}",
            file=sys.stderr,
        )


def cmd_eval(args: argparse.Namespace, pools: list[Pool], claim: Callable) -> None:
    """Run a script on one leased card, then kill what is left of it, clean up and release the lease it took.

    Args:
        args: The parsed `just res eval` arguments (with ``script_args``).
        pools: Configured pools.
        claim: ``res.claim``, which leases the card.
    """
    repo, rel = _repo_script(args.script, args.wt)
    script = os.path.abspath(args.script) if not repo else rel
    if not repo and not os.path.isfile(script):
        sys.exit(f"[res] no script {args.script}")
    if sum(map(bool, (args.on, args.any, args.lease))) != 1:
        sys.exit("[res] pick the card with exactly one of --on host:gpu, --any or --lease <id>")
    env = parse_env(args.env or [])
    refs = parse_checkpoints(args.checkpoint or [])
    taken = None
    if args.lease:
        pool, host, gpu = _lease_card(args.lease, args.holder, pools)
    else:
        usable = [p for p in pools if p.kind in SUPPORTED]
        named = [p for p in pools if any(p.name.startswith(s) for s in args.pool or [])]
        if args.pool and named and not any(p.kind in SUPPORTED for p in named):
            make_target(named[0], "", 0)  # exits with the per-backend hint
        if args.on:
            _check_on(args.on, pools)
        ns = argparse.Namespace(
            cards=[args.on] if args.on else [],
            any=args.any,
            count=1,
            min_free_gb=args.min_free_gb,
            pool=args.pool,
            holder=args.holder,
            note=args.note or f"res eval {os.path.basename(script)}",
            for_=0.0,
        )
        [(taken, card, rep)] = claim(ns, usable)[0]
        pool = next(p for p in pools if p.name == rep.pool.split("/")[0])
        host, gpu = card.host, card.index
        print(f"[res] leased {taken.id}: {taken.card} for {args.holder}", file=sys.stderr)
    rc, stage, detached = 1, None, False
    handlers = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        t = make_target(pool, host, gpu)
        stage = Stage(t)
        stage.make()
        paths = {ref.name: stage.link_checkpoint(fetch_checkpoint(ref), ref.name) for ref in refs}
        source = t.workspace if t.kind == "local" else local_workspace()
        code = local_code(source, args.wt)
        pythonpath, manifest = stage.sync_code(code)
        stage.write("\n".join(manifest) + "\n", "MANIFEST", mode=0o644)
        if repo:  # run the shipped repo's own file, from its root, so its sibling imports resolve
            cwd = dict(zip(code, pythonpath, strict=True))[repo]
            target_script = f"{cwd}/{rel}"
        else:
            cwd, target_script = "", stage.put(script, os.path.basename(script), mode=0o644)
        lines = [f"{k}={shlex.quote(v)}" for k, v in {**wandb_env(), **env, **paths}.items()]
        stage.write("\n".join(lines) + "\n", "env")
        runner = runner_script(t, stage.dir, target_script, sorted({**env, **paths}), pythonpath, cwd)
        stage.write(runner, "run.sh", mode=0o700)
        if taken is not None:
            _touch_lease(taken.id)
        if args.detach:
            stage.start_detached(args.script_args)
            detached = True
            _print_detached(stage, taken)
            rc = 0
        else:
            print(
                f"[res] running {os.path.basename(script)} on {t.host}:gpu{t.gpu} (stage {stage.dir})", file=sys.stderr
            )
            rc = run(stage, args.script_args, args.timeout, args.stall)
    except KeyboardInterrupt as exc:
        rc = 130
        print(f"[res] interrupted ({exc or 'SIGINT'})", file=sys.stderr)
    finally:
        for sig, old in handlers.items():
            signal.signal(sig, old)
        if detached:  # the run owns its stage and lease now
            sys.exit(rc)
        gone = True
        if stage is not None and stage.proc is not None and not stage.dead:
            gone = stage.kill()
            print(f"[res] stopped the run ({'gone' if gone else 'STILL RUNNING, check the card'})", file=sys.stderr)
        if stage is not None:
            stage.remove(keep_log=rc != 0)
            if rc != 0:
                print(f"[res] FAILED with status {rc}; log at {stage.t.host}:{stage.dir}/log", file=sys.stderr)
        if taken is not None and gone:
            with ls.locked_store() as leases:
                leases[:] = [x for x in leases if x.id != taken.id]
            print(f"[res] released {taken.id}", file=sys.stderr)
        elif taken is not None:
            print(
                f"[res] KEPT lease {taken.id}: the run may still hold the card; release it once it is gone",
                file=sys.stderr,
            )
    sys.exit(rc)


def _duration(text: str) -> float:
    """A duration like ``90s``, ``30m``, ``2h`` in seconds; ``0`` turns the limit off."""
    if text.rstrip("smhd") in ("0", "0.0"):
        return 0.0
    try:
        return ls.parse_duration(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register `eval` on the `just res` subparsers."""
    ev = sub.add_parser("eval", help="run a one-off script on a leased GPU (local or ssh pools)")
    ev.add_argument(
        "script", help="script on this machine, or <repo>:<path> inside a shipped repo; its arguments follow --"
    )
    ev.add_argument("--detach", action="store_true", help="start the run and return; the lease stays held")
    ev.add_argument("--on", help="card as host:gpu")
    ev.add_argument("--any", action="store_true", help="take any free card")
    ev.add_argument("--lease", help="run on a card you already lease (left leased afterwards)")
    ev.add_argument("--pool", action="append", help="with --any/--on: only these pools (prefix match)")
    ev.add_argument("--min-free-gb", type=float, default=0, help="with --any: free memory the card needs")
    ev.add_argument("--holder", required=True, help="your session name")
    ev.add_argument("--note", default="", help="lease note")
    ev.add_argument(
        "--checkpoint",
        action="append",
        help="[NAME=]<W&B run URL | entity/project/run[@iter] | path>, exported as NAME",
    )
    ev.add_argument("--env", action="append", help="KEY=VALUE for the script (repeatable)")
    ev.add_argument("--wt", default="", help="run this machine's worktree set (resources/<repo>/worktrees/<name>)")
    ev.add_argument(
        "--timeout", type=_duration, default=0.0, help="kill the run after this long, e.g. 2h (default: none)"
    )
    ev.add_argument(
        "--stall", type=_duration, default=900.0, help="kill the run after this long with no output (15m; 0 = off)"
    )


def split_script_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split an `eval` command line at its first ``--``: res's own arguments, then the script's (verbatim)."""
    if argv[:1] == ["eval"] and "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []
