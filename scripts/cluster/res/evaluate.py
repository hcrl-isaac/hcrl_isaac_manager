"""`pls run --on <card>`: run a one-off script on a leased GPU, with its checkpoints and code.

Local and ssh pools run the workspace's ilab python on the box. A SLURM card is one of a held dev sentinel's: the
script runs in the container through a `develop exec` step on that job, against the cluster's shared checkout with
the ``--wt`` repos staged over it as a code tree.
"""

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
from inventory import MANAGER_DIR, RES_DIR, Pool, profile_value
from probe import MASTER_OPTS, SSH_OPTS

CLUSTER_DEV = MANAGER_DIR / "scripts" / "cluster" / "cluster_dev" / "cluster_dev.sh"
SUPPORTED = ("local", "ssh", "slurm")
# where node_exec binds the cluster checkout's artifacts/ inside the container, and the container's interpreter
CONTAINER_ARTIFACTS = "/workspace/artifacts"
CONTAINER_PYTHON = "/isaac-sim/python.sh"
CONTAINER_TMP = "/tmp/res-eval"  # in the container's /tmp, the job's node-local dir
CKPT_CACHE = Path.home() / ".cache" / "hcrl_res" / "checkpoints"
TRACEBACK = "Traceback (most recent call last)"
TIMED_OUT = 124
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class Target:
    """Where the script runs: one card of a local or ssh pool, or of a SLURM job's node.

    Args:
        pool: Pool name (for slurm, the cluster profile).
        kind: ``local``, ``ssh`` or ``slurm``.
        host: Host name as the probe reports it.
        gpu: Physical GPU index.
        ssh: ``user@fqdn`` for ssh pools, the login node for slurm, "" for local.
        workspace: Manager checkout on the target (holds ilab/ and the asset repos; the cluster checkout for slurm).
        scratch: Per-box scratch dir that stage dirs and code snapshots go under.
        pin: ``cvd`` masks CUDA_VISIBLE_DEVICES to the card; ``device`` leaves it unmasked and the script
            selects the card from RES_EVAL_DEVICE.
        job: The SLURM job (a held dev sentinel) whose node the card is on.
        uuid: The card's GPU UUID, which pins it inside a SLURM step whose device numbering may differ.
    """

    pool: str
    kind: str
    host: str
    gpu: int
    ssh: str = ""
    workspace: str = ""
    scratch: str = ""
    pin: str = "cvd"
    job: str = ""
    uuid: str = ""

    @property
    def ssh_opts(self) -> list[str]:
        """ssh options: a SLURM login only over its existing master (a new connection would need 2FA)."""
        return [*MASTER_OPTS, *SSH_OPTS] if self.kind == "slurm" else list(SSH_OPTS)


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


def make_target(pool: Pool, host: str, gpu: int, job: str = "", uuid: str = "") -> Target:
    """Target for one card; refuses pools whose backend `eval` cannot run on yet.

    Args:
        pool: The card's pool (for a SLURM card, the cluster profile).
        host: Host name from the probe.
        gpu: Physical GPU index.
        job: The SLURM job holding the card.
        uuid: The card's GPU UUID (SLURM cards).

    Returns:
        The target, with workspace, scratch and pin from the pool settings or their defaults.
    """
    if pool.kind not in SUPPORTED:
        hint = {"ray": "use `just ray run <repo>/<script>.py` (Ray queues it until a GPU frees)"}
        sys.exit(f"[res] eval does not run on {pool.kind} pools yet; {hint.get(pool.kind, 'pick another card')}")
    s = pool.settings
    if pool.kind == "slurm":
        if not job or not uuid:
            sys.exit(f"[res] {host}:{gpu} on {pool.name}: eval runs only on a card of a running job")
        remote = profile_value(pool.name, "CLUSTER_ISAACLAB_DIR")
        if not remote:
            sys.exit(f"[res] cluster profile {pool.name} sets no CLUSTER_ISAACLAB_DIR")
        login = s.get("login") or profile_value(pool.name, "CLUSTER_LOGIN")
        return Target(pool.name, "slurm", host, gpu, login, remote, f"{remote}/artifacts", "cvd", job, uuid)
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
    cmd = [py, str(RES_DIR / "checkpoints.py"), os.path.join(ws, "resources", "hcrl_isaaclab"), str(CKPT_CACHE)]
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
    sys.path.insert(0, str(MANAGER_DIR / "scripts"))
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
mkdir -p {q(cwd or t.workspace)} && cd {q(cwd or t.workspace)} || exit 97
"$py" {q(script)} "$@"
"""


def cluster_dev_cmd(profile: str, job: str, *args: str) -> tuple[list[str], dict[str, str]]:
    """``cluster_dev.sh`` argv and environment for one profile, aimed at ``job`` (a held sentinel) when given.

    Args:
        profile: Cluster profile name.
        job: SLURM job id, or "" for commands that need no job.
        *args: The cluster_dev.sh subcommand and its arguments.

    Returns:
        The argv and the environment to run it with.
    """
    env = {**os.environ, "CLUSTER": profile, "LOCAL_ISAACLAB_DIR": local_workspace()}
    if job:
        env["DEV_JOBID"] = job
    return ["bash", str(CLUSTER_DEV), *args], env


TREE_KEEP_DAYS = 14  # an unused res-eval tree older than this is removed when a newer one is staged


def stage_tree(t: Target, code: dict[str, str]) -> str:
    """Stage ``code`` on the cluster as a code tree (deduplicated against earlier trees) and return its id.

    Earlier ``res-eval`` trees older than ``TREE_KEEP_DAYS`` that no running step marks in use are removed.

    Args:
        t: The SLURM target.
        code: Repo name -> local checkout or worktree path.

    Returns:
        The tree id (``res-eval-<fingerprint>``) for ``develop exec --tree``.
    """
    specs = [f"{repo}={src}" for repo, src in code.items() if repo != "IsaacLab" and not repo.endswith("_robots")]
    cmd, env = cluster_dev_cmd(t.pool, "", "stage", "res-eval", *specs)
    print(f"[res] staging {', '.join(code)} on {t.pool} as a code tree", file=sys.stderr)
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, text=True)
    found = re.search(r"--tree (\S+) --", res.stdout)
    if res.returncode != 0 or not found:
        sys.exit(f"[res] could not stage the code on {t.pool} (develop stage exited {res.returncode})")
    tree = found.group(1)
    q = shlex.quote
    trees = q(profile_value(t.pool, "CLUSTER_TREES_DIR") or f"{t.workspace}/trees")
    # the tree's mtime is its last use: a reused (deduplicated) tree is touched, so a concurrent prune keeps it
    prune = (
        f"touch {trees}/{q(tree)}; "
        f'for d in {trees}/res-eval-*; do [ "$(basename "$d")" = {q(tree)} ] && continue; '
        f'[ -n "$(find "$d" -maxdepth 0 -mtime +{TREE_KEEP_DAYS})" ] || continue; '
        '[ -z "$(ls -A "$d/.in-use" 2>/dev/null)" ] || continue; chmod -R u+w "$d" && rm -rf "$d"; done'
    )
    subprocess.run(["ssh", *t.ssh_opts, t.ssh, prune], capture_output=True)
    return tree


def container_path(t: Target, path: str) -> str:
    """A path under the cluster checkout's artifacts/ as the container sees it (unchanged off SLURM)."""
    if t.kind == "slurm" and path.startswith(t.scratch + "/"):
        return CONTAINER_ARTIFACTS + path[len(t.scratch) :]
    return path


HEARTBEAT_S = 2  # a SLURM runner touches <stage>/heartbeat this often while its script runs
DRAIN_S = 60  # how long a SLURM runner waits for the card's processes to go after its script ends
START_S = 1200  # how long a detached SLURM run may take to reach its runner (a first .sif copy to the node, boot)
STOP_WAIT_S = 90  # how long a stop waits for the runner's status: its TERM/KILL (~10 s) plus the drain


def slurm_runner_script(t: Target, stage: str, script: str, env_names: list[str], cwd: str, python: str = "") -> str:
    """The bash that runs inside the container on a SLURM node: pin by UUID, run, stop on request, record status.

    Args:
        t: The target.
        stage: The stage dir as the container sees it.
        script: The script's path in the container.
        env_names: Names of the exported variables, for the log.
        cwd: Directory to run the script from, in the container.
        python: The interpreter (default: ``CONTAINER_PYTHON``).

    Returns:
        The run.sh text. The container hides its processes from the node, so ``<stage>/heartbeat`` shows the run
        alive, ``<stage>/stop`` asks it to end, and ``<stage>/status`` holds its exit status once the card drained.
    """
    q = shlex.quote
    python = python or CONTAINER_PYTHON
    s = {k: q(f"{stage}/{k}") for k in ("log", "env", "status", "stop", "heartbeat", "MANIFEST")}
    # node-local: the container's /tmp is the job's per-node dir, shared by every step of the sentinel, so the run
    # takes its own TMPDIR and per-card Kit caches inside it (one lease per card)
    run_id, local = os.path.basename(stage), CONTAINER_TMP
    tmp, cache = q(f"{local}/{run_id}"), q(f"{local}/cache/{t.host}-gpu{t.gpu}")
    apps = f"nvidia-smi -i {q(t.uuid)} --query-compute-apps=pid --format=csv,noheader 2>/dev/null"
    return f"""#!/usr/bin/env bash
set -u
exec > >(tee -a {s["log"]}) 2>&1
if [ -e {s["stop"]} ]; then  # stopped before the step started (a slow start, or Ctrl-C during boot)
    rm -f {s["env"]}; echo "[res] stopped before it started"; echo 130 > {s["status"]}; exit 130
fi
set -a; source {s["env"]}; set +a
rm -f {s["env"]}  # credentials stay in this process's environment only
touch {s["heartbeat"]}
if ! nvidia-smi -i {q(t.uuid)} > /dev/null 2>&1; then
    echo "[res] card {t.uuid} ({t.host}:gpu{t.gpu}) is not visible in this step; another step of job {t.job} may hold it"
    echo 98 > {s["status"]}; exit 98
fi
export CUDA_VISIBLE_DEVICES={q(t.uuid)} RES_EVAL_DEVICE=cuda:0 PYTHONUNBUFFERED=1
export TMPDIR={tmp} XDG_CACHE_HOME={cache}/xdg OMNI_CACHE_DIR={cache}/omni
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$OMNI_CACHE_DIR"
trap 'rm -rf "$TMPDIR"' EXIT
cat {s["MANIFEST"]}
echo "[res] {t.host}:gpu{t.gpu} job {t.job} (uuid) env: {" ".join(env_names) or "-"}"
mkdir -p {q(cwd)} && cd {q(cwd)} || {{ echo 97 > {s["status"]}; exit 97; }}
script_from=$(( $(stat -c %s {s["log"]}) + 1 ))  # the traceback check reads only the script's own output
setsid {q(python)} {q(script)} "$@" &
child=$!
stop_child() {{
    kill -TERM -- -"$child" 2>/dev/null
    for _ in $(seq 20); do kill -0 "$child" 2>/dev/null || return 0; sleep 0.5; done
    kill -KILL -- -"$child" 2>/dev/null
}}
trap stop_child TERM INT HUP
while kill -0 "$child" 2>/dev/null; do
    touch {s["heartbeat"]}
    if [ -e {s["stop"]} ]; then echo "[res] stop requested"; stop_child; break; fi
    sleep {HEARTBEAT_S}
done
wait "$child"; rc=$?
if [ "$rc" = 0 ] && tail -c +"$script_from" {s["log"]} | grep -q "{TRACEBACK}"; then
    echo "[res] the script exited 0 but printed a Python traceback; reporting failure"; rc=1
fi
for _ in $(seq {DRAIN_S}); do [ -z "$({apps})" ] && break; touch {s["heartbeat"]}; sleep 1; done
[ -z "$({apps})" ] || echo "[res] the card still has processes {DRAIN_S} s after the run ended"
echo "$rc" > {s["status"]}
exit "$rc"
"""


STAGE_KEEP_DAYS = 7  # a finished or dead SLURM stage (log, status) is kept this long


def _prune_stages(root: str) -> str:
    """Shell pruning old SLURM stages: credentials of never-started runs, and dead stages and work dirs.

    Credentials go once a stage is 30 min old without a heartbeat; a stage or work dir goes once it is older than
    ``STAGE_KEEP_DAYS`` and its run has had no heartbeat in the last hour.
    """
    q = shlex.quote
    return (
        f'for d in {q(root)}/*/ {q(root)}/work/*/; do [ -d "$d" ] || continue; d="${{d%/}}"; '
        'case "$(basename "$d")" in cache|work) continue;; '
        'esac; [ -n "$(find "$d" -maxdepth 0 -mmin +30)" ] || continue; [ -e "$d/heartbeat" ] || rm -f "$d/env"; '
        '[ -n "$(find "$d" -maxdepth 1 -name heartbeat -mmin -60)" ] && continue; '
        f'[ -n "$(find "$d" -maxdepth 0 -mtime +{STAGE_KEEP_DAYS})" ] && rm -rf "$d"; done; true'
    )


class Stage:
    """A fresh per-invocation directory on the target, plus the immutable code snapshots it links to."""

    def __init__(self, t: Target) -> None:
        self.t = t
        self.dir = f"{t.scratch}/res-eval/{uuid.uuid4().hex[:8]}"
        self.proc: subprocess.Popen | None = None
        self.dead = False
        self.tree = ""  # the staged code tree a SLURM run uses

    def _ssh(self, cmd: str, **kw: object) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh", *self.t.ssh_opts, self.t.ssh, cmd], **kw)

    def _rsync_ssh(self) -> str:
        return "ssh " + " ".join(shlex.quote(o) for o in self.t.ssh_opts)

    def sh(self, cmd: str) -> int:
        """Run a shell command on the target (here, for a local pool) and return its status."""
        if self.t.kind == "local":
            return subprocess.run(["bash", "-c", cmd]).returncode
        return self._ssh(cmd).returncode

    def make(self) -> None:
        """Create the stage dir (0700); on SLURM, first prune earlier detached stages."""
        q = shlex.quote
        if self.t.kind == "slurm":
            self.sh(_prune_stages(f"{self.t.scratch}/res-eval"))
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
            cmd = ["rsync", "-L", "-s", f"--chmod=F{mode:o}", "-e", self._rsync_ssh(), real, f"{self.t.ssh}:{dest}"]
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
            ssh = self._rsync_ssh()
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
        if self.t.kind == "slurm":
            # only worktree-set repos are staged; the rest run from the cluster's shared checkout (`develop sync`)
            main = os.path.join(local_workspace(), "resources")
            staged = {
                repo: src
                for repo, src in code.items()
                if os.path.realpath(src) != os.path.realpath(os.path.join(main, repo))
            }
            self.tree = stage_tree(self.t, staged) if staged else ""
            shared = f"{self.t.workspace}/resources"
            q = shlex.quote
            heads = self._ssh(
                " ".join(
                    f"echo {q(r)} $(git -C {q(f'{shared}/{r}')} rev-parse --short HEAD 2>/dev/null);" for r in code
                ),
                capture_output=True,
                text=True,
            ).stdout
            commit = dict(line.split()[:2] for line in heads.splitlines() if len(line.split()) >= 2)
            lines = [
                f"{repo} {src} {_describe(src)} -> tree {self.tree}"
                if repo in staged
                else f"{repo} {shared}/{repo} {commit.get(repo, '(no git metadata: as develop sync left it)')} (shared checkout)"
                for repo, src in code.items()
            ]
            return [f"/workspace/ext/{repo}" for repo in code], lines
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
        if self.t.kind == "slurm":
            cmd, env = self._exec_cmd(argv)
            self.proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            return self.proc
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
        if self.t.kind == "slurm":
            cmd, env = self._exec_cmd(argv, "--detach", "--log", f"{self.dir}/wrapper.log")
            started = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL).returncode == 0
            beat = f"for _ in $(seq {START_S}); do [ -e {q(self.dir)}/heartbeat ] && exit 0; sleep 1; done; exit 1"
            if not started or self.sh(beat) != 0:
                self.sh(f"touch {q(self.dir)}/stop")  # a step that starts after all ends at once
                sys.exit(
                    f"[res] the run did not start on job {self.t.job} within {START_S} s; see {self.dir}/wrapper.log"
                )
            return
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

    def _exec_cmd(self, argv: list[str], *opts: str) -> tuple[list[str], dict[str, str]]:
        """`develop exec` of this stage's run.sh in the staged tree, on the target's job."""
        run_sh = container_path(self.t, f"{self.dir}/run.sh")
        tree = ["--tree", self.tree] if self.tree else []
        return cluster_dev_cmd(self.t.pool, self.t.job, "exec", *tree, *opts, "--", "bash", run_sh, *argv)

    def kill(self) -> bool:
        """Kill the runner's process group on the target (TERM, then KILL) and report whether it is gone."""
        if self.t.kind == "slurm":
            # the container hides the run from the node: ask its runner to stop and wait for its status, which it
            # writes once the card drained; a runner that never started has nothing to stop
            q = shlex.quote
            d = q(self.dir)
            wait = f"[ -e {d}/heartbeat ] || exit 0; for _ in $(seq {STOP_WAIT_S}); do [ -f {d}/status ] && exit 0; sleep 1; done; exit 1"
            self.dead = self.sh(f"touch {d}/stop && {{ {wait}; }}") == 0
            if self.proc is not None and self.proc.poll() is None:
                _kill_local_group(self.proc)
            return self.dead
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


def slurm_profile(report_pool: str, job: str, pools: list[Pool]) -> Pool:
    """The cluster profile to run a SLURM card through: the one named after its job's partition, else the first.

    Args:
        report_pool: The card's report label, ``<site> (<profile>, ...)``.
        job: The job holding the card.
        pools: Configured pools.

    Returns:
        The profile's pool; its #SBATCH options are the ones `develop exec` steps onto the job with.
    """
    names = [n.strip() for n in report_pool.partition("(")[2].rstrip(")").split(",")]
    profiles = [p for n in names for p in pools if p.kind == "slurm" and p.name == n]
    if not profiles:
        sys.exit(f"[res] no cluster profile for {report_pool}")
    login = profiles[0].settings.get("login", "")
    res = subprocess.run(
        ["ssh", *MASTER_OPTS, *SSH_OPTS, login, f"squeue -h -j {shlex.quote(job)} -o %P"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 or not res.stdout.strip():
        sys.exit(f"[res] cannot read job {job}'s partition on {login}: {(res.stderr.strip() or 'no such job')[:160]}")
    partition = res.stdout.strip().splitlines()[-1].strip()
    return next((p for p in profiles if p.name == partition), profiles[0])


def _lease_card(lease_id: str, holder: str, pools: list[Pool]) -> tuple[Pool, str, int, str, str]:
    """Pool, host, gpu, job and card UUID of a lease the caller already holds.

    Args:
        lease_id: The lease id.
        holder: The caller, who must hold it.
        pools: Configured pools.

    Returns:
        The lease's pool (a cluster profile for a SLURM card), host, GPU index, job and UUID; exits when the lease
        is missing or foreign.
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
    if lease.job:
        return slurm_profile(lease.report.split(" job ")[0], lease.job, pools), host, int(gpu), lease.job, lease.key
    name = lease.report.split("/")[0]
    pool = next((p for p in pools if p.name == name), None)
    if pool is None:
        sys.exit(f"[res] lease {lease_id} is on {lease.report}, which eval cannot run on")
    return pool, host, int(gpu), "", ""


def _touch_lease(lease_id: str) -> None:
    """Mark a lease active, so a long staging does not let it lapse before the script starts."""
    with ls.locked_store() as leases:
        for lease in leases:
            if lease.id == lease_id:
                lease.last_active, lease.idle_since = time.time(), 0.0


def resolve_on(spec: str, pools: list[Pool]) -> str:
    """``--on`` with ``local`` as this machine's host name, refused when no pool eval runs on can hold it.

    Args:
        spec: ``host:gpu`` or ``host:job:gpu``; ``local:<gpu>`` names this machine's card.
        pools: Configured pools.

    Returns:
        The card spec for the claim. A SLURM node is left for the claim's probe to find.
    """
    host, sep, rest = spec.partition(":")
    has_local = any(p.kind == "local" for p in pools)
    if host == "local" and has_local:
        host = os.uname().nodename.split(".")[0]
    known = {os.uname().nodename.split(".")[0]} if has_local else set()
    known |= {h for p in pools if p.kind == "ssh" for h in p.settings.get("hosts", [])}
    if host not in known and not any(p.kind == "slurm" for p in pools):
        other = "/".join(sorted({p.kind for p in pools if p.kind not in SUPPORTED})) or "other"
        sys.exit(f"[res] --on {spec}: not a host of a local, ssh or slurm pool; eval does not run on {other} pools")
    return f"{host}{sep}{rest}"


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
    opts = " ".join(q(o) for o in stage.t.ssh_opts) if stage.t.kind == "slurm" else ""
    on = (
        (lambda c: f"ssh {opts + ' ' if opts else ''}{stage.t.ssh} {q(c)}")
        if stage.t.kind != "local"
        else (lambda c: c)
    )
    where = f"{stage.t.host}:gpu{stage.t.gpu}" + (f" of job {stage.t.job}" if stage.t.job else "")
    print(f"[res] started detached on {where} (stage {stage.dir})", file=sys.stderr)
    print(f"[res] follow: {on('tail -f ' + q(stage.dir + '/log'))}", file=sys.stderr)
    if stage.t.kind == "slurm":
        print(f"[res] stop:   {on('touch ' + q(stage.dir + '/stop'))}", file=sys.stderr)
        print(
            f"[res] alive while {stage.dir}/heartbeat is fresh (every {HEARTBEAT_S} s); the exit status lands in status",
            file=sys.stderr,
        )
    else:
        print(f"[res] stop:   {on('kill -TERM -- -$(cat ' + q(stage.dir + '/pid') + ')')}", file=sys.stderr)
    if taken is not None:
        print(
            f"[res] lease {taken.id} stays held; it releases once the card idles, or: just res release {taken.id}",
            file=sys.stderr,
        )


def cmd_eval(args: argparse.Namespace, pools: list[Pool], claim: Callable) -> None:
    """Run a script on one leased card, then kill what is left of it, clean up and release the lease it took.

    Args:
        args: The parsed `pls run --on <card>` arguments (with ``script_args``).
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
    taken, job, uuid = None, "", ""
    if args.lease:
        pool, host, gpu, job, uuid = _lease_card(args.lease, args.holder, pools)
    else:
        named = [p for p in pools if any(p.name.startswith(s) for s in args.pool or [])]
        if args.pool and named and not any(p.kind in SUPPORTED for p in named):
            make_target(named[0], "", 0)  # exits with the per-backend hint
        on = resolve_on(args.on, pools) if args.on else ""
        # a cluster job's cards only when named: --any stays on the boxes unless --pool picks a cluster
        usable = [p for p in pools if p.kind in SUPPORTED and (p.kind != "slurm" or on or p in named)]
        ns = argparse.Namespace(
            cards=[on] if on else [],
            any=args.any,
            count=1,
            min_free_gb=args.min_free_gb,
            pool=args.pool,
            holder=args.holder,
            note=args.note or f"res eval {os.path.basename(script)}",
            for_=0.0,
        )
        [(taken, card, rep)] = claim(ns, usable)[0]
        host, gpu = card.host, card.index
        print(f"[res] leased {taken.id}: {taken.card} for {args.holder}", file=sys.stderr)
        if rep.kind == "slurm":
            pool, job, uuid = None, card.job, card.uuid
        else:
            pool = next(p for p in pools if p.name == rep.pool.split("/")[0])
    rc, stage, detached = 1, None, False
    handlers = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        if pool is None:  # inside the try, so a refusal still releases the lease
            pool = slurm_profile(rep.pool, job, pools)
        t = make_target(pool, host, gpu, job, uuid)
        stage = Stage(t)
        stage.make()
        paths = {ref.name: container_path(t, stage.link_checkpoint(fetch_checkpoint(ref), ref.name)) for ref in refs}
        source = t.workspace if t.kind == "local" else local_workspace()
        code = local_code(source, args.wt)
        pythonpath, manifest = stage.sync_code(code)
        stage.write("\n".join(manifest) + "\n", "MANIFEST", mode=0o644)
        # snapshots are read-only, runs may be concurrent and a stage is removed after a success, so relative outputs
        # (train.py's logs/, a census's tables) go to a writable working dir of this run's own, kept afterwards
        work = container_path(t, f"{t.scratch}/res-eval/work/{os.path.basename(stage.dir)}")
        if repo:  # run the shipped repo's own file, from its root, so its sibling imports resolve
            root = dict(zip(code, pythonpath, strict=True))[repo]
            target_script, cwd = f"{root}/{rel}", work
        else:
            put = stage.put(script, os.path.basename(script), mode=0o644)
            cwd, target_script = work if t.kind == "slurm" else "", container_path(t, put)
        lines = [f"{k}={shlex.quote(v)}" for k, v in {**wandb_env(), **env, **paths}.items()]
        stage.write("\n".join(lines) + "\n", "env")
        names = sorted({**env, **paths})
        if t.kind == "slurm":
            runner = slurm_runner_script(t, container_path(t, stage.dir), target_script, names, cwd)
        else:
            runner = runner_script(t, stage.dir, target_script, names, pythonpath, cwd)
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
                where = stage.t.ssh if stage.t.kind == "slurm" else stage.t.host
                print(f"[res] FAILED with status {rc}; log at {where}:{stage.dir}/log", file=sys.stderr)
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


def parser() -> argparse.ArgumentParser:
    """The card backend's arguments; `pls run` builds them from its own ``--on``/``--wt`` and passes the rest."""
    ev = argparse.ArgumentParser(
        prog="pls run --on <card>",
        description="run a one-off script on a leased GPU (local, ssh, or a held SLURM sentinel)",
    )
    ev.add_argument(
        "script", help="script on this machine, or <repo>:<path> inside a shipped repo; its arguments follow --"
    )
    ev.add_argument("--detach", action="store_true", help="start the run and return; the lease stays held")
    # the card selectors and --wt come from `pls run --on/--wt` (see `pls run --help`), so they are not listed here
    ev.add_argument("--on", help=argparse.SUPPRESS)  # host:gpu, host:job:gpu, or local:<gpu>
    ev.add_argument("--any", action="store_true", help=argparse.SUPPRESS)
    ev.add_argument("--lease", help=argparse.SUPPRESS)  # a card already leased, left leased afterwards
    ev.add_argument("--pool", action="append", help="only these pools (repeatable; prefix match)")
    ev.add_argument("--min-free-gb", type=float, default=0, help="with --any: free memory the card needs")
    ev.add_argument("--holder", required=True, help="your session name")
    ev.add_argument("--note", default="", help="lease note")
    ev.add_argument(
        "--checkpoint",
        action="append",
        help="[NAME=]<W&B run URL | entity/project/run[@iter] | path>, exported as NAME",
    )
    ev.add_argument("--env", action="append", help="KEY=VALUE for the script (repeatable)")
    ev.add_argument("--wt", default="", help=argparse.SUPPRESS)  # this machine's resources/<repo>/worktrees/<name>
    ev.add_argument(
        "--timeout", type=_duration, default=0.0, help="kill the run after this long, e.g. 2h (default: none)"
    )
    ev.add_argument(
        "--stall", type=_duration, default=900.0, help="kill the run after this long with no output (15m; 0 = off)"
    )
    return ev


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse at the first ``--``: the backend's own arguments, then the script's (verbatim) as ``script_args``."""
    i = argv.index("--") if "--" in argv else len(argv)
    args = parser().parse_args(argv[:i])
    args.script_args = argv[i + 1 :]
    return args


def main(argv: list[str] | None = None) -> None:
    """Entry point of `pls run --on <card>`: claim the card through res, run the script, release."""
    import res

    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        pools = res.load_pools()
    except Exception as exc:  # a broken inventory must say so, not print a traceback
        sys.exit(f"[res] cannot read the compute inventory: {type(exc).__name__}: {exc}")
    cmd_eval(args, pools, res.claim)


if __name__ == "__main__":
    main()
