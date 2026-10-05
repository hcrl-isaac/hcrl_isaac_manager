"""`just res eval`: run a one-off script on a leased GPU of a local or ssh pool, with its checkpoints brought along."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
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
        workspace: Manager checkout on the target (holds ilab/ and resources/).
        scratch: Per-box scratch dir that stage dirs go under.
    """

    pool: str
    kind: str
    host: str
    gpu: int
    ssh: str = ""
    workspace: str = ""
    scratch: str = ""


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
        The target, with workspace and scratch from the pool settings or their defaults.
    """
    if pool.kind not in SUPPORTED:
        hint = {"ray": "use `just ray job`", "slurm": "use a dev-node step (`just cluster <c> develop exec`)"}
        sys.exit(f"[res] eval does not run on {pool.kind} pools yet; {hint.get(pool.kind, 'pick a local/ssh card')}")
    s = pool.settings
    if pool.kind == "local":
        ws = s.get("workspace") or local_workspace()
        return Target(pool.name, "local", host, gpu, workspace=ws, scratch=s.get("scratch", f"{Path.home()}/tmp"))
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


def runner_script(t: Target, stage: str, script: str, env_names: list[str], pythonpath: list[str]) -> str:
    """The bash that runs on the target: isolated caches, the given PYTHONPATH, then the script's own status."""
    cache = f"{t.scratch}/res-eval/cache/{t.host}-gpu{t.gpu}"  # one lease per card, so per-card caches never race
    return f"""#!/usr/bin/env bash
set -u
cd {t.workspace} || {{ echo "[res] no workspace {t.workspace} on {t.host}"; exit 97; }}
set -a; source {stage}/env; set +a
export CUDA_VISIBLE_DEVICES={t.gpu} TMPDIR={stage}/tmp XDG_CACHE_HOME={cache}/xdg OMNI_CACHE_DIR={cache}/omni
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$OMNI_CACHE_DIR"
gomp="$(ls ilab/lib/python3.11/site-packages/torch/lib/libgomp-*.so.1 2>/dev/null | head -1)"
[ -n "$gomp" ] && export LD_PRELOAD="$gomp${{LD_PRELOAD:+:$LD_PRELOAD}}"
export PYTHONPATH="{":".join(pythonpath)}${{PYTHONPATH:+:$PYTHONPATH}}"
echo "[res] {t.host}:gpu{t.gpu} python=$PWD/ilab/bin/python env: {" ".join(env_names) or "-"}"
./ilab/bin/python {stage}/{os.path.basename(script)} "$@"
"""


class Stage:
    """A per-invocation directory on the target; created fresh, never synced with --delete."""

    def __init__(self, t: Target) -> None:
        self.t = t
        self.dir = f"{t.scratch}/res-eval/{uuid.uuid4().hex[:8]}"

    def _ssh(self, cmd: str, **kw: object) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh", *SSH_OPTS, self.t.ssh, cmd], **kw)

    def make(self) -> None:
        """Create the stage dir (0700)."""
        if self.t.kind == "local":
            os.makedirs(f"{self.dir}/ckpt", mode=0o700)
        elif self._ssh(f"mkdir -p -m 700 {shlex.quote(self.dir)}/ckpt").returncode != 0:
            sys.exit(f"[res] cannot create {self.dir} on {self.t.host}")

    def put(self, src: str, rel: str, link: bool = False, mode: int = 0o600) -> str:
        """Place a local file at ``<stage>/<rel>``; local checkpoints are symlinked instead of copied."""
        dest = f"{self.dir}/{rel}"
        if self.t.kind == "local":
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.symlink(src, dest) if link else shutil.copyfile(src, dest)
            if not link:
                os.chmod(dest, mode)
            return dest
        self._ssh(f"mkdir -p {shlex.quote(os.path.dirname(dest))}")
        ssh = "ssh " + " ".join(shlex.quote(o) for o in SSH_OPTS)
        cmd = ["rsync", "-s", f"--chmod=F{mode:o}", "-e", ssh, src, f"{self.t.ssh}:{dest}"]
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"[res] could not copy {src} to {self.t.host}:{dest}")
        return dest

    def write(self, text: str, rel: str, mode: int = 0o600) -> str:
        """Write ``text`` to ``<stage>/<rel>``."""
        tmp = Path(CKPT_CACHE.parent / f"stage-{uuid.uuid4().hex}")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text)
        try:
            return self.put(str(tmp), rel, mode=mode)
        finally:
            tmp.unlink()

    def run(self, argv: list[str]) -> subprocess.Popen:
        """Start the runner with the script's arguments, stdout and stderr merged."""
        quoted = " ".join(shlex.quote(a) for a in argv)
        cmd = (
            ["bash", f"{self.dir}/run.sh", *argv]
            if self.t.kind == "local"
            else ["ssh", *SSH_OPTS, self.t.ssh, f"bash {shlex.quote(self.dir)}/run.sh {quoted}"]
        )
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def sync_code(self, code: dict[str, str]) -> list[str]:
        """Bring the package repos to the target; return the PYTHONPATH entries that import them there.

        A local target imports them in place. An ssh target gets their tracked and non-ignored files in a per-card
        code dir (one lease per card, so no other run writes it), with the target's own asset repos linked beside
        them so RESOURCES_DIR resolves; nothing is ever deleted there.
        """
        if self.t.kind == "local":
            return list(code.values())
        root = f"{self.t.scratch}/res-eval/code/{self.t.host}-gpu{self.t.gpu}"
        ssh = "ssh " + " ".join(shlex.quote(o) for o in SSH_OPTS)
        for repo, src in code.items():
            files = subprocess.run(
                ["git", "-C", src, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                capture_output=True,
                check=True,
            ).stdout
            present = b"\0".join(f for f in files.split(b"\0") if f and os.path.lexists(os.path.join(src, f.decode())))
            print(f"[res] syncing {repo} ({src}) -> {self.t.host}:{root}/{repo}", file=sys.stderr)
            self._ssh(f"mkdir -p {shlex.quote(root)}/{repo}")
            cmd = [
                "rsync",
                "-rlpt",
                "-s",
                "--from0",
                "--files-from=-",
                "-e",
                ssh,
                f"{src}/",
                f"{self.t.ssh}:{root}/{repo}/",
            ]
            if subprocess.run(cmd, input=present).returncode != 0:
                sys.exit(f"[res] could not sync {repo} to {self.t.host}")
        ws = self.t.workspace
        link = (
            f'for d in {ws}/resources/*/; do n=$(basename "$d"); [ -e {root}/$n ] || ln -s "${{d%/}}" {root}/$n; done'
        )
        if self._ssh(link).returncode != 0:
            sys.exit(f"[res] could not link the asset repos on {self.t.host}")
        return [f"{root}/{repo}" for repo in code]

    def remove(self, keep_log: bool) -> None:
        """Delete the stage dir, or only its checkpoints and credentials when the log is kept for a failure."""
        what = f"{self.dir}/ckpt {self.dir}/env" if keep_log else self.dir
        if self.t.kind == "local":
            for p in what.split():
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else Path(p).unlink(missing_ok=True)
        else:
            self._ssh("rm -rf " + " ".join(shlex.quote(p) for p in what.split()))


def run(stage: Stage, argv: list[str]) -> int:
    """Stream the run to stdout and a stage log; exit 0 with a Python traceback counts as failure."""
    proc = stage.run(argv)
    saw_traceback = False
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        saw_traceback |= TRACEBACK in line
    rc = proc.wait()
    if rc == 0 and saw_traceback:
        print("[res] the script exited 0 but printed a Python traceback; reporting failure", file=sys.stderr)
        return 1
    return rc


def _lease_card(lease_id: str, holder: str, pools: list[Pool]) -> tuple[Pool, str, int]:
    """Pool, host and gpu of a lease the caller already holds."""
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


def cmd_eval(args: argparse.Namespace, pools: list[Pool], claim: Callable) -> None:
    """Run a script on one leased card; release the lease eval took, on success and failure alike."""
    script = os.path.abspath(args.script)
    if not os.path.isfile(script):
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
    rc = 1
    try:
        t = make_target(pool, host, gpu)
        stage = Stage(t)
        stage.make()
        paths = {}
        for ref in refs:
            src = fetch_checkpoint(ref)
            paths[ref.name] = stage.put(src, f"ckpt/{ref.name}/{os.path.basename(src)}", link=True)
        source = t.workspace if t.kind == "local" else local_workspace()
        pythonpath = stage.sync_code(local_code(source, args.wt))
        stage.put(script, os.path.basename(script), mode=0o644)
        lines = [f"{k}={shlex.quote(v)}" for k, v in {**wandb_env(), **env, **paths}.items()]
        stage.write("\n".join(lines) + "\n", "env")
        runner = runner_script(t, stage.dir, script, sorted({**env, **paths}), pythonpath)
        stage.write(runner, "run.sh", mode=0o700)
        print(f"[res] running {os.path.basename(script)} on {t.host}:gpu{t.gpu} (stage {stage.dir})", file=sys.stderr)
        rc = run(stage, args.script_args)
        stage.remove(keep_log=rc != 0)
        if rc != 0:
            print(f"[res] FAILED with status {rc}; stage kept at {t.host}:{stage.dir}", file=sys.stderr)
    finally:
        if taken is not None:
            with ls.locked_store() as leases:
                leases[:] = [x for x in leases if x.id != taken.id]
            print(f"[res] released {taken.id}", file=sys.stderr)
    sys.exit(rc)


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register `eval` on the `just res` subparsers."""
    ev = sub.add_parser("eval", help="run a one-off script on a leased GPU (local or ssh pools)")
    ev.add_argument("script", help="script on this machine (copied to the target); its arguments follow --")
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


def split_script_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split an `eval` command line at its first ``--``: res's own arguments, then the script's (verbatim)."""
    if argv[:1] == ["eval"] and "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []
