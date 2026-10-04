"""Backends that probe each compute pool for per-card ground truth (memory, processes, owners, wall time)."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, field

from inventory import Pool

# A card with this much memory in use but no visible process is held (e.g. Isaac kept its VRAM after exit).
HELD_MIB = 1024
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
MASTER_OPTS = ["-o", "ControlMaster=auto", "-o", f"ControlPath={os.path.expanduser('~')}/.ssh/cm/%C"]

# One round trip per host: card table, compute processes, then the owners of those processes.
GPU_QUERY = r"""
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo @@APPS
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
echo @@PS
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)
[ -n "$pids" ] && ps -o pid=,user=,etime=,args= -p "$pids"
true
"""


@dataclass
class Proc:
    """A process using a card."""

    pid: int
    mem_mib: int
    user: str = "?"
    elapsed: str = ""
    cmd: str = ""


@dataclass
class Card:
    """One GPU and what is on it."""

    pool: str
    host: str
    index: int
    model: str
    mem_used: int
    mem_total: int
    util: int
    state: str = "free"
    job: str = ""
    wall_left: str = ""
    procs: list[Proc] = field(default_factory=list)


@dataclass
class Report:
    """Probe result for one pool (or one host/job of it)."""

    pool: str
    kind: str
    cards: list[Card] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def short_cmd(args: str) -> str:
    """Compact a process command line: the script name plus --task, if any."""
    toks = args.split()
    script = next((os.path.basename(t) for t in toks if t.endswith(".py")), os.path.basename(toks[0]) if toks else "")
    task = next((toks[i + 1] for i, t in enumerate(toks[:-1]) if t == "--task"), "")
    return f"{script} {task}".strip()


def parse_gpu_query(out: str, pool: str, host: str) -> list[Card]:
    """Turn GPU_QUERY output into cards with their processes and a free/busy/held state."""
    sections = {"cards": [], "apps": [], "ps": []}
    key = "cards"
    for line in out.splitlines():
        if line.startswith("@@APPS"):
            key = "apps"
        elif line.startswith("@@PS"):
            key = "ps"
        elif line.strip():
            sections[key].append(line)
    owners = {}
    for line in sections["ps"]:
        parts = line.split(None, 3)
        if len(parts) >= 3 and parts[0].isdigit():
            owners[int(parts[0])] = (parts[1], parts[2], short_cmd(parts[3]) if len(parts) > 3 else "")
    procs_by_uuid: dict[str, list[Proc]] = {}
    for line in sections["apps"]:
        f = [x.strip() for x in line.split(",")]
        if len(f) < 3 or not f[1].isdigit():
            continue
        user, elapsed, cmd = owners.get(int(f[1]), ("?", "", ""))
        mem = int(f[2]) if f[2].isdigit() else 0
        procs_by_uuid.setdefault(f[0], []).append(Proc(int(f[1]), mem, user, elapsed, cmd))
    cards = []
    for line in sections["cards"]:
        f = [x.strip() for x in line.split(",")]
        if len(f) < 6 or not f[0].isdigit():
            continue
        procs = procs_by_uuid.get(f[1], [])
        used = max(int(f[3]) if f[3].isdigit() else 0, sum(p.mem_mib for p in procs))
        state = "busy" if procs else ("held" if used >= HELD_MIB else "free")
        util = int(f[5]) if f[5].isdigit() else 0
        cards.append(Card(pool, host, int(f[0]), f[2], used, int(f[4]) if f[4].isdigit() else 0, util, state, procs=procs))
    return cards


def probe_local(pool: Pool) -> list[Report]:
    host = os.uname().nodename.split(".")[0]
    try:
        res = _run(["bash", "-c", GPU_QUERY], timeout=30)
    except subprocess.TimeoutExpired:
        return [Report(pool.name, pool.kind, error="nvidia-smi timed out")]
    return [Report(pool.name, pool.kind, parse_gpu_query(res.stdout, pool.name, host))]


def _ssh_target(pool: Pool, host: str) -> str:
    if "@" in host:
        return host
    fqdn = host if "." in host or not pool.settings.get("domain") else f"{host}.{pool.settings['domain']}"
    return f"{pool.settings['user']}@{fqdn}" if pool.settings.get("user") else fqdn


def probe_ssh_host(pool: Pool, host: str) -> Report:
    try:
        res = _run(["ssh", *SSH_OPTS, _ssh_target(pool, host), GPU_QUERY], timeout=40)
    except subprocess.TimeoutExpired:
        return Report(f"{pool.name}/{host}", pool.kind, error="unreachable (ssh timed out)")
    cards = parse_gpu_query(res.stdout, pool.name, host)
    if res.returncode != 0 and not cards:
        err = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else f"ssh exit {res.returncode}"
        return Report(f"{pool.name}/{host}", pool.kind, error=f"unreachable ({err})")
    return Report(f"{pool.name}/{host}", pool.kind, cards)


def probe_ray(pool: Pool) -> list[Report]:
    addr = pool.settings["address"].rstrip("/")
    try:
        with urllib.request.urlopen(f"{addr}/nodes?view=summary", timeout=10) as r:
            nodes = json.load(r)["data"]["summary"]
        with urllib.request.urlopen(f"{addr}/api/jobs/", timeout=10) as r:
            jobs = json.load(r)
    except Exception as exc:  # noqa: BLE001 - any failure means the pool's state is unknown
        return [Report(pool.name, pool.kind, error=f"dashboard unreachable ({exc})")]
    report = Report(pool.name, pool.kind)
    for node in nodes:
        if node.get("raylet", {}).get("state") != "ALIVE":
            continue
        host = node.get("hostname", "?").split(".")[0]
        for g in node.get("gpus") or []:
            used = int(g.get("memoryUsed") or 0)
            pids = g.get("processesPids") or []
            procs = [Proc(int(p.get("pid", 0)), int(p.get("gpuMemoryUsage") or 0)) for p in pids if isinstance(p, dict)]
            state = "busy" if procs else ("held" if used >= HELD_MIB else "free")
            report.cards.append(
                Card(pool.name, host, int(g.get("index", 0)), g.get("name", ""), used,
                     int(g.get("memoryTotal") or 0), int(g.get("utilizationGpu") or 0), state, procs=procs)
            )
    for job in jobs:
        if job.get("status") in ("RUNNING", "PENDING"):
            ep = short_cmd(job.get("entrypoint", "")) or job.get("entrypoint", "")[:60]
            report.notes.append(f"job {job.get('submission_id') or job.get('job_id')} {job['status']}: {ep}")
    return [report]


def _master_alive(login: str) -> bool:
    try:
        return _run(["ssh", *MASTER_OPTS, "-O", "check", login], timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def probe_slurm_login(login: str, pools: list[Pool]) -> list[Report]:
    """Probe every job of the login's user once, for all profiles that share that login."""
    parts = login.split("@")[-1].split(".")
    site = next((p for p in parts if not p.startswith("login")), parts[0])
    label = f"{site} ({', '.join(p.name for p in pools)})"
    if not _master_alive(login):
        name = pools[0].name
        return [Report(label, "slurm", error=f"SSH master down: state unknown (just cluster {name} develop open)")]
    user = login.split("@")[0]
    fmt = "%i|%P|%a|%j|%T|%N|%L|%S"
    try:
        res = _run(["ssh", *MASTER_OPTS, *SSH_OPTS, login, f"squeue -u {user} -h -o '{fmt}'"], timeout=30)
    except subprocess.TimeoutExpired:
        return [Report(label, "slurm", error="squeue timed out")]
    if res.returncode != 0:
        return [Report(label, "slurm", error=f"squeue failed: {res.stderr.strip()[:200]}")]
    report = Report(label, "slurm")
    for line in res.stdout.splitlines():
        f = line.split("|")
        if len(f) < 8:
            continue
        jobid, part, acct, name, state, node, left, start = f
        if state != "RUNNING":
            report.notes.append(f"job {jobid} ({part}, {name}) {state}, est. start {start}")
            continue
        opts = f"--jobid={jobid} --overlap -N1 -n1 -t 00:02:00" + (f" -p {part}" if part else "")
        opts += f" -A {acct}" if acct and acct != "(null)" else ""
        remote = f"srun {opts} bash -c {shlex.quote(GPU_QUERY)}"
        try:
            step = _run(["ssh", *MASTER_OPTS, *SSH_OPTS, login, remote], timeout=60)
        except subprocess.TimeoutExpired:
            report.notes.append(f"job {jobid} on {node}: GPU probe timed out (state unknown)")
            continue
        cards = parse_gpu_query(step.stdout, report.pool, node)
        if not cards:
            report.notes.append(f"job {jobid} on {node}: GPU probe failed: {step.stderr.strip()[-200:]}")
        for card in cards:
            card.job, card.wall_left = jobid, left
        report.cards.extend(cards)
    if not report.cards and not report.notes:
        report.notes.append("no jobs held")
    return [report]
