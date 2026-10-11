"""Backends that probe each compute pool for per-card ground truth (memory, processes, owners, wall time)."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, field

from inventory import CLUSTER_CONFIG_DIR, Pool

# Idle cards read 0-545 MiB and the smallest card seen in use reads ~2.4 GB, so 1 GiB sits in that gap: at or
# above this with no visible process a card is held (e.g. Isaac kept its VRAM after exit).
HELD_MIB = 1024
# Utilization at or above this with no visible process also means the card is in use.
HELD_UTIL = 10
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
# Over an existing master only: with its socket gone, ProxyCommand=false makes the call fail (UNKNOWN) instead
# of dialing a new connection that would need 2FA.
MASTER_OPTS = ["-o", "ControlMaster=no", "-o", f"ControlPath={os.path.expanduser('~')}/.ssh/cm/%C"]
MASTER_OPTS += ["-o", "ProxyCommand=false"]

# One round trip per host. Each section ends with a marker carrying its nvidia-smi exit code, and the ps pid
# list comes from the same apps query, so a truncated or failed probe is detectable.
GPU_QUERY = r"""
echo "@@CARDS"
out=$(nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>&1)
rc=$?; printf '%s\n' "$out"; echo "@@APPS rc=$rc"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>&1)
rc=$?; printf '%s\n' "$apps"; echo "@@PS rc=$rc"
pids=$(printf '%s\n' "$apps" | awk -F', *' '$2 ~ /^[0-9]+$/ {print $2}' | paste -sd, -)
[ -n "$pids" ] && ps -o pid=,user=,etime=,args= -p "$pids"
echo "@@END"
"""
REMOTE_QUERY = f"bash -c {shlex.quote(GPU_QUERY)}"


class ProbeError(Exception):
    """The probe output is incomplete or reports a failure, so the cards' state is unknown."""


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
    uuid: str = ""
    note: str = ""
    procs: list[Proc] = field(default_factory=list)


@dataclass
class Report:
    """Probe result for one pool (or one host/job of it); `error` means its cards' state is unknown."""

    pool: str
    kind: str
    cards: list[Card] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str = ""
    owner: str = ""  # OS user the probe ran as (whose processes count as a lease holder's); "" = unknown
    partial: bool = False  # some cards of this report could not be listed (absence proves nothing)

    def to_dict(self) -> dict:
        """Return the report as plain data for --json."""
        return asdict(self)


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _int(text: str) -> int | None:
    text = text.strip()
    return int(text) if text.isdigit() else None


def short_cmd(args: str) -> str:
    """Compact a process command line to the script name plus its --task, if any.

    Args:
        args: Full command line.

    Returns:
        E.g. `train.py T1-Kick-v0`.
    """
    toks = args.split()
    script = next((os.path.basename(t) for t in toks if t.endswith(".py")), os.path.basename(toks[0]) if toks else "")
    task = next((toks[i + 1] for i, t in enumerate(toks[:-1]) if t == "--task"), "")
    task = task or next((t.split("=", 1)[1] for t in toks if t.startswith("--task=")), "")
    return f"{script} {task}".strip()


MARKERS = ["@@CARDS", "@@APPS", "@@PS", "@@END"]


def _sections(out: str) -> dict[str, list[str]]:
    """Split GPU_QUERY output into its sections; banner text around them is ignored.

    A site's job wrapper (TACC's srun) prints its own checks on the step's stdout, before the query and after it, and
    can leave the first marker mid-line ("Checking available allocation (X)...@@CARDS").
    """
    sections: dict[str, list[str]] = {"cards": [], "apps": [], "ps": []}
    keys = {"@@CARDS": "cards", "@@APPS": "apps", "@@PS": "ps"}
    rcs: dict[str, str] = {}
    seen: list[str] = []
    key = None
    for line in out.splitlines():
        at = line.find("@@")
        marker = line[at:].split(" ", 1)[0] if at >= 0 else None
        if marker in MARKERS:
            line = line[at:]
            if marker in seen or MARKERS.index(marker) != len(seen):
                raise ProbeError(f"probe markers out of order at {marker}")
            seen.append(marker)
            if marker == "@@APPS":
                rcs["cards"] = line.partition("rc=")[2].strip()
            elif marker == "@@PS":
                rcs["apps"] = line.partition("rc=")[2].strip()
            key = keys.get(marker)
        elif key is not None and line.strip():
            sections[key].append(line)
    if seen != MARKERS:
        raise ProbeError("probe output truncated")
    for name, label in (("cards", "card query"), ("apps", "process query")):
        if rcs[name] != "0":
            first = sections[name][0] if sections[name] else ""
            raise ProbeError(f"nvidia-smi {label} failed (rc={rcs[name]}): {first}"[:200])
    return sections


def parse_gpu_query(out: str, pool: str, host: str) -> list[Card]:
    """Turn GPU_QUERY output into cards with their processes and a free/busy/held/unknown state.

    Args:
        out: Stdout of GPU_QUERY.
        pool: Pool name for the cards.
        host: Host name for the cards.

    Returns:
        The cards; raises ProbeError when the output is incomplete, failed or has no cards.
    """
    sections = _sections(out)
    owners = {}
    for line in sections["ps"]:
        parts = line.split(None, 3)
        if len(parts) >= 3 and parts[0].isdigit():
            owners[int(parts[0])] = (parts[1], parts[2], short_cmd(parts[3]) if len(parts) > 3 else "")
    procs_by_uuid: dict[str, list[Proc]] = {}
    for line in sections["apps"]:
        f = [x.strip() for x in line.split(",")]
        if len(f) != 3 or _int(f[1]) is None:
            raise ProbeError(f"unparsable process line: {line[:120]}")
        user, elapsed, cmd = owners.get(int(f[1]), ("?", "", ""))
        procs_by_uuid.setdefault(f[0], []).append(Proc(int(f[1]), _int(f[2]) or 0, user, elapsed, cmd))
    cards = []
    for line in sections["cards"]:
        f = [x.strip() for x in line.split(",")]
        if len(f) < 6 or _int(f[0]) is None:
            raise ProbeError(f"unparsable card line: {line[:120]}")
        # name may contain commas: index and uuid from the front, the three numbers from the back
        index, uuid, name = int(f[0]), f[1], ", ".join(f[2:-3])
        used, total, util = (_int(x) for x in f[-3:])
        procs = procs_by_uuid.pop(uuid, [])
        card = Card(
            pool, host, index, name, used if used is not None else -1, total or 0, util or 0, uuid=uuid, procs=procs
        )
        if used is None or util is None:
            card.state, card.note = "unknown", "memory/utilization not reported"
        cards.append(card)
    if not cards:
        raise ProbeError("nvidia-smi listed no cards")
    if procs_by_uuid:
        raise ProbeError(f"processes on cards missing from the card table: {', '.join(procs_by_uuid)}")
    _drop_stray_contexts(cards)
    for card in cards:
        if card.state == "unknown":
            continue
        if card.procs:
            card.state, card.mem_used = "busy", max(card.mem_used, sum(p.mem_mib for p in card.procs))
        elif card.mem_used >= HELD_MIB or card.util >= HELD_UTIL:
            card.state = "held"
    return cards


def _drop_stray_contexts(cards: list[Card]) -> None:
    """Move a process off a card where it only holds a CUDA context and into that card's note.

    A process under HELD_MIB on a card is dropped only when it holds at least HELD_MIB on another card of the host.
    """
    main: dict[int, tuple[int, Card]] = {}  # pid -> (its largest memory on one card, that card)
    for card in cards:
        for p in card.procs:
            if p.mem_mib >= HELD_MIB and p.mem_mib > main.get(p.pid, (0, None))[0]:
                main[p.pid] = (p.mem_mib, card)
    for card in cards:
        stray = [p for p in card.procs if p.mem_mib < HELD_MIB and p.pid in main and main[p.pid][1] is not card]
        if not stray:
            continue
        card.procs = [p for p in card.procs if p not in stray]
        what = ", ".join(f"{p.cmd or p.pid} (card {main[p.pid][1].index})" for p in stray)
        card.note = "; ".join(n for n in (card.note, f"CUDA context only: {what}") if n)


def probe_local(pool: Pool) -> list[Report]:
    """Probe this machine's cards."""
    host = os.uname().nodename.split(".")[0]
    try:
        res = _run(["bash", "-c", GPU_QUERY], timeout=30)
        cards = parse_gpu_query(res.stdout, pool.name, host)
        return [Report(pool.name, pool.kind, cards, owner=os.environ.get("USER", ""))]
    except (subprocess.TimeoutExpired, ProbeError) as exc:
        return [Report(pool.name, pool.kind, error=str(exc) or "nvidia-smi timed out")]


def _ssh_target(pool: Pool, host: str) -> str:
    if "@" in host:
        return host
    fqdn = host if "." in host or not pool.settings.get("domain") else f"{host}.{pool.settings['domain']}"
    return f"{pool.settings['user']}@{fqdn}"


def probe_ssh_host(pool: Pool, host: str) -> Report:
    """Probe one ssh host of a pool."""
    label = f"{pool.name}/{host}"
    if not pool.settings.get("user") and "@" not in host:
        return Report(
            label, pool.kind, error=f"no ssh user: set user under [compute.{pool.name}] in compute.local.toml"
        )
    try:
        res = _run(["ssh", *SSH_OPTS, _ssh_target(pool, host), REMOTE_QUERY], timeout=40)
    except subprocess.TimeoutExpired:
        return Report(label, pool.kind, error="unreachable (ssh timed out)")
    try:
        cards = parse_gpu_query(res.stdout, pool.name, host)
        return Report(label, pool.kind, cards, owner=_ssh_target(pool, host).split("@")[0])
    except ProbeError as exc:
        tail = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else ""
        return Report(label, pool.kind, error=f"{exc}{f' ({tail})' if tail else ''}")


def _num(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def ray_card(g: dict, pool: str, host: str) -> Card:
    """Card from one Ray dashboard `gpus` entry; missing readings make it unknown, never free.

    Args:
        g: The dashboard's GPU entry.
        pool: Pool name for the card.
        host: Node host name.

    Returns:
        The card; raises ProbeError when the entry lacks its index or uuid.
    """
    index, uuid = _num(g.get("index")), g.get("uuid")
    if index is None or not isinstance(uuid, str):
        raise ProbeError(f"unrecognized GPU entry (keys: {', '.join(sorted(g))[:120]})")
    used, util, total = _num(g.get("memoryUsed")), _num(g.get("utilizationGpu")), _num(g.get("memoryTotal"))
    raw = g.get("processesPids")
    raw = [] if raw is None else raw
    well_formed = isinstance(raw, list) and all(isinstance(p, dict) and _num(p.get("pid")) is not None for p in raw)
    pids = raw if well_formed else []
    procs = [Proc(_num(p.get("pid")) or 0, _num(p.get("gpuMemoryUsage")) or 0, cmd="ray worker") for p in pids]
    mem = -1 if used is None else used
    card = Card(pool, host, index, str(g.get("name", "")), mem, total or 0, util or 0, uuid=uuid, procs=procs)
    if used is None or util is None or used < 0 or util < 0:
        card.state, card.note = "unknown", "memory/utilization not reported"
    elif not well_formed:
        card.state, card.note = "unknown", "unreadable process list"
    elif procs:
        card.state = "busy"
    elif used >= HELD_MIB or util >= HELD_UTIL:
        card.state, card.note = "busy", "in use, process not visible to Ray"
    return card


def probe_ray(pool: Pool) -> list[Report]:
    """Probe a Ray cluster through its dashboard API."""
    addr = pool.settings["address"].rstrip("/")
    try:
        with urllib.request.urlopen(f"{addr}/nodes?view=summary", timeout=10) as r:
            nodes = json.load(r)["data"]["summary"]
        with urllib.request.urlopen(f"{addr}/api/jobs/", timeout=10) as r:
            jobs = json.load(r)
    except Exception as exc:  # any failure means the pool's state is unknown
        return [Report(pool.name, pool.kind, error=f"dashboard unreachable ({exc})")]
    if not isinstance(nodes, list) or not isinstance(jobs, list):
        return [Report(pool.name, pool.kind, error="dashboard returned an unexpected format")]
    report = Report(pool.name, pool.kind)
    by_ip: dict[str, list[Card]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("gpus") or [], list):
            return [Report(pool.name, pool.kind, error="dashboard returned an unexpected node entry")]
        host = str(node.get("hostname", "?")).split(".")[0]
        state = node.get("raylet", {}).get("state")
        if state != "ALIVE":
            report.notes.append(f"node {host} is {state}: its cards are unknown")
            report.partial = True
            continue
        try:
            cards = [ray_card(g, pool.name, host) for g in node.get("gpus") or []]
        except ProbeError as exc:
            return [Report(pool.name, pool.kind, error=f"node {host}: {exc}")]
        report.cards.extend(cards)
        by_ip.setdefault(str(node.get("ip", "")), []).extend(cards)
    for job in jobs:
        if job.get("status") in ("RUNNING", "PENDING"):
            job_id = job.get("submission_id") or job.get("job_id")
            ep = short_cmd(job.get("entrypoint", "")) or job.get("entrypoint", "")[:60]
            report.notes.append(f"job {job_id} {job['status']}: {ep}")
        if job.get("status") == "RUNNING":
            # the dashboard's GPU readings lag a starting job, so a card can read idle under a running one
            ip = str((job.get("driver_info") or {}).get("node_ip_address", ""))
            for card in by_ip.get(ip, []):
                if card.state == "free":
                    card.state, card.note = "busy", f"Ray job {job_id} runs on this node"
    return [report]


def _master_alive(login: str) -> bool:
    try:
        return _run(["ssh", *MASTER_OPTS, "-O", "check", login], timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _gres_gpus(gres: str) -> int | None:
    """GPU count from a squeue %b value such as `gres/gpu:4` or `gpu:a40:4`."""
    for part in gres.split(","):
        if "gpu" in part:
            count = part.rsplit(":", 1)[-1]
            return int(count) if count.isdigit() else None
    return None


def profile_accounts(pools: list[Pool]) -> dict[str, str]:
    """The accounts these pools' profiles submit with (``#SBATCH -A``), keyed by their lowercase form.

    squeue reports an account lowercased (cda26011), and TACC's submit filter refuses that spelling in an overlap step.
    """
    found = {}
    for pool in pools:
        script = CLUSTER_CONFIG_DIR / pool.name / "submit_job_slurm.sh"
        if not script.is_file():
            continue
        for line in script.read_text().splitlines():
            m = re.match(r"#SBATCH\s+(?:-A\s*|--account[=\s]+)(\S+)", line.strip())
            if m:
                found[m.group(1).lower()] = m.group(1)
    return found


def probe_slurm_login(login: str, pools: list[Pool]) -> list[Report]:
    """Probe every job of the login's user once, for all profiles that share that login.

    Args:
        login: `user@host` of the cluster's login node (over its SSH master).
        pools: Profiles that use this login.

    Returns:
        One report for the login plus one error report per job whose cards could not be probed.
    """
    parts = login.split("@")[-1].split(".")
    site = next((p for p in parts if not p.startswith("login")), parts[0])
    label = f"{site} ({', '.join(p.name for p in pools)})"
    if not _master_alive(login):
        return [
            Report(label, "slurm", error=f"SSH master down: state unknown (pls cluster {pools[0].name} develop open)")
        ]
    fmt = "%i|%P|%a|%T|%N|%D|%L|%S|%b|%j"
    try:
        res = _run(["ssh", *MASTER_OPTS, *SSH_OPTS, login, f"squeue --me -h -o '{fmt}'"], timeout=30)
    except subprocess.TimeoutExpired:
        return [Report(label, "slurm", error="squeue timed out")]
    if res.returncode != 0:
        return [Report(label, "slurm", error=f"squeue failed: {res.stderr.strip()[:200]}")]
    report, failed = Report(label, "slurm", owner=login.split("@")[0]), []
    spelling = profile_accounts(pools)
    for line in res.stdout.splitlines():
        f = line.split("|", 9)
        if len(f) < 10:
            report.notes.append(f"unparsable squeue line: {line[:120]}")
            report.partial = True
            continue
        jobid, part, acct, state, node, nnodes, left, start, gres, name = f
        if state == "PENDING":
            report.notes.append(f"job {jobid} ({part}, {name}) PENDING, est. start {start}")
            continue
        if state != "RUNNING":
            failed.append(Report(f"{label} job {jobid}", "slurm", error=f"job is {state}: its cards are unknown"))
            continue
        node = node.split(",")[0]
        # -p/-A because TACC refuses overlap steps without them on multi-project accounts, and -A as the profile
        # spells it
        acct = spelling.get(acct.lower(), acct)
        opts = f"--jobid={jobid} --overlap --immediate=10 --job-name=res-probe -N1 -n1 -t 00:02:00"
        opts += (f" -p {part}" if part else "") + (f" -A {acct}" if acct and acct != "(null)" else "")
        remote = f"timeout 45 srun {opts} {REMOTE_QUERY}"
        try:
            step = _run(["ssh", *MASTER_OPTS, *SSH_OPTS, login, remote], timeout=60)
            cards = parse_gpu_query(step.stdout, report.pool, node)
        except subprocess.TimeoutExpired:
            failed.append(Report(f"{label} job {jobid}", "slurm", error=f"GPU probe on {node} timed out"))
            continue
        except ProbeError as exc:
            tail = step.stderr.strip()[-160:]
            failed.append(Report(f"{label} job {jobid}", "slurm", error=f"GPU probe on {node}: {exc} {tail}".strip()))
            continue
        for card in cards:
            card.job, card.wall_left = jobid, left
        report.cards.extend(cards)
        if nnodes.isdigit() and int(nnodes) > 1:
            failed.append(Report(f"{label} job {jobid}", "slurm", error=f"probed 1 of {nnodes} nodes"))
        alloc = _gres_gpus(gres)
        if alloc is not None and alloc > len(cards):
            msg = f"saw {len(cards)} of {alloc} allocated cards; {alloc - len(cards)} unknown"
            failed.append(Report(f"{label} job {jobid}", "slurm", error=msg))
    if not report.cards and not report.notes and not failed:
        report.notes.append("no jobs held")
    return [report, *failed]
