"""`pls <verb> [args...]`: parse the verb, then hand the rest to it unchanged (`pls res --help` is res.py's help)."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence

from hcrl_cli import infra, launch, proc, workspace
from hcrl_cli.proc import ROOT
from hcrl_cli.sim2real import sim2real

KEEP_CWD = {"sim2real"}  # verbs whose path arguments are relative to the caller, not the manager root


def _no_args(fn: Callable[[], None]) -> Callable[[list[str]], None]:
    def call(args: list[str]) -> None:
        if args:
            sys.exit(f"[pls] unexpected arguments: {' '.join(args)}")
        fn()

    return call


def _first(name: str, fn: Callable[[str, list[str]], None], default: str | None = None) -> Callable[[list[str]], None]:
    """A verb whose first argument is required (or defaulted) and the rest pass through."""

    def call(args: list[str]) -> None:
        if args:
            fn(args[0], args[1:])
        elif default is not None:
            fn(default, [])
        else:
            sys.exit(f"[pls] missing <{name}>")

    return call


def _new(name: str, rest: list[str]) -> None:
    _no_args(lambda: workspace.new(name))(rest)


SETUP_HELP = """usage: pls setup                                  this machine: deps + Isaac Lab 3 + every workspace package
       pls setup --on <cluster> [--build-only|--push-only]  build the shared image and push it to that cluster
                                                  (--on ray: set up the Ray cluster's credentials and venv)
       pls setup --add <name> [--update]          create (or regenerate) a SLURM cluster profile
       pls setup --add <name> --kind ssh --hosts <a,b> --user <login> [--domain <d>] [--pin device] [--render]
                                                  add ssh GPU boxes as a pool (in compute.local.toml, gitignored)
       pls setup --add ray --kind ray --address http://<head>:8265   add your Ray cluster's dashboard as the ray pool"""


def _setup(args: list[str]) -> None:
    if not args:
        return workspace.setup()
    if args[0] in ("-h", "--help"):
        print(SETUP_HELP)
        sys.exit(0)
    if args[0].startswith("--add="):
        return infra.add_cluster(args[0].removeprefix("--add="), args[1:])
    if args[0] == "--add":
        named = len(args) > 1 and not args[1].startswith("-")
        return infra.add_cluster(args[1] if named else "", args[2:] if named else args[1:])
    if any(a == "--on" or a.startswith("--on=") for a in args):
        cluster, rest = infra._on("setup", args)
        return infra.setup_on(cluster, rest)
    sys.exit(f"[pls] setup: unexpected arguments {' '.join(args)}\n{SETUP_HELP}")


# verb -> (group, usage, summary, handler). Argument-passing verbs forward everything after the verb verbatim.
VERBS: dict[str, tuple[str, str, str, Callable[[list[str]], None]]] = {
    "run": ("run", "[--on T] [--wt S] [--batch] [--cmd] -- <script|command> [args]", "run anywhere: here, a leased card, a cluster batch job or Ray", launch.main),
    "jobs": ("run", "--on C [list|logs J|stop J|kill]", "batch jobs, Ray jobs and run steps on a cluster", infra.jobs),
    "res": ("compute", "[status|claim|release|hold|drop|...] [args]", "GPUs: who holds what, leases, and held cluster nodes", infra.res),
    "connect": ("cluster", "--on C", "open a cluster's SSH connection (2FA once; others reuse it)", infra.connect),
    "disconnect": ("cluster", "--on C", "close it", infra.disconnect),
    "shell": ("cluster", "--on C", "an interactive shell on the cluster's held node", infra.shell),
    "stage": ("cluster", "--on C [data|list|rm T|<name> <repo>=<ref>...]", "stage code trees on a cluster (pls run does it for you)", infra.stage),
    "setup": ("workspace", "[--on C | --add NAME]", "install this machine; or build/push a cluster's image, or add a cluster", _setup),
    "deps": ("workspace", "", "manager base env: the ilab venv (with pls), uv, gitman, git-lfs, W&B creds", _no_args(workspace.deps)),
    "install": ("workspace", "", "layer Isaac Lab 3 (uv-locked source clone) + every workspace package onto the ilab venv", _no_args(workspace.install)),
    "resolve": ("workspace", "[args]", "merge selection + defaults -> gitman.yaml, fetch every repo under resources/", workspace.resolve),
    "sync": ("workspace", "<host> [args]", "push this workspace + Claude sessions to a peer box", _first("host", infra.sync)),
    "new": ("workspace", "<name>", "scaffold a new <name>_tasks repo under resources/", _first("name", _new)),
    "vscode": ("workspace", "[no-kit]", "generate .vscode/settings.json (no-kit: reuse the cached Kit paths)", workspace.vscode),
    "docker": ("workspace", "[args]", "build the shared Isaac image (Ray + the HPC .sif)", workspace.docker),
    "docker-deps": ("workspace", "", "regenerate the image's workspace-dependency list", _no_args(workspace.docker_deps)),
    "clean": ("workspace", "", "remove the ilab venv, generated workspace config and the shell alias (asks first)", _no_args(workspace.clean)),
    "sim2real": ("other", "<cmd> [args]", "hcrl_sim2real: MuJoCo sim/sysid, replay, fits, measure, fetch-model/policy", sim2real),
    "upload-artifacts": ("other", "[args]", "upload managed large-file resources to W&B as artifacts", infra.upload_artifacts),
    "test": ("other", "[<repo>] [args]", "a workspace repo's CPU smoke tests (default hcrl_isaaclab; needs a GPU)", _first("repo", workspace.test, "hcrl_isaaclab")),
    "test-scripts": ("other", "", "the manager's own script tests (stubs only; CI runs these)", _no_args(workspace.test_scripts)),
    "cluster": ("retired", "", "", infra.cluster),
}  # fmt: skip
GROUPS = {"run": "running", "compute": "compute", "cluster": "clusters (C = a SLURM profile; jobs and setup also take ray)",
          "workspace": "workspace", "other": "other"}  # fmt: skip


def parser() -> argparse.ArgumentParser:
    shown = {v: e for v, e in VERBS.items() if e[0] in GROUPS}
    width = max(len(v) for v in shown)
    sections = []
    for group, title in GROUPS.items():
        rows = [f"  {v:<{width}}  {s}" for v, (g, _, s, _) in shown.items() if g == group]
        sections.append(f"{title}:\n" + "\n".join(rows))
    p = argparse.ArgumentParser(
        prog="pls",
        usage="pls <verb> [args]   (pls <verb> --help for its usage)",
        description="Workspace command line. A command that acts somewhere else names it with --on <where>:\n"
        "  pls run --on horizon -- train --task <id>     pls jobs --on horizon     pls res status --on horizon",
        epilog="\n\n".join(sections) + "\n\nEach verb has its own --help (e.g. pls run --help, pls res --help).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("verb", choices=VERBS, metavar="verb", help=argparse.SUPPRESS)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    if not argv:
        p.print_help()
        sys.exit(2)
    # argparse sees only the verb: the rest reaches the verb byte for byte, a literal `--` included.
    verb = p.parse_args(argv[:1]).verb
    proc.CALLER_CWD = os.getcwd()
    if verb not in KEEP_CWD:
        os.chdir(ROOT)  # the other verbs run from the manager root, whatever the caller's cwd
    try:
        VERBS[verb][3](argv[1:])
    except KeyboardInterrupt:
        sys.exit(130)
