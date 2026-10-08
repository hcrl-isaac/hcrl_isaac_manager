"""`pls <verb> [args...]`: parse the verb, then hand the rest to it unchanged (`pls res --help` is res.py's help)."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence

from hcrl_cli import infra, workspace
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


# verb ->(usage, summary, handler). Argument-passing verbs forward everything after the verb verbatim.
VERBS: dict[str, tuple[str, str, Callable[[list[str]], None]]] = {
    "setup": ("", "full local install: deps + Isaac Lab / Isaac Sim + every workspace package", _no_args(workspace.setup)),
    "deps": ("", "manager base env: the ilab venv (with pls), uv, gitman, git-lfs, W&B creds", _no_args(workspace.deps)),
    "resolve": ("[args]", "merge selection + defaults -> gitman.yaml, fetch every repo under resources/", workspace.resolve),
    "run": ("<script> [args]", "run hcrl_isaaclab/scripts/<script>.py with the ilab venv (WT=<name>: worktree set)", _first("script", infra.run_script)),
    "res": ("[args]", "compute: probe every GPU, claim/release leases, eval on a leased card", infra.res),
    "cluster": ("[<name>] [args]", "cluster interface: add/setup/job/develop/repush/build", infra.cluster),
    "ray": ("[args]", "Ray interface: setup/job/bench/push/list/logs/stop", infra.ray),
    "sync": ("<host> [args]", "push this workspace + Claude sessions to a peer box", _first("host", infra.sync)),
    "sim2real": ("<cmd> [args]", "hcrl_sim2real: MuJoCo sim/sysid, replay, fits, measure, fetch-model/policy", sim2real),
    "upload-artifacts": ("[args]", "upload managed large-file resources to W&B as artifacts", infra.upload_artifacts),
    "new": ("<name>", "scaffold a new <name>_tasks repo under resources/", _first("name", _new)),
    "vscode": ("[no-kit]", "generate .vscode/settings.json (no-kit: reuse the cached Kit paths)", workspace.vscode),
    "docker": ("[args]", "build the shared Isaac image (Ray + the HPC .sif)", workspace.docker),
    "docker-deps": ("", "regenerate the image's workspace-dependency list", _no_args(workspace.docker_deps)),
    "test": ("[<repo>] [args]", "a workspace repo's CPU smoke tests (default hcrl_isaaclab; needs a GPU)", _first("repo", workspace.test, "hcrl_isaaclab")),
    "test-scripts": ("", "the manager's own script tests (stubs only; CI runs these)", _no_args(workspace.test_scripts)),
    "clean": ("", "remove the ilab venv, generated workspace config and the shell alias (asks first)", _no_args(workspace.clean)),
}  # fmt: skip


def parser() -> argparse.ArgumentParser:
    width = max(len(f"{v} {u}") for v, (u, _, _) in VERBS.items())
    epilog = "\n".join(f"  {f'{v} {u}':<{width}}  {s}" for v, (u, s, _) in VERBS.items())
    p = argparse.ArgumentParser(
        prog="pls",
        description="Workspace command line: install, clusters, Ray, compute leases and runs.",
        epilog=f"verbs:\n{epilog}\n\nEach verb's own --help comes from the script it runs (e.g. pls res --help).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("verb", choices=VERBS, metavar="verb", help="one of the verbs below")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    if not argv:
        p.print_help()
        sys.exit(2)
    # argparse sees only the verb: the rest reaches the verb byte for byte, a literal `--` included.
    verb = p.parse_args(argv[:1]).verb
    if verb not in KEEP_CWD:
        os.chdir(ROOT)  # the other verbs run from the manager root, whatever the caller's cwd
    try:
        VERBS[verb][2](argv[1:])
    except KeyboardInterrupt:
        sys.exit(130)
