#!/usr/bin/env python3
"""Checkpoint refs for `pls run --on <card>`: parse them anywhere, download W&B ones under the ilab venv (needs wandb).

A ref is a local path, a W&B run URL (https://wandb.ai/<entity>/<project>/runs/<id>), or <entity>/<project>/<id>,
optionally suffixed with @<iteration> (model_<iteration>.pt); without it the run's latest checkpoint is used.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import types
from dataclasses import dataclass

_URL = re.compile(r"^https?://(www\.)?wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#@]+)")
_PATH3 = re.compile(r"^([\w.-]+)/([\w.-]+)/([\w-]+)$")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class CheckpointRef:
    """One parsed checkpoint ref.

    Args:
        name: Env var the resolved path is exported as.
        local: Local path, or "" for a W&B ref.
        run_path: ``entity/project/run_id`` for a W&B ref.
        model: ``model_<it>.pt`` to fetch, or "" for the run's latest.
    """

    name: str
    local: str = ""
    run_path: str = ""
    model: str = ""


def parse(spec: str, default_name: str = "CHECKPOINT") -> CheckpointRef:
    """Parse ``[NAME=]<ref>``; raises ValueError for anything that is neither an existing path nor a W&B run.

    Args:
        spec: The ``--checkpoint`` value.
        default_name: Env var name when the spec has no ``NAME=``.

    Returns:
        The parsed ref.
    """
    name, ref = default_name, spec
    head, sep, tail = spec.partition("=")
    if sep and _NAME.match(head) and not head.startswith(("http", "/")):
        name, ref = head, tail
    if not ref:
        raise ValueError(f"empty checkpoint ref in {spec!r}")
    base, it = ref, ""
    if "@" in ref and not os.path.exists(ref):
        base, _, it = ref.rpartition("@")
        if not it.isdigit():
            raise ValueError(f"{spec!r}: @ must be followed by an iteration number")
    model = f"model_{it}.pt" if it else ""
    m = _URL.match(base)
    if m:
        return CheckpointRef(name, run_path=f"{m.group(2)}/{m.group(3)}/{m.group(4)}", model=model)
    if os.path.exists(ref):
        return CheckpointRef(name, local=os.path.abspath(ref))
    m = _PATH3.match(base)
    if m and not base.startswith((".", "/")):
        return CheckpointRef(name, run_path=base, model=model)
    raise ValueError(f"{spec!r} is not an existing path, a W&B run URL or entity/project/run_id")


def _cli_args(core: str) -> types.ModuleType:
    """hcrl_isaaclab's scripts/cli_args.py, whose W&B download is the one `--load_run` uses."""
    path = os.path.join(core, "scripts", "cli_args.py")
    spec = importlib.util.spec_from_file_location("hcrl_cli_args", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def download(core: str, cache: str, run_path: str, model: str) -> str:
    """Download one W&B checkpoint into the cache, or reuse the cached copy.

    Args:
        core: The hcrl_isaaclab checkout whose scripts/cli_args.py does the download.
        cache: Cache root.
        run_path: ``entity/project/run_id``.
        model: ``model_<it>.pt``, or "" for the run's latest.

    Returns:
        ``<cache>/<entity>/<project>/<run_id>/<model>``, which is only ever a complete file.
    """
    import shutil
    import uuid

    import wandb

    if not model:
        names = [f.name for f in wandb.Api().run(run_path).files() if re.fullmatch(r"model_\d+\.pt", f.name)]
        if not names:
            raise FileNotFoundError(f"no model_*.pt checkpoints in W&B run {run_path}")
        model = max(names, key=lambda n: int(n[6:-3]))
    final = os.path.join(cache, run_path, model)
    if os.path.isfile(final):
        return final
    tmp = os.path.join(cache, f".tmp-{uuid.uuid4().hex}")
    try:
        got = _cli_args(core).download_checkpoint_from_wandb(tmp, run_path, model)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        os.replace(got, final)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final


def main() -> None:
    """``checkpoints.py <hcrl_isaaclab dir> <cache dir> <run_path> [model]``: print the downloaded path last."""
    core, cache, run_path = sys.argv[1:4]
    model = sys.argv[4] if len(sys.argv) > 4 else ""
    print(download(core, cache, run_path, model))


if __name__ == "__main__":
    main()
