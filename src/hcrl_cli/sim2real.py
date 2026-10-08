"""`pls sim2real <cmd>`: hcrl_sim2real's commands, dispatched through its own registry (hcrl_sim2real.cli.COMMANDS)."""

from __future__ import annotations

import importlib
import sys


def sim2real(args: list[str]) -> None:
    """Run one hcrl_sim2real command in this interpreter; its paths stay relative to the caller's cwd."""
    try:
        cli = importlib.import_module("hcrl_sim2real.cli")
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] == "hcrl_sim2real":
            sys.exit(
                "[pls] hcrl_sim2real is not installed in this interpreter: select it in the workspace, then `pls setup`."
            )
        raise
    if not args or args[0] in ("-h", "--help") or args[0] not in cli.COMMANDS:
        print(cli.__doc__)
        sys.exit(0 if args and args[0] in ("-h", "--help") else 1)
    module, function = cli.COMMANDS[args[0]]
    sys.argv[0] = f"pls sim2real {args[0]}"  # the command's argparse names itself after argv[0] in its usage
    getattr(importlib.import_module(module), function)(args[1:])
