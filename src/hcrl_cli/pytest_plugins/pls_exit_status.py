"""Pytest plugin `pls test` loads: the process exits with pytest's own status even when Kit is running.

A task repo's conftest launches Kit at import, and Kit's exit handler ends the process with 0, so a failed run
otherwise exits 0 and a ``pls test || exit 1`` gate lets it through.
"""

from __future__ import annotations

import os
import sys

import pytest

_STATUS: list[int] = []


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _STATUS.append(int(exitstatus))


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    # only with Kit loaded: elsewhere pytest's own exit already carries the status
    if _STATUS and "omni.kit.app" in sys.modules:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_STATUS[-1])
