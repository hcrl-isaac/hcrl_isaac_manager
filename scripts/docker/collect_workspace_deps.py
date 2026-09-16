#!/usr/bin/env python3
"""Collect the runtime dependencies of the mounted workspace packages into a requirements file.

Workspace packages are PYTHONPATH'd into the image rather than pip-installed, so their dependencies
have to be baked in separately. Reading them from each repo's own metadata keeps that list from
drifting the way a hand-maintained one does.

Repos come from the committed catalogue (``workspace.defaults.yaml``: the ``always`` list plus every
project's task repo and their ``dependencies.yaml`` closure), never from whatever happens to be cloned
locally, so the output is the same on every machine. Both declaration styles are read:
``[project].dependencies`` in pyproject.toml and ``INSTALL_REQUIRES`` in setup.py (AST, never executed).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path

import yaml

MANAGER_DIR = Path(__file__).resolve().parents[2]

# Packages the base image already provides. torch/torchvision matter most: the image ships +cu128
# builds and a plain "torch==2.7.0" from PyPI would be a CPU/other-CUDA wheel.
BASE_PROVIDED = {"torch", "torchvision", "isaaclab", "isaacsim"}

# Nested packages outside the catalogue: the retargeting stack is the runtime part of holosoma (its
# sibling `holosoma` package declares a dev-only set that has no place in the image).
NESTED_PACKAGES = ("holosoma/src/holosoma_retargeting",)


def normalize(name: str) -> str:
    """Normalize a distribution name for comparison (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(spec: str) -> str:
    """Extract the distribution name from a requirement specifier.

    Args:
        spec: A PEP 508 requirement string.

    Returns:
        The normalized distribution name, without extras, markers or version specifiers.
    """
    head = re.split(r"[<>=!~;\[@]", spec, maxsplit=1)[0]
    return normalize(head.strip())


def from_pyproject(path: Path) -> list[str]:
    """Read ``[project].dependencies`` from a pyproject.toml, if it declares any."""
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return list(data.get("project", {}).get("dependencies", []))


def from_setup_py(path: Path) -> list[str]:
    """Read a literal ``INSTALL_REQUIRES`` list from a setup.py without executing it."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "INSTALL_REQUIRES" not in names or not isinstance(node.value, ast.List):
            continue
        return [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def declared_name(directory: Path) -> str | None:
    """The distribution name a package directory declares, used to drop workspace self-references."""
    pyproject = directory / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as handle:
            name = tomllib.load(handle).get("project", {}).get("name")
        if name:
            return normalize(name)
    return None


def catalogue_repos(defaults: Path, resources: Path) -> list[str]:
    """List every repo the committed catalogue can select, in resolution order.

    Args:
        defaults: The ``workspace.defaults.yaml`` to read ``always`` and ``available_projects`` from.
        resources: The workspace ``resources/`` directory, for each cloned repo's ``dependencies.yaml``.

    Returns:
        Repo names: the ``always`` list, every project's task repo, and their declared dependencies
        (transitively, as far as the repos are cloned).
    """
    manifest = yaml.safe_load(defaults.read_text()) or {}
    queue = list(manifest.get("always", []))
    queue += [f"{p['name']}_tasks" for p in manifest.get("available_projects", [])]
    repos: list[str] = []
    while queue:
        name = queue.pop(0)
        if name in repos:
            continue
        repos.append(name)
        deps_file = resources / name / "dependencies.yaml"
        if deps_file.is_file():
            queue += [d["name"] for d in (yaml.safe_load(deps_file.read_text()) or {}).get("deps", []) or []]
    return repos


def collect(resources: Path, repos: list[str]) -> tuple[dict[str, list[tuple[str, str]]], set[str]]:
    """Gather every declared requirement, keyed by distribution name.

    Args:
        resources: The workspace ``resources/`` directory.
        repos: Repo names under ``resources/`` to read; ones not cloned are skipped with a warning.

    Returns:
        Mapping of distribution name to (specifier, source repo) pairs, and the set of workspace
        package names to exclude as self-references.
    """
    directories = [resources / name for name in repos]
    directories += [resources / nested for nested in NESTED_PACKAGES]

    workspace_names = set()
    for directory in directories:
        if not directory.is_dir():
            print(f"[deps] WARNING: {directory.relative_to(resources)} is not cloned -- skipped", file=sys.stderr)
            continue
        workspace_names.add(normalize(directory.name))
        if name := declared_name(directory):
            workspace_names.add(name)

    found: dict[str, list[tuple[str, str]]] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        specs: list[str] = []
        if (pyproject := directory / "pyproject.toml").is_file():
            specs += from_pyproject(pyproject)
        if (setup := directory / "setup.py").is_file():
            specs += from_setup_py(setup)
        for spec in specs:
            found.setdefault(requirement_name(spec), []).append((spec.strip(), directory.name))
    return found, workspace_names


def main() -> int:
    """CLI: write the workspace requirements file, or verify the checked-in one is current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, default=MANAGER_DIR / "resources")
    parser.add_argument("--defaults", type=Path, default=MANAGER_DIR / "workspace.defaults.yaml")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "requirements.workspace.txt")
    parser.add_argument("--check", action="store_true", help="Exit non-zero if --out is out of date.")
    args = parser.parse_args()

    if not args.resources.is_dir():
        print(f"[deps] no resources dir at {args.resources}", file=sys.stderr)
        return 1

    found, workspace_names = collect(args.resources, catalogue_repos(args.defaults, args.resources))
    excluded = workspace_names | BASE_PROVIDED

    lines, conflicts = [], []
    for name in sorted(found):
        if name in excluded:
            continue
        by_spec: dict[str, set[str]] = {}
        for spec, source in found[name]:
            by_spec.setdefault(spec, set()).add(source)
        if len(by_spec) > 1:
            conflicts.append(f"{name}: {sorted(by_spec)} (from {sorted(set().union(*by_spec.values()))})")
        # Every declared specifier is emitted: pip intersects repeated requirements for one project,
        # so the tightest pin wins without this script having to choose (and choose wrong).
        for spec in sorted(by_spec):
            lines.append(f"{spec}  # {', '.join(sorted(by_spec[spec]))}")

    header = [
        "# GENERATED by scripts/docker/collect_workspace_deps.py -- do not edit by hand.",
        "# Runtime deps of the workspace packages, which are PYTHONPATH'd into the image rather than",
        "# pip-installed. Regenerate with `just docker-deps` after changing any repo's dependencies.",
        f"# Excluded as base-image-provided: {', '.join(sorted(BASE_PROVIDED))}.",
        "",
    ]
    content = "\n".join(header + lines) + "\n"

    if conflicts:
        print("[deps] packages declared with differing specifiers (pip will intersect them):", file=sys.stderr)
        for conflict in conflicts:
            print(f"  {conflict}", file=sys.stderr)

    if args.check:
        current = args.out.read_text() if args.out.is_file() else ""
        if current != content:
            print(
                f"[deps] {args.out.name} is STALE -- run `just docker-deps` (with every catalogue repo cloned)",
                file=sys.stderr,
            )
            return 1
        print(f"[deps] {args.out.name} is up to date ({len(lines)} requirements)")
        return 0

    args.out.write_text(content)
    print(f"[deps] wrote {args.out} ({len(lines)} requirements from {len(found)} declared)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
