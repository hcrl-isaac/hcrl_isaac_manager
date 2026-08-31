#!/usr/bin/env python3
"""Collect the runtime dependencies of the mounted workspace packages into a requirements file.

Workspace packages are PYTHONPATH'd into the image rather than pip-installed, so their dependencies
have to be baked in separately. Reading them from each repo's own metadata keeps that list from
drifting the way a hand-maintained one does.

Both declaration styles are read: ``[project].dependencies`` in pyproject.toml and ``INSTALL_REQUIRES``
in setup.py (parsed via AST, never executed).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path

# Packages the base image already provides. torch/torchvision matter most: the image ships +cu128
# builds and a plain "torch==2.7.0" from PyPI would be a CPU/other-CUDA wheel.
BASE_PROVIDED = {"torch", "torchvision", "isaaclab", "isaacsim"}

# Nested packages worth including even though they are not a top-level resources/<repo>. The
# retargeting stack is the installable part of holosoma; its sibling `holosoma` package declares a
# much larger dev-oriented set (open3d, opencv, notebook, mypy) that has no place in a runtime image.
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


def collect(resources: Path) -> tuple[dict[str, list[tuple[str, str]]], set[str]]:
    """Gather every declared requirement, keyed by distribution name.

    Args:
        resources: The workspace ``resources/`` directory.

    Returns:
        Mapping of distribution name to (specifier, source repo) pairs, and the set of workspace
        package names to exclude as self-references.
    """
    directories = sorted(p for p in resources.iterdir() if p.is_dir())
    directories += [resources / nested for nested in NESTED_PACKAGES]

    workspace_names = set()
    for directory in directories:
        if not directory.is_dir():
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
    parser.add_argument("--resources", type=Path, default=Path(__file__).resolve().parents[2] / "resources")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "requirements.workspace.txt")
    parser.add_argument("--check", action="store_true", help="Exit non-zero if --out is out of date.")
    args = parser.parse_args()

    if not args.resources.is_dir():
        print(f"[deps] no resources dir at {args.resources}", file=sys.stderr)
        return 1

    found, workspace_names = collect(args.resources)
    excluded = workspace_names | BASE_PROVIDED

    lines, conflicts = [], []
    for name in sorted(found):
        if name in excluded:
            continue
        specs = sorted({spec for spec, _ in found[name]})
        sources = sorted({source for _, source in found[name]})
        if len(specs) > 1:
            conflicts.append(f"{name}: {specs} (from {sources})")
        # Every declared specifier is emitted: pip intersects repeated requirements for one project,
        # so the tightest pin wins without this script having to choose (and choose wrong).
        for spec in specs:
            lines.append(f"{spec}  # {', '.join(sources)}")

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
            print(f"[deps] {args.out.name} is STALE -- run `just docker-deps`", file=sys.stderr)
            return 1
        print(f"[deps] {args.out.name} is up to date ({len(lines)} requirements)")
        return 0

    args.out.write_text(content)
    print(f"[deps] wrote {args.out} ({len(lines)} requirements from {len(found)} declared)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
