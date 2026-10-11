"""Before a Ray submit: fail if an entry the job leaves to W&B artifacts has no artifact at its path.

Ray jobs exclude every exported policy (``policies/<task>/<robot>/<name>``, a dir or a file) and every
``style_data`` dir from their upload (scripts/ray/tools/*.template.yaml), which keeps the py_modules zip under Ray's
100 MiB limit; the container then fetches them as artifacts by rel_path. An entry with no artifact would only fail
inside the job, so this checks every such entry in the sources the job mounts.
"""

from __future__ import annotations

import importlib.util
import os
import sys

MANAGER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESOURCES = os.path.join(MANAGER_DIR, "resources")
SKIP_DIRS = {".git", "worktrees", "logs", "outputs", "wandb", "__pycache__", ".claude"}
# the excludes every job template must carry for artifact_only_dirs() to describe what the job leaves out
ARTIFACT_EXCLUDES = ("**/policies/*/*/*", "**/style_data/**")
# other excludes every job template must carry: gitignored session docs, and git state (which collides on the worker)
UPLOAD_EXCLUDES = ("**/.claude/**", "**/.git")

sys.path.insert(0, os.path.join(MANAGER_DIR, "scripts"))


def artifact_only_dirs(src: str, repo: str) -> list[str]:
    """Rel paths (``<repo>/...``) of the entries under ``src`` that Ray jobs exclude and fetch as artifacts.

    Args:
        src: The repo checkout or worktree the job mounts.
        repo: The repo name, i.e. its directory under the resources dir.

    Returns:
        Sorted rel paths matching ``**/policies/*/*/*`` (dirs and files) and ``**/style_data``.
    """
    found = []
    for root, dirs, files in os.walk(src):
        parts = [] if root == src else os.path.relpath(root, src).split(os.sep)
        if len(parts) >= 3 and parts[-3] == "policies":
            found += ["/".join([repo, *parts, e]) for e in [*dirs, *files] if e not in SKIP_DIRS]
            dirs[:] = []
            continue
        keep = []
        for d in sorted(dirs):
            if d in SKIP_DIRS:
                continue
            if d == "style_data":
                found.append("/".join([repo, *parts, d]))
            else:
                keep.append(d)
        dirs[:] = keep
    return sorted(found)


def mounted_sources(wt: str) -> list[tuple[str, str]]:
    """(source path, repo) for every workspace repo, using ``resources/<repo>/worktrees/<wt>`` where it exists."""
    from worktree_env import workspace_repos

    out = []
    for repo in workspace_repos(RESOURCES):
        base = os.path.join(RESOURCES, repo)
        wdir = os.path.join(base, "worktrees", wt) if wt else ""
        out.append((wdir if wt and os.path.isdir(wdir) else base, repo))
    return out


def _published_rel_paths() -> set[str]:
    path = os.path.join(RESOURCES, "hcrl_isaaclab", "hcrl_isaaclab", "utils", "artifacts.py")
    spec = importlib.util.spec_from_file_location("hcrl_artifacts_preflight", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    specs = module.discover()
    return {s.rel_path for s in specs} | {r for s in specs for r in s.extra_rel_paths}


def main() -> None:
    """Exit 1 listing each artifact-only entry with no artifact; warn and pass if W&B cannot be read."""
    sources = mounted_sources(os.environ.get("WT", ""))
    wanted = [
        (rel, os.path.join(src, os.path.relpath(rel, repo)))
        for src, repo in sources
        for rel in artifact_only_dirs(src, repo)
    ]
    if not wanted:
        return
    try:
        published = _published_rel_paths()
    except Exception as exc:  # cannot verify: the job's own fetch reports a missing artifact by name
        print(f"[WARN] preflight: could not list W&B artifacts ({exc}); not checking.", file=sys.stderr)
        return
    missing = [(rel, local) for rel, local in wanted if rel not in published]
    if not missing:
        print(f"[INFO] preflight: all {len(wanted)} artifact-only entries are published.")
        return
    print(
        "[ERROR] preflight: the job excludes these from its upload, but no W&B artifact provides them:",
        file=sys.stderr,
    )
    for rel, local in missing:
        print(f"  {rel}\n    publish: just upload-artifacts {local} --rel-path {rel} --tier cache", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
