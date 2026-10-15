"""Before a Ray submit: fail if an entry the job leaves to W&B artifacts has no artifact at its path, or (from a
worktree set) holds other data than that artifact, which the job would silently train on instead.

Ray jobs exclude every exported policy (``policies/<task>/<robot>/<name>``, a dir or a file) and every
``style_data`` dir from their upload (scripts/ray/tools/*.template.yaml), which keeps the py_modules zip under Ray's
100 MiB limit; the container then fetches them as artifacts by rel_path. An entry with no artifact would only fail
inside the job, so this checks every such entry in the sources the job mounts.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import os
import sys
from types import ModuleType

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


def _artifacts_module() -> ModuleType:
    if "hcrl_artifacts_preflight" not in sys.modules:
        path = os.path.join(RESOURCES, "hcrl_isaaclab", "hcrl_isaaclab", "utils", "artifacts.py")
        spec = importlib.util.spec_from_file_location("hcrl_artifacts_preflight", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["hcrl_artifacts_preflight"]


def _published_rel_paths() -> set[str]:
    specs = _artifacts_module().discover()
    return {s.rel_path for s in specs} | {r for s in specs for r in s.extra_rel_paths}


def _published_digests(rels: list[str]) -> dict[str, tuple[str, list[str]]]:
    """rel path -> (artifact ``name:version`` the job fetches, its files' W&B digests) for each published rel."""
    import wandb

    module = _artifacts_module()
    by_rel = {r: s for s in module.discover() for r in (s.rel_path, *s.extra_rel_paths)}
    api, out = wandb.Api(), {}
    for rel in rels:
        if rel in by_rel:
            s = by_rel[rel]
            art = api.artifact(f"{module._project()}/{s.key}:{s.version}")
            entries = art.manifest.entries.items()
            digests = sorted(e.digest for name, e in entries if not name.lower().endswith(DOC_SUFFIXES))
            out[rel] = (f"{s.key}:{s.version}", digests)
    return out


def has_lfs_pointer(path: str) -> bool:
    """True if a file at ``path`` (or under it) is a git-lfs pointer rather than its content."""
    files = [path] if os.path.isfile(path) else [os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs]
    for f in files:
        with open(f, "rb") as fh:
            if fh.read(40).startswith(b"version https://git-lfs.github.com/spec/"):
                return True
    return False


DOC_SUFFIXES = (".md", ".rst")  # documentation beside the weights: never what a job loads, so not compared


def local_digests(path: str) -> list[str]:
    """W&B-style digests (base64 md5) of every file at ``path`` (a file, or a dir walked recursively), docs left out."""
    files = [path] if os.path.isfile(path) else [os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs]
    files = [f for f in files if not f.lower().endswith(DOC_SUFFIXES)]
    out = []
    for f in files:
        h = hashlib.md5()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        out.append(base64.b64encode(h.digest()).decode())
    return sorted(out)


def main() -> None:
    """Exit 1 listing each artifact-only entry with no artifact; warn and pass if W&B cannot be read."""
    wt = os.environ.get("WT", "")
    sources = mounted_sources(wt)
    from_wt = {repo for src, repo in sources if wt and src.endswith(os.path.join("worktrees", wt))}
    if wt:
        print(f"[INFO] preflight: WT={wt} mounts {', '.join(sorted(from_wt))} from worktrees/{wt}; others from main")
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
        _check_worktree_data([(rel, local) for rel, local in wanted if rel.split("/")[0] in from_wt], wt)
        return
    print(
        "[ERROR] preflight: the job excludes these from its upload, but no W&B artifact provides them:",
        file=sys.stderr,
    )
    for rel, local in missing:
        print(f"  {rel}\n    publish: pls upload-artifacts {local} --rel-path {rel} --tier cache", file=sys.stderr)
    sys.exit(1)


def _check_worktree_data(entries: list[tuple[str, str]], wt: str) -> None:
    """Exit 1 if a worktree's artifact-only entry differs from the artifact the job fetches in its place."""
    if not entries:
        return
    pointers = [rel for rel, local in entries if has_lfs_pointer(local)]
    if pointers:  # pointers never match an artifact's bytes; the job would get the published data, unchecked
        print(f"[ERROR] preflight: WT={wt} holds git-lfs pointers, not data, so it cannot be compared with W&B's:",
              file=sys.stderr)  # fmt: skip
        for rel in pointers:
            print(f"  {rel}", file=sys.stderr)
        print("  run `git lfs pull` in those worktrees, then submit again.", file=sys.stderr)
        sys.exit(1)
    try:
        published = _published_digests([rel for rel, _ in entries])
    except Exception as exc:  # cannot verify: warn, as for the presence check
        print(f"[WARN] preflight: could not read the artifacts' contents ({exc}); not comparing WT={wt} data.",
              file=sys.stderr)  # fmt: skip
        return
    differ = [(rel, published[rel][0]) for rel, local in entries
              if rel in published and local_digests(local) != published[rel][1]]  # fmt: skip
    accepted = {r for r in os.environ.get("WT_DATA_OK", "").split(os.pathsep) if r}  # pls run --wt-data-ok
    for rel, art in [x for x in differ if x[0] in accepted]:
        print(f"[WARN] preflight: WT={wt} differs from {art} at {rel}; --wt-data-ok accepts it, so the job trains on"
              " the published copy.", file=sys.stderr)  # fmt: skip
    differ = [x for x in differ if x[0] not in accepted]
    if not differ:
        return
    print(f"[ERROR] preflight: WT={wt} holds other data than the W&B artifact a Ray job fetches in its place:",
          file=sys.stderr)  # fmt: skip
    for rel, art in differ:
        print(f"  {rel}  (the job would get {art})", file=sys.stderr)
    print("  Ray excludes these from its upload; run this set where its files ship (a card: pls run --on <host:gpu>"
          f" --wt {wt}, or a SLURM tree), or, when the job does not read them, pass"
          f" --wt-data-ok {' '.join(rel for rel, _ in differ)} to train on the published copy.", file=sys.stderr)  # fmt: skip
    sys.exit(1)


if __name__ == "__main__":
    main()
