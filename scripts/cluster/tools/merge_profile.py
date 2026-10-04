#!/usr/bin/env python3
"""Merge a regenerated cluster profile with its previous version so `just cluster add --update` keeps hand edits."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Keys/flags that come from the prompts (whose defaults were the old values): the new file wins for these.
PROMPTED_KEYS = {"CLUSTER_LOGIN", "CLUSTER_ISAACLAB_DIR", "CLUSTER_SIF_PATH"}
PROMPTED_FLAGS = {"-p", "-A", "-n", "-c", "-t", "--mail-user"}
# Keys derived from a prompt answer: they take the new value when that answer changed.
DERIVED_KEYS = {"CLUSTER_ISAAC_SIM_CACHE_DIR": "CLUSTER_SIF_PATH", "OMP_NUM_THREADS": "-c"}
KEY_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)=")
# Aliases and mutually exclusive flags share one key, so an old `--mem=0` replaces the template's `--mem-per-cpu`.
FLAG_GROUPS = {
    "--partition": "-p",
    "--account": "-A",
    "--ntasks": "-n",
    "--nodes": "-N",
    "--cpus-per-task": "-c",
    "--time": "-t",
    "--job-name": "-J",
    "--output": "-o",
    "--error": "-e",
    "--mem-per-cpu": "--mem",
    "--mem-per-gpu": "--mem",
}


def sbatch_flags(line: str) -> list[str]:
    """Canonical flags set by one `#SBATCH` line (several flags may share a line).

    Args:
        line: A line of a submit script.

    Returns:
        The flags, grouped by alias; empty for any other line.
    """
    parts = line.split()
    if not parts or parts[0] != "#SBATCH":
        return []
    flags = [t.split("=", 1)[0] for t in parts[1:] if t.startswith("-")]
    return [FLAG_GROUPS.get(f, f) for f in flags]


def flag_value(text: str, flag: str) -> str:
    """Value of a canonical `#SBATCH` flag in a submit script, whatever its spelling ('' if absent)."""
    for line in text.splitlines():
        parts = line.split()
        if parts[:1] != ["#SBATCH"]:
            continue
        for i, tok in enumerate(parts[1:], 1):
            name, eq, val = tok.partition("=")
            if FLAG_GROUPS.get(name, name) == flag:
                return val if eq else (parts[i + 1] if i + 1 < len(parts) else "")
    return ""


def module_name(line: str) -> str | None:
    """Module a `module load` line loads, without its version (None for any other line)."""
    parts = line.split()
    return parts[2].split("/", 1)[0] if len(parts) >= 3 and parts[:2] == ["module", "load"] else None


def _env_values(text: str) -> dict[str, str]:
    return {m.group(1): line for line in text.splitlines() if (m := KEY_RE.match(line))}


def merge_env(old: str, new: str, follow: set[str] = frozenset()) -> str:
    """Keep the old file's lines and order, taking the new value only for prompted and `follow` keys.

    Keys the template adds are appended with the comment lines above them in the template.

    Args:
        old: Previous .env.cluster text.
        new: Regenerated .env.cluster text.
        follow: Extra keys whose new value wins (derived from a changed prompt answer).

    Returns:
        The merged .env.cluster text.
    """
    new_values = _env_values(new)
    out = []
    for line in old.splitlines():
        m = KEY_RE.match(line)
        if m and m.group(1) in new_values and (m.group(1) in PROMPTED_KEYS or m.group(1) in follow):
            line = new_values[m.group(1)]
        out.append(line)
    old_keys = set(_env_values(old))
    new_lines = new.splitlines()
    for i, line in enumerate(new_lines):
        m = KEY_RE.match(line)
        if not m or m.group(1) in old_keys:
            continue
        start = i
        while start > 0 and new_lines[start - 1].startswith("#"):
            start -= 1
        out += new_lines[start : i + 1]
    return "\n".join(out) + "\n"


def merge_submit(old: str, new: str) -> tuple[str, list[str]]:
    """Keep old #SBATCH lines for non-prompted flags, carry over flags and module lines the template lacks.

    Args:
        old: Previous submit_job_slurm.sh text.
        new: Regenerated submit_job_slurm.sh text.

    Returns:
        The merged text and the old lines (commands, #SBATCH or module lines) that are not in it.
    """
    old_lines = old.splitlines()
    old_flags = {f: line for line in old_lines for f in sbatch_flags(line)}
    old_mods = {m: line for line in old_lines if (m := module_name(line))}
    out, new_flags, new_mods = [], set(), set()
    for line in new.splitlines():
        flags, mod = sbatch_flags(line), module_name(line)
        if flags:
            new_flags.update(flags)
            if flags[0] in old_flags and flags[0] not in PROMPTED_FLAGS:
                line = old_flags[flags[0]]
        elif mod:
            new_mods.add(mod)
            line = old_mods.get(mod, line)
        out.append(line)
    extra_flags = [ln for ln in old_lines if sbatch_flags(ln) and not set(sbatch_flags(ln)) & new_flags]
    extra_mods = [line for m, line in old_mods.items() if m not in new_mods]
    if extra_flags:
        last = max(i for i, line in enumerate(out) if sbatch_flags(line))
        out[last + 1 : last + 1] = list(dict.fromkeys(extra_flags))
    if extra_mods:
        mod_idx = [i for i, line in enumerate(out) if module_name(line)]
        at = mod_idx[-1] + 1 if mod_idx else next(i for i, line in enumerate(out) if line.startswith("cat <<"))
        out[at:at] = extra_mods
    merged = set(out)

    def dropped(line: str) -> bool:
        if not line.strip() or line in merged:
            return False
        flags = sbatch_flags(line)
        # a prompted flag's new value is the user's answer; any other flag missing from the output is lost
        return bool(set(flags) - PROMPTED_FLAGS) if flags else not line.startswith("#")

    return "\n".join(out) + "\n", [line for line in old_lines if dropped(line)]


def main() -> None:
    """Usage: merge_profile.py <backup_dir> <profile_dir> (rewrites the profile in place), or get <file> <flag>."""
    if sys.argv[1] == "get":
        path = Path(sys.argv[2])
        print(flag_value(path.read_text(), FLAG_GROUPS.get(sys.argv[3], sys.argv[3])) if path.is_file() else "")
        return
    backup, profile = Path(sys.argv[1]), Path(sys.argv[2])
    env_old, job_old = backup / ".env.cluster", backup / "submit_job_slurm.sh"
    env_new, job_new = profile / ".env.cluster", profile / "submit_job_slurm.sh"
    old_job = job_old.read_text() if job_old.is_file() else ""
    if env_old.is_file():
        old_env, new_env = env_old.read_text(), env_new.read_text()
        changed = {
            "CLUSTER_SIF_PATH": _env_values(old_env).get("CLUSTER_SIF_PATH")
            != _env_values(new_env).get("CLUSTER_SIF_PATH"),
            "-c": flag_value(old_job, "-c") != flag_value(job_new.read_text(), "-c"),
        }
        follow = {key for key, source in DERIVED_KEYS.items() if changed[source]}
        env_new.write_text(merge_env(old_env, new_env, follow))
    if old_job:
        text, lost = merge_submit(old_job, job_new.read_text())
        job_new.write_text(text)
        for line in lost:
            print(f"[WARN] not carried over from the old submit_job_slurm.sh: {line}")


if __name__ == "__main__":
    main()
