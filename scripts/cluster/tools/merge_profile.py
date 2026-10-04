#!/usr/bin/env python3
"""Merge a regenerated cluster profile with its previous version so `just cluster add --update` keeps hand edits."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Keys/flags that come from the prompts (whose defaults were the old values): the new file wins for these.
PROMPTED_KEYS = {"CLUSTER_LOGIN", "CLUSTER_ISAACLAB_DIR", "CLUSTER_SIF_PATH"}
PROMPTED_FLAGS = {"-p", "-A", "-n", "--cpus-per-task", "--time", "--mail-user"}
KEY_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)=")
# Aliases and mutually exclusive flags share one key, so an old `--mem=0` replaces the template's `--mem-per-cpu`.
FLAG_GROUPS = {
    "--partition": "-p",
    "--account": "-A",
    "--ntasks": "-n",
    "--nodes": "-N",
    "--mem-per-cpu": "--mem",
    "--mem-per-gpu": "--mem",
    "--output": "-o",
    "--error": "-e",
}


def sbatch_flag(line: str) -> str | None:
    parts = line.split(None, 2)
    if len(parts) < 2 or parts[0] != "#SBATCH":
        return None
    flag = parts[1].split("=", 1)[0]
    return FLAG_GROUPS.get(flag, flag)


def module_name(line: str) -> str | None:
    parts = line.split()
    return parts[2].split("/", 1)[0] if len(parts) >= 3 and parts[:2] == ["module", "load"] else None


def merge_env(old: str, new: str) -> str:
    """Keep the old value of every non-prompted key and append keys the template doesn't emit.

    Args:
        old: Previous .env.cluster text.
        new: Regenerated .env.cluster text.

    Returns:
        The merged .env.cluster text.
    """
    old_lines = {m.group(1): line for line in old.splitlines() if (m := KEY_RE.match(line))}
    out, seen = [], set()
    for line in new.splitlines():
        m = KEY_RE.match(line)
        if m:
            seen.add(m.group(1))
            if m.group(1) in old_lines and m.group(1) not in PROMPTED_KEYS:
                line = old_lines[m.group(1)]
        out.append(line)
    out += [line for key, line in old_lines.items() if key not in seen]
    return "\n".join(out) + "\n"


def merge_submit(old: str, new: str) -> tuple[str, list[str]]:
    """Keep old #SBATCH lines for non-prompted flags, carry over flags and module lines the template lacks.

    Args:
        old: Previous submit_job_slurm.sh text.
        new: Regenerated submit_job_slurm.sh text.

    Returns:
        The merged text and the old lines that could not be carried over.
    """
    old_lines = old.splitlines()
    old_flags = {f: line for line in old_lines if (f := sbatch_flag(line))}
    old_mods = {m: line for line in old_lines if (m := module_name(line))}
    out, new_flags, new_mods = [], set(), set()
    for line in new.splitlines():
        flag, mod = sbatch_flag(line), module_name(line)
        if flag:
            new_flags.add(flag)
            if flag in old_flags and flag not in PROMPTED_FLAGS:
                line = old_flags[flag]
        elif mod:
            new_mods.add(mod)
            line = old_mods.get(mod, line)
        out.append(line)
    extra_flags = [line for f, line in old_flags.items() if f not in new_flags]
    extra_mods = [line for m, line in old_mods.items() if m not in new_mods]
    if extra_flags:
        last = max(i for i, line in enumerate(out) if sbatch_flag(line))
        out[last + 1 : last + 1] = extra_flags
    if extra_mods:
        mod_idx = [i for i, line in enumerate(out) if module_name(line)]
        at = mod_idx[-1] + 1 if mod_idx else next(i for i, line in enumerate(out) if line.startswith("cat <<"))
        out[at:at] = extra_mods
    merged = set(out)
    lost = [line for line in old_lines if line.strip() and not line.startswith("#") and line not in merged]
    return "\n".join(out) + "\n", lost


def main() -> None:
    """Usage: merge_profile.py <backup_dir> <profile_dir>; rewrites the profile's two files in place."""
    backup, profile = Path(sys.argv[1]), Path(sys.argv[2])
    env_old, job_old = backup / ".env.cluster", backup / "submit_job_slurm.sh"
    env_new, job_new = profile / ".env.cluster", profile / "submit_job_slurm.sh"
    if env_old.is_file():
        env_new.write_text(merge_env(env_old.read_text(), env_new.read_text()))
    if job_old.is_file():
        text, lost = merge_submit(job_old.read_text(), job_new.read_text())
        job_new.write_text(text)
        for line in lost:
            print(f"[WARN] not carried over from the old submit_job_slurm.sh: {line}")


if __name__ == "__main__":
    main()
