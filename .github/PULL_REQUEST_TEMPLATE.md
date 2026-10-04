## Summary

<!-- What changed and why. -->

## Checklist

- [ ] Changes fit within this repo's scope — workspace tooling, Ray/cluster, and docs. Task or env
      code belongs in `hcrl_isaaclab` / `*_tasks`.
- [ ] If `workspace.yaml` / `resolve_workspace.py` / the `justfile` changed, `just resolve` (and
      `just setup` if touched) still produces a working workspace.
- [ ] No secrets or cluster-specific absolute paths committed.
- [ ] Added or updated tests under `scripts/<area>/tests/` for the change (`just test-scripts` passes locally; CI
      runs it). If a change cannot be tested without a cluster, leave this unticked and say why.
