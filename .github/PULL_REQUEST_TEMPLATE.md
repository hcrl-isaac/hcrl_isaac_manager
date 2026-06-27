## Summary

<!-- What does this change and why. One or two sentences. -->

## Scope check (manager)

`hcrl_isaac_manager` is workspace orchestration: dependency resolution, the `justfile`, Ray /
cluster tooling, and docs. It does not hold task or env code.

- [ ] Change is tooling/infra/docs — task or env code goes in `hcrl_isaaclab` / `*_tasks`.
- [ ] If `workspace.yaml` / `resolve_workspace.py` / `dependencies.yaml` handling changed,
      `just resolve` still produces a valid flat `gitman.yml`.
- [ ] If `justfile` / `setup` changed, `just setup` still yields a working env (pip-or-source
      IsaacLab, flat `resources/` layout).
- [ ] No secrets/credentials or cluster-specific absolute paths committed.

## Testing

<!-- Commands run + result. -->

- [ ] Relevant recipe exercised (e.g. `just resolve`, `just new-tasks <name>`, `scripts/ray.sh list`).

## Notes

<!-- Related core / *_tasks PRs, docs touched, follow-ups. -->
