---
name: convention-review
description: In-depth per-branch convention review across the workspace repos — finds obsolete/superseded code, audits comment/docstring style against the project charter, and flags design-decision issues (naming vs scope, base/subclass splits). Reviews each branch diff against its base in stack order so lower PRs are cleaned before the ones stacked on them.
---

# Convention review

Reviews one or more branches (default: every unmerged branch in
`resources/{hcrl_isaaclab,robot_rl,hhlm_tasks,ssti_tasks,hcrl_sim2real}`)
against the project charter, then applies mechanical fixes and reports design findings for sign-off.
Invoke as `/convention-review [repo:branch[:base] ...]`; with no args, discover targets via
`git branch -vv` + open PRs (`gh pr list`) and order them bottom-of-stack first (`gh stack` order).

## The charter (review lenses)

1. **Comments** — 1 line ideal, 2 hard max; only the non-obvious *why*. Never: narration of the next
   line, historical war stories, review/measurement attributions, or references to internal milestones
   (version tags like "v2"/"v3", spec-section refs like "§4.2", decision dates, people's names, W&B run
   ids as provenance, plan identifiers like "SEAM 1"/"WI-7").
2. **Docstrings** — one-line summary of what THIS unit does; optional `Args:`/`Returns:` with entries
   ≤2 lines. Never: mechanism narration, other components' behavior, historical defect stories, or
   obvious use-case explanations (e.g. don't explain that `"cpu"` storage fits a bigger buffer in RAM).
   A single ≤2-line constraint sentence between summary and Args is allowed when a caller MUST know it
   (data contract, ordering requirement, legality condition). Script module docstrings double as
   `--help`: purpose + usage + outputs, ≤12 lines, no rationale stories.
3. **Obsolete/superseded** — dead flags, unused cfgs/envs/tools, code superseded elsewhere on the same
   branch, launch wrappers for finished runs, references from committed code into gitignored/uncommitted
   docs. Verify "unused" claims by grepping consumers before reporting.
4. **Design decisions** — names must match scope (a generic name must not carry feature-specific
   content: rename it to the feature, or move the specific fields into a subclass); shared knobs belong
   in a base class with feature subclasses; API placement (does this helper belong in this package?);
   artifact/file/task-id names containing versions are fine — prose is not.

## Procedure

1. **Targets**: build the (repo, branch, base) list; review in parallel, APPLY bottom-up.
2. **Review**: one agent per target reading via git plumbing only — `git diff <base>...<branch>`,
   `git show <branch>:<file>`. Agents must NEVER checkout/switch/stash in the main trees (live editable
   installs). Findings are structured: `{repo, branch, file, locator, category: obsolete|style|design,
   summary, suggested_fix, confidence}`. Review several targets with parallel subagents.
3. **Verify**: adversarially verify `obsolete` and `design` findings (a skeptic agent tries to refute:
   is it really unused? is the rename actually scope-correct?). Style findings are judged directly.
4. **Apply**: mechanical fixes (style, verified-dead code) are committed per branch, bottom-of-stack
   first, then propagated upward (merge the base branch into stacked branches if they must contain the
   fix). Run per-repo checks before each commit: `uvx ruff@<rev from .pre-commit-config.yaml> check` +
   `format`, robot_rl `pytest tests`, and the task repos' CPU smoke (`pls test <repo>`).
5. **Report**: design findings + anything not auto-applied go to the user as a decision list. Never
   push or edit PRs without being asked; no Claude/AI trailers in commits or PR bodies.
