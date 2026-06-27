# hcrl Isaac Lab — Repository Reorganization Design Spec

Status: **draft for review** · Owner: Emily · Last updated: 2026-06-26

## 1. Motivation & problems with today's layout

Today everything lives in **one repo** (`hcrl_isaaclab`, mounted at
`hcrl_isaac_manager/resources/IsaacLab/source/hcrl_isaaclab`) with a **branch lineage** that
encodes hierarchy:

```
main (infra/scripts only)
  └─ ssti ─ ssti-mcp, ssti-rough, …
  └─ umrl ─ umrl-fri, umrl-dex, …
```

Infra reaches a dev branch only by a **merge chain** (`main → ssti → ssti-mcp`). Pain points:

1. **Infra propagation is manual and N-way.** A one-line script/infra fix must be merged through
   every project branch and then every dev branch. (We just lived this getting `video_logger.py`
   into ssti-mcp.)
2. **Hierarchy is invisible.** A DAG of merges doesn't render as a tree; "what depends on what" is
   tribal knowledge.
3. **History is tangled.** Infra commits and research commits interleave on every branch.
4. **Coupling forces divergence.** Because project task code and shared infra share a branch, any
   project-local edit to a shared file blocks/complicates infra merges.

**Root cause:** we use *git merge* to do *dependency propagation*. Branches can't express "these N
projects all consume one evolving core."

## 2. Goals

- Infra updates propagate by **bumping one dependency pointer per project**, not a merge chain.
- **Explicit, visible hierarchy** (a manifest is the tree).
- Projects (`ssti`, `umrl`) are **isolated** and **independently shareable**.
- **Multiple projects coexist in one workspace** (a single dev may work on ssti *and* umrl at once).
- Clean separation of **infra/core vs project task layout vs robot/asset packages**.
- Support **pip-installed IsaacLab** *or* **source IsaacLab**, selectable per workspace.
- A **generator** to scaffold new `*_tasks` repos.
- **Decentralized dependencies**: each repo declares its own direct deps; shared deps are
  **deduplicated** (one checkout per workspace).
- Set up a later, clean **Ray container / HPC deployment** restructure (decouple from IsaacLab
  scripts; simplify the `.sif` build).

## 3. Target repository topology

GitHub org **`hcrl`** (see §10 for teams/permissions):

| Repo | Role | Branching |
| --- | --- | --- |
| `hcrl_isaac_manager` | **Meta-repo**: workspace manifests, env build, dep resolver, `new-tasks` | trunk + feature branches |
| `IsaacLab` (fork) | Optional, **source mode only** | tracks upstream + local patches |
| `hcrl_isaaclab` | **Core extension**: scripts (`train`/`play`/`video_logger`/`bench`/`ray`), `envs/`, `managers/`, shared `mdp/`, `utils/`, **reference tasks** (locomotion + template) | **single `main`** + short-lived feature branches |
| `robot_rl` | RL algorithms (already separate) | single `main` |
| `ssti_tasks` | ssti task extension; deps: `hcrl_robots` (shared), `ssti_robots` (project-local) | dev branches live **here** (`mcp`, `rough`, …) |
| `umrl_tasks` | umrl task extension; deps: `hcrl_robots` (shared) | dev branches live **here** (`fri`, `dex`, …) |
| `hcrl_robots` | **Shared** robot/asset package — used by **both** `ssti_tasks` and `umrl_tasks` (the canonical dedup case, §7) | single `main` |
| `ssti_robots` | **Project-local** robot/asset package (ssti only) | single `main` |

Key property: **`hcrl_isaaclab` is a leaf in the dependency DAG** (depends only on IsaacLab +
robot packages), and every `*_tasks` repo depends on it. Infra changes land once, in core.

### Why the branch chain disappears

Infra change → commit to `hcrl_isaaclab` `main`. Each project advances its **`hcrl_isaaclab`
dependency ref** when ready: one commit, conflict-free (projects never patch core), and the manager
can sweep the bump across all projects. There is no `main → ssti → ssti-mcp` anymore.

## 4. Core (`hcrl_isaaclab`) contents and the task-config refactor

`hcrl_isaaclab` holds **infra + a small reference task set whose job is to exercise infra** (CI/smoke
target), not to do research:

```
hcrl_isaaclab/
  scripts/            train.py play.py video_logger.py bench.py ray/ utils/ export.py …
  hcrl_isaaclab/
    envs/             custom ManagerBasedRLEnv(+Cfg)
    managers/         custom ObservationManager/Cfg, RewardManager/Cfg, …
    mdp/              shared MDP terms (rewards/obs/events/terminations) + high-level/module mdp framework
    utils/            artifacts (ensure_for_cfg), articulation cache, …
    tasks_registry.py NEW — namespaced register/resolve (§5)
    tasks/
      locomotion/     biped + quadruped BASE env cfgs + reference robots (G1, Go1)
      template/       trivial smoke task the generator clones
```

**Explicitly core (shared infra), not project code:** the custom `ManagerBasedRLEnv`, the custom
managers (`ObservationManager`/`RewardManager`/…), and the **high-level / module MDP functions** (the
hierarchical-control machinery used by e.g. MCP — HL command terms, primitive-module actions) all
live in core (`hcrl_isaaclab.envs` / `managers` / `mdp`). They are reusable infrastructure, not task
content. Only *project-specific* MDP terms stay with a project under
`*_tasks/.../tasks/<domain>/mdp/`; a term graduates to core the moment a second project would want it.

### 4.1 The task-config refactor is the load-bearing seam (moves into core)

The refactor already implemented — `tasks/<task>/<task>_env_cfg.py` defining a **parent task config**
with a `configure_env(robot_cfg, …)` method, plus thin per-robot `task_cfg.py`/`ppo_cfg.py`
overrides — is exactly what makes a clean core/project split possible. Mapping:

- **Parent (base) task configs live in core.** e.g. `LocomotionEnvCfg` + `configure_env(robot_cfg,
  …)`, the **biped** base (modeled on G1) and **quadruped** base (modeled on Go1, gait masks), and
  the shared `*PPORunnerCfg` bases.
- **Reference-robot configs live in core** (G1, Go1) so locomotion is runnable as the infra smoke
  test.
- **Project robots + task variants live in `*_tasks`** as *thin* subclasses calling the core base's
  `configure_env(<project robot cfg>)`. Examples:
  - umrl T1 biped velocity → `umrl_tasks/.../t1/task_cfg.py` calls core biped `configure_env(T1_CFG)`.
  - ssti crab tasks → `ssti_tasks/.../crab/…` over core bases, with crab assets from `ssti_robots`.
- **Robot cfgs/USDs live in the robots packages** (`hcrl_robots`/`ssti_robots`/`umrl_robots`), not in
  task repos, so multiple task repos can share one robot definition (deduped, §7).

Consequence: a project task is "base (core) + robot cfg (robots pkg) + a few overrides (project)."
Updating a base reward/obs in core propagates to all projects via the core dep bump — **no
re-implementation, no merge**. This is the single biggest payoff of the refactor in the new layout.

**Migration note:** the "move the task refactor here" work = carve the **base** cfgs into core and
leave **per-robot/project** cfgs in the respective `*_tasks`, keeping every gym id stable (§9). What
counts as "shared mdp" (→ core) vs "project mdp" (→ `*_tasks`): a term goes to core only if it is (or
should be) reusable across projects; otherwise it stays with the project under
`*_tasks/.../tasks/<domain>/mdp/`.

## 5. Multi-project coexistence: namespaced tasks + `--source`

Both `ssti_tasks` and `umrl_tasks` may be installed in one env and register into the **same gym
registry**. Requirement: bare task ids work when unique; on a name collision, an extra `--source`
selects the project. `--source` is **redundant on non-overlapping tasks**.

### 5.1 `tasks_registry` (new, in core)

```python
# hcrl_isaaclab/tasks_registry.py  (sketch)
def register_task(source: str, id: str, **gym_kwargs) -> None:
    """Register under the gym namespace `source/id` and record bare-id -> {sources}."""
    gym.register(id=f"{source}/{id}", **gym_kwargs)   # collision-free at the gym layer
    _BARE_TO_SOURCES.setdefault(id, set()).add(source)

def resolve(id: str, source: str | None = None) -> str:
    """Map a (possibly bare) task id (+ optional source) to the concrete gym id."""
    if source:
        return f"{source}/{id}"
    if "/" in id:
        return id                              # already namespaced
    srcs = _BARE_TO_SOURCES.get(id)
    if not srcs:
        raise KeyError(f"unknown task {id!r}")
    if len(srcs) > 1:
        raise ValueError(f"task {id!r} is ambiguous (registered by {sorted(srcs)}); pass --source")
    return f"{next(iter(srcs))}/{id}"
```

- gym actually holds `ssti/Crab-MCP-v0`, `umrl/<…>` (gym supports `Namespace/Name-vN`).
- `--task Crab-MCP-v0` with no source → used directly if unique; **errors with the offending
  sources if ambiguous**.
- `--source ssti --task Crab-MCP-v0` → `ssti/Crab-MCP-v0`.

### 5.2 Script changes — one shared arg helper for the common boilerplate

Today every entry script re-declares the same `--task`/`--video`/`--num_envs`/… args and repeats the
post-parse logic. Centralize this the same way `cli_args.add_rsl_rl_args` / `update_rsl_rl_cfg`
already centralize the RL args — a **paired helper** in `cli_args` (core):

```python
# cli_args.py (core)
def add_isaaclab_args(parser) -> None:
    """Add the args common to every entry script: --task, --source, --video[_*], --num_envs, …."""
    parser.add_argument("--task", ...); parser.add_argument("--source", default=None, ...)
    parser.add_argument("--video", action="store_true", ...); parser.add_argument("--video_num_envs", ...)
    # …video_length/video_interval/num_envs/async/no_video…

def resolve_isaaclab_args(args) -> None:
    """Resolve those args in-place: namespaced task id, video strategy, camera enable, etc."""
    args.task = tasks_registry.resolve(args.task, args.source)   # bare→namespaced, or error if ambiguous
    # …apply the --video/--async/--no_video strategy + enable_cameras, exactly once, here…
```

`train.py` / `play.py` / `video_logger.py` / `bench.py` then just call `add_isaaclab_args(parser)`
before parsing and `resolve_isaaclab_args(args)` after — `--source` resolution and the video strategy
live in **one** place instead of being copy-pasted per script. (Name negotiable:
`resolve_isaaclab_args` for the broad "Isaac-Lab-script common args" framing, or `resolve_task_args`
if scoped to task/source/video. Recommend the broad `*_isaaclab_args` pair since it also owns the
video strategy and camera-enable, not just task resolution.)

Optional: a workspace **default source** (env var / config) consumed inside `resolve_isaaclab_args`,
so a dev mostly on ssti rarely types `--source`.

### 5.3 `*_tasks` registration convention

Each `*_tasks` package registers via `register_task(source=<derived from package>, id=…, …)` instead
of raw `gym.register`. Source auto-derives from the package name (`ssti_tasks` → `ssti`). The
generator (§8) bakes this in; existing tasks migrate mechanically. Core/reference tasks register
under `hcrl` (or stay unnamespaced).

## 6. Manager workspace layout + pip-or-source IsaacLab

Flatten `resources/` to siblings (no nesting under `IsaacLab/source/`):

```
hcrl_isaac_manager/resources/
  IsaacLab/        # present only in source mode
  hcrl_isaaclab/   # core
  robot_rl/
  ssti_tasks/      # whichever projects this workspace includes
  umrl_tasks/
  hcrl_robots/  ssti_robots/  umrl_robots/   # hoisted, deduped (§7)
```

This works because **Isaac Lab extensions are discovered via the Python env (installed packages /
entry points), not by directory location.** Each repo is `pip install -e`'d into the env; placement
is free. (Putting `hcrl_isaaclab` under `IsaacLab/source/` today is incidental.)

**IsaacLab mode** is an env-build switch, because `hcrl_isaaclab`/`*_tasks` depend on `isaaclab*`
*by package name*:

- **pip mode (default):** `pip install isaacsim-* isaaclab isaaclab_rl isaaclab_tasks …` (pinned
  version); no `resources/IsaacLab`. Note this also pulls the **Isaac Sim runtime** via pip
  `isaacsim` (that's the real weight).
- **source mode:** clone `resources/IsaacLab`, editable-install its `source/*` (for patching
  IsaacLab itself).

Manager config flag, e.g.:

```yaml
isaaclab:
  source: false          # true → use resources/IsaacLab editable
  version: "x.y.z"       # pin for pip mode
```

Downstream (env build, Ray image, HPC) is identical regardless of mode. Pin the IsaacLab version the
`managers/` refactor + gym-registration assume.

## 7. Dependency management: per-repo manifests + hoist/dedup

Move from "manager knows every dep" to **each repo declares its direct deps; manager resolves the
graph and hoists**.

### 7.1 Manifest (per repo)

`dependencies.yaml` (or `[tool.hcrl.deps]` in `pyproject.toml`):

```yaml
# ssti_tasks/dependencies.yaml
deps:
  - name: hcrl_isaaclab   # core
    git: git@github.com:hcrl/hcrl_isaaclab.git
    ref: main
  - name: ssti_robots
    git: git@github.com:hcrl/ssti_robots.git
    ref: main
```

### 7.2 Resolver (in the manager)

1. Read the workspace manifest (which projects are included) + each repo's `dependencies.yaml`.
2. Walk **transitively**; build a flat set **keyed by `name`**.
3. **Dedup**: one checkout per unique name under `resources/<name>`, editable-installed once. Shared
   `hcrl_robots` appears once even if `ssti_tasks` and `umrl_tasks` both need it.
4. **Conflict policy**: same `name`, different `ref` in one workspace → **error with both refs +
   require an explicit workspace override** (one ref per name per workspace). Cross-project conflicts
   are rare because workspaces usually include one project, but multi-project workspaces must
   reconcile shared deps.
5. Install order respects the DAG (robots → core → tasks).

### 7.3 Build vs buy — west vs gitman vs custom

The deciding axis is **transitive dedup/hoisting across multiple consumers** (the
`hcrl_robots`-shared-by-both case):

- **west** (Zephyr meta-tool): manifest repo + `import:` resolves *all* manifests (incl. transitive)
  into a **single flat, name-keyed project list** → shared deps collapse to **one** checkout, with a
  single place to detect ref conflicts. Matches the requirement by construction. Cost: Zephyr baggage
  (build/flash extensions, import precedence/allowlist semantics) you don't need.
- **gitman**: simple, Python-native, great for "pin these repos at these revs." But its transitive
  model **vendors each consumer's nested deps under that consumer** (recurse-and-clone, not hoist) →
  `ssti_tasks` and `umrl_tasks` would each get their **own** `hcrl_robots` copy, with no cross-consumer
  single-version reconciliation. That directly conflicts with the dedup requirement. (Confirm against
  current gitman docs, but this nested-vendoring is the crux.)
- **vcstool** (`vcs import resources < workspace.repos`): dead-simple flat import, **no**
  transitive/dedup — compose the manifest yourself.

**Recommendation:** don't adopt west wholesale. Build a **thin resolver in the manager** that borrows
*west's model* (flat, name-keyed, transitive, dedup + conflict-error), using **gitman or plain `git`
as the fetch/checkout mechanism** underneath. You get hoisting semantics without Zephyr, the manager
keeps ownership of the workspace, and gitman still does the part it's good at (materializing a repo at
a rev). gitman-alone would only suffice if shared deps were rare or duplication were acceptable —
`hcrl_robots` shared by both projects rules that out.

### 7.4 `robot_rl` version coupling (surfaced 2026-06-26)

`robot_rl` is a **shared dep that different projects pin to different refs** — the textbook §7.2
conflict case, and it bites *core*:

- `hcrl_isaaclab/robot_rl/rl_cfg.py` (core) defines the RL **model config** classes, and those are
  **coupled to robot_rl's feature set**. The `ssti` line tracks a *richer* robot_rl (meta-RL / URL /
  off-policy: `MetaRlCfg`, `first_activation`, `RNN/TXL/CNN/Fuse` cfgs, `OffPolicyRunner`, URL env
  wrappers); the `ssti-mcp` line uses the *lean* `robot_rl@main` (plain on-policy PPO + `MLPModel`).
- Concretely: a blind `ssti→ssti-mcp` merge pulled ssti's `rl_cfg.py`, which `import`s `meta_rl_cfg`
  (absent on mcp) and emits `first_activation` into kwargs that `robot_rl@main`'s `MLPModel` rejects —
  crashing `train.py` at import. (Fixed on the mcp video branch by keeping the lean `rl_cfg.py` +
  making the umrl-only imports optional.)

Implication for this refactor: **core cannot be trivially robot_rl-version-agnostic.** Options to
decide (see §12):
1. **Core pins one `robot_rl`** and every project uses it (simplest; forces convergence).
2. **Feature-gate** the richer model cfgs in core (import-guard `meta_rl`/URL/off-policy; degrade to
   MLP when the installed robot_rl lacks them) — what the mcp branch does ad hoc; could be formalized.
3. **Move model cfgs out of core** (into robot_rl itself, or a per-project layer) so `rl_cfg.py`
   isn't a core file straddling two robot_rl feature sets.
The resolver's per-workspace single-ref rule (§7.2) still applies, but core's coupling means the
*choice of robot_rl ref* is a cross-cutting decision, not a free per-project pin.

## 8. `*_tasks` starter generator

Mirror IsaacLab's external-extension template, using **copier** (supports `copier update` to pull
template improvements into existing repos later — propagation at the scaffold layer).

- **Template lives in core** (`hcrl_isaaclab/templates/tasks_repo/`); **orchestration in the
  manager**: `manager new-tasks <name>` runs copier → produces `<name>_tasks` with:
  - `pyproject.toml` (deps: `hcrl_isaaclab`, `isaaclab`), `<name>_tasks/__init__.py` registering a
    sample task via `register_task` (namespaced by `<name>`),
  - a `tasks/<domain>/` skeleton with a template task subclassing a core base via `configure_env`,
  - inherited `ruff.toml` / `.pre-commit-config.yaml`, `dependencies.yaml`, `.gitignore` + LFS,
    README.
  - then registers the repo in the workspace manifest + adds it to `resources/`.

## 9. Migration plan (history-preserving)

1. **Carve task repos** with `git filter-repo`, applying the carve rule: for each project branch,
   a task dir moves to that project's `*_tasks` repo **iff the branch edited/used it** (diff the
   branch's `tasks/<domain>/` against `main`); any task dir that has **not diverged from `main` is
   removed** (inherited, unused). Concretely: ssti's edited domains (mcp/approach/traverse/reorient/…)
   → `ssti_tasks`; umrl's edited domains → `umrl_tasks`; unchanged-from-main dirs dropped. Extract
   robot/asset dirs → `hcrl_robots` (shared) / `ssti_robots` (project-local).
2. **Slim core**: keep `scripts/`, `envs/`, `managers/`, shared `mdp/`, `utils/`, and **base** task
   cfgs + reference robots (locomotion + template); remove project task dirs.
3. **Apply §5**: add `tasks_registry`, convert registrations to `register_task`, add `--source` to
   scripts. Keep every gym id stable (now namespaced; bare ids still resolve when unique).
4. **Add manifests** (`dependencies.yaml`) to every repo; implement/adopt the resolver (§7).
5. **Stand up the org + teams** (§10); push repos.
6. **Flatten `resources/`** + add the pip/source IsaacLab switch (§6).
7. **Validate**: per project, boot the sim and load every registered task old-vs-new; diff env cfgs
   (expect only namespace/path noise), exactly like the refactor verification we already did.

Sequencing: land §5 (`tasks_registry` + `--source`) and the core/base carve **first** (they're the
risky, behavior-touching parts and can be verified in the current monorepo), then do the physical
repo split, then the manager/resolver/IsaacLab-mode work.

## 10. GitHub org + per-project permissions

- One **Organization** (`hcrl`). Per-project sharing primitive = **Teams** (not an org per project).
- Teams: `infra` (read `hcrl_isaaclab`, `robot_rl`, `hcrl_isaac_manager`), `ssti`, `umrl`.
- Grant each team exactly the repos at the right level, e.g. `ssti` → `ssti_tasks` (write),
  `ssti_robots` (write), `hcrl_isaaclab` (read), `robot_rl` (read/write), `hcrl_isaac_manager`
  (read). Teams support per-repo read/triage/write/maintain/admin.
- **Onboarding = add to one team** → all and only that project's repos. Offboard = remove from team.
  Scoped (no access to other projects). The team's repo set should mirror the workspace manifest's
  repo set (could be generated from it).

## 11. Ray container / HPC (deferred — next phase)

The split enables a cleaner container story: once `hcrl_isaaclab` + `*_tasks` are flat editable
packages over pip-or-source IsaacLab, the `.sif`/docker image becomes "Isaac Sim base + `vcs`/manager
resolve + editable installs" (no code baked into `IsaacLab/source/`), and HPC runs `hcrl_isaaclab`'s
`train.py` directly instead of via `isaaclab.sh -p`. Full spec to follow.

## 12. Testing & CI strategy (future)

Goal: **an infra change can never silently break a task.** Two layers, sharing one harness:

- **Core (`hcrl_isaaclab`) base/unit tests** — fast, mostly sim-free: `tasks_registry` resolve/namespacing
  + `--source` ambiguity, `cli_args.add/resolve_isaaclab_args`, the namespaced `hydra_task_config`
  sanitization, the `configure_env` bases, `ensure_for_cfg`, the articulation cache. Plus a
  **reference-task smoke** (locomotion + template) that boots the sim and runs the full
  load→build→step→one-iteration path (what the reorg validation already does ad hoc).
- **Per-task smokes (in each `_tasks` repo)** — for *every* registered task in that repo: resolve the
  id, `parse_env_cfg` at tiny `num_envs`, build the env, step a few times (optionally one train
  iteration). 

**Shared harness in core.** A `smoke_task(task_id, num_envs=…, steps=…)` helper lives in
`hcrl_isaaclab` and is parametrized off `tasks_registry` (enumerate a source's registered ids), so
core's reference smoke and every `_tasks` repo's smokes call the same code — the seed already exists
in the reorg-validation scripts.

**CI wiring (the asymmetric part the user asked for):**

- **PR into a `_tasks` repo → run only that repo's smokes** against the *pinned* core (fast, scoped).
- **PR into `hcrl_isaaclab` (core) → run core unit tests + every `_tasks` repo's smokes**
  (downstream/consumer testing), because an infra change can break any task. Mechanically: a reusable
  GitHub Actions workflow that, on a core PR, matrices over the registered `_tasks` repos — checks out
  `core@PR` + each task repo (+ its robot deps via the §7 resolver) and runs that repo's smokes.

**Practical notes:** smokes need a **GPU runner** (Isaac Sim) — self-hosted GPU runner or a cluster
job; keep `num_envs`/steps tiny so the full matrix is minutes, not hours. The task-repo list for the
core matrix should come from the workspace manifest (§7), so adding a `_tasks` repo automatically
enrolls it in core's CI. The shared harness + `tasks_registry` enumeration are the enablers; this is
deferred but the design above should hold.

## 13. Decisions

Resolved (2026-06-26):
- **Shared deps**: `hcrl_robots` is **shared** by both `ssti_tasks` and `umrl_tasks`;
  `ssti_robots` is **project-local**. **`umrl_robots` is not needed** — umrl rides on `hcrl_robots`.
- **Core contents**: custom `ManagerBasedRLEnv`, custom managers, and high-level/module MDP functions
  are **core** (§4). **Reference task set = locomotion + template only.**
- **`robot_rl` coupling (§7.4)**: **core pins ONE `robot_rl`** (single ref). `robot_rl` is expected to
  *consolidate* features over time (the `umrl/` robot_rl branch will be **merged into `robot_rl`
  `main`** soon), so core's `rl_cfg.py` targets that one converging robot_rl. Unlike the task dirs,
  `robot_rl` does **not** fork per project. Local dev may check out a different `robot_rl` branch ad
  hoc; that's managed outside the pinned workspace.
- **`tasks_registry` / `--source` (§5)**: `--source` is a **CLI arg** (with a per-workspace default in
  config). **Core tasks ARE namespaced** (under `hcrl`). Consequence: **there is no `hcrl_tasks`
  repo** — the `hcrl` namespace belongs to core itself.
- **Carve rule (§9)**: a task goes to a project's `*_tasks` repo **iff that branch edited/used it**;
  any task that has **not diverged from `main` is removed** (inherited cruft, not real content).
- **Dependency tooling (§7.3)**: **custom resolver in the manager** borrowing west's model, gitman/
  `git` as fetch backend.
- **Common-arg helper (§5.2)**: paired `add_isaaclab_args` / `resolve_isaaclab_args` in `cli_args`.
- **Testing/CI (§12, future)**: two-layer smokes off a shared core harness — a core PR runs core unit
  tests **+ every `_tasks` repo's per-task smokes**; a `_tasks` PR runs only its own.
