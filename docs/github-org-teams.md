# GitHub org + Teams setup (admin step — run these yourself)

The reorg splits into per-project repos; group them in one org and grant access via **Teams** so
onboarding is a single action and scoped per project. Repos use org **`Creampelt`** (adjust if you
make a dedicated org). Run with the `gh` CLI (authenticated as an org admin).

## 1. Create the repos (from the local carves)
```bash
# from each local repo dir (worktrees/carve/<name> or the core worktree)
gh repo create Creampelt/hcrl_isaaclab --private --source=. --remote=origin --push
gh repo create Creampelt/umrl_tasks    --private --source=. --remote=origin --push
gh repo create Creampelt/ssti_tasks    --private --source=. --remote=origin --push
gh repo create Creampelt/ssti_robots   --private --source=. --remote=origin --push
# hcrl_robots + robot_rl already exist
```

## 2. Teams (per project; one membership grants exactly the needed repos)
```bash
ORG=Creampelt
gh api orgs/$ORG/teams -f name='infra' -f privacy='closed'
gh api orgs/$ORG/teams -f name='ssti'  -f privacy='closed'
gh api orgs/$ORG/teams -f name='umrl'  -f privacy='closed'

# infra: read the shared core + RL + manager
for r in hcrl_isaaclab robot_rl hcrl_isaac_manager hcrl_robots; do
  gh api -X PUT orgs/$ORG/teams/infra/repos/$ORG/$r -f permission=pull; done

# ssti: write ssti repos, read core
for r in ssti_tasks ssti_robots; do gh api -X PUT orgs/$ORG/teams/ssti/repos/$ORG/$r -f permission=push; done
for r in hcrl_isaaclab robot_rl hcrl_robots; do gh api -X PUT orgs/$ORG/teams/ssti/repos/$ORG/$r -f permission=pull; done

# umrl: write umrl repo, read core
gh api -X PUT orgs/$ORG/teams/umrl/repos/$ORG/umrl_tasks -f permission=push
for r in hcrl_isaaclab robot_rl hcrl_robots; do gh api -X PUT orgs/$ORG/teams/umrl/repos/$ORG/$r -f permission=pull; done
```

## 3. Onboard / offboard (the "one thing")
```bash
gh api -X PUT orgs/Creampelt/teams/ssti/memberships/<user>   # add to ssti -> gets all & only ssti repos
gh api -X DELETE orgs/Creampelt/teams/ssti/memberships/<user>
```
The team's repo set mirrors `workspace.yaml`'s project closure, so adding a `_tasks` repo to a project
should add it to that team's grant too.
