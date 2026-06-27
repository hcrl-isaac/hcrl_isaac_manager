# GitHub org + Teams setup (admin step — run these yourself)

The reorg splits into per-project repos; group them in one org and grant access via **Teams** so
onboarding is a single action and scoped per project. Org: **`hcrl-isaac`** (the `Creampelt` name is
a personal user account, so the org must use a distinct name).

## 0. Prerequisites (one-time, manual)
Org creation is **not** scriptable — do it in the browser, then grant `gh` the org scope:
```bash
# 1. Create the org in a browser: https://github.com/account/organizations/new  (name: hcrl-isaac)
# 2. Re-auth gh with admin:org (current token only has repo/read:org):
gh auth refresh -h github.com -s admin:org
```

## 1. Move the existing repos into the org
These three already exist under the `Creampelt` user. Transfer preserves history/issues/stars
(remotes update afterward — see step 5):
```bash
for r in hcrl_isaaclab robot_rl hcrl_robots; do
  gh api -X POST repos/Creampelt/$r/transfer -f new_owner=hcrl-isaac; done
```

## 2. Create the new repos (from the local carves)
```bash
# from each local repo dir (worktrees/carve/<name> or the core worktree)
gh repo create hcrl-isaac/umrl_tasks  --private --source=. --remote=origin --push
gh repo create hcrl-isaac/ssti_tasks  --private --source=. --remote=origin --push
gh repo create hcrl-isaac/ssti_robots --private --source=. --remote=origin --push
```

## 3. Teams (per project; one membership grants exactly the needed repos)
```bash
ORG=hcrl-isaac
gh api orgs/$ORG/teams -f name='infra' -f privacy='closed'
gh api orgs/$ORG/teams -f name='ssti'  -f privacy='closed'
gh api orgs/$ORG/teams -f name='umrl'  -f privacy='closed'

# infra: read the shared core + RL + manager
for r in hcrl_isaaclab robot_rl hcrl_robots; do
  gh api -X PUT orgs/$ORG/teams/infra/repos/$ORG/$r -f permission=pull; done

# ssti: write ssti repos, read core
for r in ssti_tasks ssti_robots; do gh api -X PUT orgs/$ORG/teams/ssti/repos/$ORG/$r -f permission=push; done
for r in hcrl_isaaclab robot_rl hcrl_robots; do gh api -X PUT orgs/$ORG/teams/ssti/repos/$ORG/$r -f permission=pull; done

# umrl: write umrl repo, read core
gh api -X PUT orgs/$ORG/teams/umrl/repos/$ORG/umrl_tasks -f permission=push
for r in hcrl_isaaclab robot_rl hcrl_robots; do gh api -X PUT orgs/$ORG/teams/umrl/repos/$ORG/$r -f permission=pull; done
```

## 4. Onboard / offboard (the "one thing")
```bash
gh api -X PUT orgs/hcrl-isaac/teams/ssti/memberships/<user>   # add to ssti -> gets all & only ssti repos
gh api -X DELETE orgs/hcrl-isaac/teams/ssti/memberships/<user>
```
The team's repo set mirrors `workspace.yaml`'s project closure, so adding a `_tasks` repo to a project
should add it to that team's grant too.

## 5. Repoint local remotes
After transfers/creates, update any local clones/worktrees and regenerate the lockfile:
```bash
# existing clones: git remote set-url origin git@github.com:hcrl-isaac/<repo>.git
just resolve        # regenerates gitman.yml from workspace.yaml (org: hcrl-isaac)
```
