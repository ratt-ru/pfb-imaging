# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues on `ratt-ru/pfb-imaging`.

**`gh issue view` and `gh issue edit` are broken on this machine** (the distro `gh` 2.46.0 asks
GraphQL for the sunset `projectCards` field; see `CLAUDE.md`). Use `gh api` for every read and
every edit. `gh issue create`, `gh issue comment` and `gh issue close` are unaffected.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body-file body.md`. Write the body to a file so markdown survives shell quoting.
- **Read an issue**: `gh api repos/ratt-ru/pfb-imaging/issues/<n> --jq '{number,title,state,body,labels:[.labels[].name]}'`, and for the comments `gh api repos/ratt-ru/pfb-imaging/issues/<n>/comments --jq '[.[].body]'`.
- **List issues**: `gh api 'repos/ratt-ru/pfb-imaging/issues?state=open&labels=<label>&per_page=100' --jq '[.[] | select(.pull_request | not) | {number, title, labels: [.labels[].name]}]'`. The REST endpoint also returns PRs, so the `select` drops them.
- **Make an issue a sub-issue of a parent**: `gh api --method POST repos/ratt-ru/pfb-imaging/issues/<parent>/sub_issues -F sub_issue_id=<child-db-id>`. Get the database id with `gh api repos/ratt-ru/pfb-imaging/issues/<child> --jq .id`. Do not use the `#number` or the `node_id`. Without sub-issues, put `Part of #<parent>` at the top of the child body.
- **Comment on an issue**: `gh issue comment <number> --body-file comment.md`
- **Apply / remove labels**: `gh api -X POST repos/ratt-ru/pfb-imaging/issues/<n>/labels -f 'labels[]=<label>'` / `gh api -X DELETE repos/ratt-ru/pfb-imaging/issues/<n>/labels/<label>`
- **Edit title/body**: `gh api -X PATCH repos/ratt-ru/pfb-imaging/issues/<n> --input body.json`
- **Close**: `gh issue close <number> --comment "..."`

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues:

- **Read a PR**: `gh pr view <number> --comments` and `gh pr diff <number>` for the diff.
- **List external PRs for triage**: `gh api --paginate 'repos/ratt-ru/pfb-imaging/pulls?state=open' --jq '.[] | select(.author_association | IN("OWNER","MEMBER","COLLABORATOR") | not) | {number, title, author: .user.login, author_association, labels: [.labels[].name]}'`.
- **Comment / label / close**: `gh pr comment`, the `gh api .../issues/<n>/labels` calls above (PRs share the issues label endpoint; `gh pr edit` is broken here), `gh pr close`.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either: `gh api repos/ratt-ru/pfb-imaging/issues/42 --jq .pull_request` is non-null for a PR.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Read it as in **Read an issue** above.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (see **Make an issue a sub-issue of a parent**). Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's native issue dependencies. Add an edge with `gh api --method POST repos/ratt-ru/pfb-imaging/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>` (database id as above). `issue_dependencies_summary.blocked_by` counts open blockers only. Fallback: a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children (`gh api repos/ratt-ru/pfb-imaging/issues/<map>/sub_issues`), drop any with an open blocker or an assignee; first in map order wins.
- **Claim**: `gh api -X POST repos/ratt-ru/pfb-imaging/issues/<n>/assignees -f 'assignees[]=<login>'`, the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.
