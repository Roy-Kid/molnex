---
name: mn-ship
description: Release gate — commit | push | merge. Runs format + lint + test parity checks before handing off to git. Read-only w.r.t. source; may execute git on user confirmation.
argument-hint: "commit | push | merge"
user-invocable: true
---

# MolNex Ship

Read `CLAUDE.md` for MolNex conventions.

## Procedure

Dispatch on `$ARGUMENTS`:

### `commit` — pre-commit gate

1. `git status` and `git diff --stat` — summarise the change set.
2. Run `pre-commit run --all-files` if configured, else `ruff check`
   and `ruff format --check`.
3. Fast tests: `pytest -x -q tests/ -k "not slow and not bench"`.
4. If all green, propose a conventional-commit message (`feat|fix|refactor|
   docs|test|chore|perf|ci: <subject>`) derived from the diff and ask the
   user to approve before committing.

### `push` — pre-push gate

1. Assume `commit` has already passed.
2. Full test suite: `pytest tests/ -v`.
3. `ruff check` + `ruff format --check` (must be clean).
4. Verify branch is up-to-date with its remote; propose `git push -u`
   and wait for user confirmation before executing.

### `merge` — CI-parity gate

1. Full `pytest tests/ -v`, optional `--cov`.
2. Fan out `/mn-review` on the diff against `master`.
3. Architecture post-check: `/mn-arch`.
4. Report a green/red verdict. Do **not** run the actual merge — that is
   the user's decision on the PR platform.

## Output

- Command outputs (truncated to pass/fail lines).
- One-line verdict per stage.
- Proposed commit message (commit mode) or proposed push command (push mode).

## Rules

- Never `push --force`, never `--no-verify`, never `--amend` without an
  explicit user request — CLAUDE.md workflow rules apply.
- Never merge on the user's behalf. The human makes the merge call.
