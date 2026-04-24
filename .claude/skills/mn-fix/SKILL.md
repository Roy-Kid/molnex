---
name: mn-fix
description: Minimal-diff bug fix — reproduce, localise, fix, add a regression test. Writes code and tests.
argument-hint: <bug description or test failure>
user-invocable: true
---

# MolNex Bug Fix

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. **Reproduce.** Write the minimum failing test or command from
   `$ARGUMENTS`. If it does not fail on `master`, ask the user for a
   clearer repro before continuing.
2. **Localise.** Read the failing path only — do not refactor, do not
   rename, do not restructure neighbouring code.
3. **Regression test.** Delegate to `mn-tester` to add a failing test
   that captures the bug. Confirm it fails.
4. **Fix.** Make the smallest change that turns the regression test
   green. If the minimal fix crosses package boundaries, stop and ask
   whether `/mn-refactor` or `/mn-impl` is more appropriate.
5. **Verify.** Run the full test file the test lives in; then run
   `/mn-test` for the touched package.

## Output

- One-line root cause.
- Diff summary (files + approximate line count).
- Regression test path.
- Test results.

## Rules

- No drive-by refactors. Fix only what's broken.
- No new public surface in a bug fix. If the fix needs new API, it's a
  feature — use `/mn-impl`.
