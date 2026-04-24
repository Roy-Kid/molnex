---
name: mn-test
description: Test coverage analysis and expansion for MolNex — symmetry, numerical-validation, and edge-case gaps. Writes to tests/.
argument-hint: <optional package or file>
user-invocable: true
---

# MolNex Test Expansion

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. Identify the target package (`$ARGUMENTS` or diff against `master`).
2. Run coverage: `pytest tests/test_<pkg>/ --cov=src/<pkg> --cov-report=term-missing`.
3. Delegate to the `mn-tester` agent with the coverage report and the
   list of touched symbols, asking for:
   - missing test archetypes (forward-shape, symmetry, numerical reference),
   - edge cases (empty graph, single atom, disconnected components),
   - finite-difference force-check where a potential is involved.
4. Let `mn-tester` write the new tests.
5. Rerun `pytest` and report pass/fail + final coverage.

## Output

- Files touched by `mn-tester`.
- New `pytest` summary.
- Coverage delta (before → after), target ≥ 80 %.
- Any gap the agent declined to fill with rationale.
