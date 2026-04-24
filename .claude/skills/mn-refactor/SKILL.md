---
name: mn-refactor
description: Restructure code preserving invariants — same public surface, same numerical outputs, better internals. Writes code.
argument-hint: <refactor goal>
user-invocable: true
---

# MolNex Refactor

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. **State the invariant.** Write down, in `$ARGUMENTS`'s terms, what
   the refactor must preserve (public API signatures, numerical outputs,
   symmetries, test pass/fail state).
2. **Architecture check (pre).** Delegate to `mn-architect` for a map of
   current module boundaries and forbidden cross-layer imports.
3. **Snapshot tests.** Delegate to `mn-tester` to add characterisation
   tests on the boundary to be refactored (if missing), so behaviour is
   pinned before moving code.
4. **Refactor.** Move, split, rename. Do not change behaviour. Keep
   commits mechanical where possible.
5. **Verify.** Run the full `pytest` suite. Diff must pass every test
   that passed before.
6. **Architecture check (post).** Re-run `mn-architect` on the new tree;
   confirm the improvement it was supposed to enable landed.

## Output

- Invariant statement (as stated before the refactor).
- Files moved / renamed / split.
- Before/after module-impact summary from `mn-architect`.
- Full test suite result (must be all green).

## Rules

- Never combine a refactor with a feature or a bug fix in the same
  commit. If you notice a bug mid-refactor, pause and handle it with
  `/mn-fix`, then resume.
