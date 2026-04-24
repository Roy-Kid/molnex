---
name: mn-docs
description: Documentation audit and expansion — Google-style docstrings with tensor shapes, paper references, and tutorial prose. Writes to src/ docstrings and docs/.
argument-hint: <optional file or package>
user-invocable: true
---

# MolNex Documentation

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. Scope = `$ARGUMENTS` if provided, else the diff against `master`
   filtered to `src/` and `docs/`.
2. Delegate to the `mn-documenter` agent, asking for:
   - docstring completeness on public symbols,
   - tensor shape annotations,
   - `Reference:` blocks for physical models,
   - tutorial updates in `docs/` where user-facing API changed.
3. Let the agent edit files directly.
4. Run `python -m pytest --doctest-modules src/<pkg>/` on any module
   whose docstrings contain doctest blocks, to catch examples that rot.

## Output

- Files with edited docstrings.
- Tutorials created / updated under `docs/`.
- Remaining undocumented public symbols (as TODOs).
