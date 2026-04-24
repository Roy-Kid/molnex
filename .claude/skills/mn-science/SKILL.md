---
name: mn-science
description: Single-axis scientific-correctness review — equations, symmetries, paper alignment. Read-only.
argument-hint: <optional file paths or module>
user-invocable: true
---

# MolNex Scientific Correctness Check

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. Scope = `$ARGUMENTS` if provided, else the diff against `master`.
2. Delegate to the `mn-scientist` agent, requesting:
   - equation-vs-paper alignment,
   - required symmetries and whether tests exist,
   - unit and constant sanity checks.
3. Render findings unchanged.

## Output

Severity table + verdict from `mn-scientist`.
