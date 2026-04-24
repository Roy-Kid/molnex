---
name: mn-arch
description: Single-axis architecture review — 4-package layering, import rules, edge convention, TensorDict nesting. Read-only.
argument-hint: <optional file paths>
user-invocable: true
---

# MolNex Architecture Check

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. Scope = `$ARGUMENTS` if provided, else `git diff master...HEAD --name-only`.
2. Delegate to the `mn-architect` agent with the scoped file list.
3. Render the agent's findings unchanged.

## Output

Architecture severity table + verdict from `mn-architect` — no reformatting.
