---
name: mn-perf
description: Single-axis performance review — PyTorch anti-patterns, torch.compile, memory, cuEquivariance tuning. Read-only.
argument-hint: <optional file paths>
user-invocable: true
---

# MolNex Performance Check

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. Scope = `$ARGUMENTS` if provided, else the diff against `master`.
2. Delegate to the `mn-optimizer` agent. Encourage the agent to run a
   quick microbench (`pytest tests/ -k bench`) if the touched files are
   on a training hot path.
3. Render findings unchanged.

## Output

Severity table + verdict from `mn-optimizer`, with estimated
speedup / memory delta where quantified.
