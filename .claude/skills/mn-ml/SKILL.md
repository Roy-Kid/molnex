---
name: mn-ml
description: Single-axis ML review — training dynamics, loss, hyperparameter sensitivity, evaluation protocol. Read-only.
argument-hint: <training run or config to review>
user-invocable: true
---

# MolNex ML Review

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. Gather context:
   - If `$ARGUMENTS` is a config path → read it.
   - If `$ARGUMENTS` is a run-id under `.benchmarks/` or `runs/` → point
     the agent at its logs.
   - Otherwise → review the training-related diff against `master`.
2. Delegate to the `ml-expert` agent with the gathered context. Ask for:
   - loss-balance sanity (per-atom energy vs force; rho weighting),
   - evaluation protocol (split seed, SI units, ≥3 seeds for published),
   - convergence / gradient-norm sanity.
3. Render findings unchanged.

## Output

Severity table + verdict from `ml-expert`.
