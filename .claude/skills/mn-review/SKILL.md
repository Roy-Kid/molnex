---
name: mn-review
description: Multi-axis review of pending changes — architecture ∥ physics ∥ ML ∥ perf ∥ tests ∥ docs in parallel. Read-only, aggregates verdicts.
argument-hint: <optional file paths or PR number>
user-invocable: true
---

# MolNex Review

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. **Scope.** Determine the change set:
   - If `$ARGUMENTS` names file paths → review those.
   - If `$ARGUMENTS` looks like a PR number → `gh pr diff <N>` into a
     temp file and review that.
   - Otherwise → `git diff master...HEAD`.

2. **Fan out in parallel** (single message, multiple `Agent` calls):

   | Agent | Focus |
   |---|---|
   | `mn-architect` | layering, edge convention, TensorDict nesting |
   | `mn-scientist` | equations, symmetries, paper alignment |
   | `ml-expert` | training dynamics, loss, eval methodology |
   | `mn-optimizer` | perf, `torch.compile`, cuEquivariance |
   | `mn-tester` | test coverage, numerical validation, RED-first check |
   | `mn-documenter` | docstrings, tensor shapes, paper refs |

3. **Aggregate** the per-agent findings.

## Output

A single table:

| Severity | File:line | Message | Source agent |
|----------|-----------|---------|--------------|
| CRITICAL | ...       | ...     | mn-...       |
| HIGH     | ...       | ...     | mn-...       |
| MEDIUM   | ...       | ...     | mn-...       |
| LOW      | ...       | ...     | mn-...       |

End with one final verdict line: `APPROVE | REQUEST CHANGES | BLOCK`,
and a one-sentence rationale.

## Rules

- All delegate calls must be issued in a single assistant turn so they
  run in parallel.
- Never summarise one agent's findings with another agent's voice.
