---
name: mn-debug
description: Diagnose-only — reproduces an issue and reports hypotheses with evidence. NEVER edits code. NEVER commits.
argument-hint: <bug description or test failure>
user-invocable: true
---

# MolNex Debug (read-only)

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. **Reproduce.** Run the failing test or command from `$ARGUMENTS` via
   `Bash`. Capture traceback and exit code.
2. **Narrow.** Use `Grep`, `Glob`, and `Read` (only) to walk from the
   symptom to the likely root cause. For physics bugs, delegate to
   `mn-scientist` for a correctness read on the suspect equation. For
   shape bugs, delegate to `mn-architect` for an edge-convention and
   TensorDict-nesting check.
3. **Hypotheses.** List the top 3 candidate root causes, each with
   evidence (file:line references or log excerpts).
4. **Experiments.** Propose read-only commands or one-line prints the
   user can run to confirm each hypothesis.

## Output

- Reproduction command + captured output.
- Ranked hypotheses with evidence per hypothesis.
- Proposed confirmation experiments.
- **Next step**: pointer to `/mn-fix` with the suspected root cause.

## Rules

- NEVER call `Edit` or `Write`. NEVER run `git commit`, `git push`, or
  any state-changing command. If you reach a point where a code change
  is needed, stop and hand off to `/mn-fix`.
- NEVER guess silently — if evidence is thin, say so.
