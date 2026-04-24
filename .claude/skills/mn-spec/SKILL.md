---
name: mn-spec
description: Turns a natural-language feature request into a technical spec under .claude/specs/<slug>.md and appends a row to specs/INDEX.md. Writes to .claude/specs/.
argument-hint: <feature description>
user-invocable: true
---

# MolNex Spec

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. Parse `$ARGUMENTS` into a feature statement. If ambiguous, ask ONE
   clarifying question before writing.
2. Derive a kebab slug (`lowercase-dashes`, ≤ 6 words). Pick a target
   package (`molix` | `molrep` | `molpot` | `molzoo`).
3. If the feature involves a physical model, potential, or operator,
   delegate to the `mn-scientist` agent to confirm scope and expected
   invariants before drafting.
4. Write `.claude/specs/<slug>.md` with these sections:
   - **Problem.** One paragraph — why the feature is needed.
   - **Non-goals.** What this spec explicitly does not cover.
   - **Public surface.** Signature(s) users will call.
   - **Data contract.** Nested `TensorDict` keys consumed and produced,
     with shapes.
   - **Invariants.** Symmetries preserved, units, dtype.
   - **Reference.** Paper citations (if any), links to prior art.
   - **Open questions.** List, to be resolved during `/mn-impl`.
5. Append a row to `.claude/specs/INDEX.md` with status `draft`.

## Output

Path of the spec written and the INDEX row added. One sentence summary.
