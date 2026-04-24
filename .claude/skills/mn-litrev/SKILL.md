---
name: mn-litrev
description: Verifies scientific basis of a proposed physical model before implementation. Read-only — produces a credibility verdict.
argument-hint: <feature or paper reference>
user-invocable: true
---

# MolNex Literature Review

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. Resolve `$ARGUMENTS` to a paper, model family, or operator.
2. Delegate to the `mn-scientist` agent with:
   - The proposed feature.
   - Any cited arXiv / DOI.
   - A request to assess: reproducibility, reference-impl availability,
     equation consistency across sources, reported tolerances.
3. If no paper is cited and the feature is physics-bearing, refuse and
   ask the user to name a source.
4. Aggregate the agent's findings into a credibility verdict.

## Output

A short report:

- **Paper(s)**: titles + arXiv/DOI.
- **Reference implementation**: link or "none found".
- **Equation consistency**: OK | discrepancies observed (list).
- **Verdict**: `GREENLIGHT` | `PROCEED WITH CAUTION` | `BLOCK (no credible basis)`.

Final line: one-sentence recommendation for `/mn-impl`.
