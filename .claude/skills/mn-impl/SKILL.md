---
name: mn-impl
description: Full implementation workflow from spec to production-ready code — scope → litrev → spec → arch → RED → GREEN → review → docs. Writes code, tests, and docs.
argument-hint: <feature description or path to spec>
user-invocable: true
---

# MolNex Implementation

Read `CLAUDE.md` for MolNex conventions.

## Procedure

1. **Scope assessment.** Classify the change:
   - *Trivial* (renames, doc-only) → skip to step 7.
   - *Standard* feature → full flow below.
   - *Novel physics* → require `/mn-litrev` first.

2. **Spec.** If `$ARGUMENTS` is a path to `.claude/specs/<slug>.md`, read
   it. Otherwise invoke `/mn-spec` to create one. Do not proceed without
   a spec on file.

3. **Architecture check (pre-impl).** Delegate to `mn-architect` with the
   spec. Request a module-impact map and a confirm/deny on the package
   placement.

4. **Physics check (if applicable).** Delegate to `mn-scientist` to
   confirm equations and required symmetries.

5. **RED.** Delegate to `mn-tester` to write failing tests covering
   required archetypes (forward shape, symmetries, numerical reference
   value). Run `pytest` and confirm the tests fail.

6. **GREEN.** Implement the feature in `src/<pkg>/` following the
   conventions in `CLAUDE.md` ("Key Design Patterns", "Adding New
   Components", "Docstring Convention", "Scientific Correctness
   Requirements"). Run `pytest` and confirm tests pass.

7. **Verify.** Fan-out `/mn-review` on the touched files.

8. **Docs.** Delegate to `mn-documenter` to fill in any missing docstrings,
   tensor-shape annotations, and tutorials.

9. **Architecture check (post-impl).** Re-run `mn-architect` on the final diff.

## Output

- Files created / modified.
- `pytest` summary and coverage for the touched module.
- Literature references recorded in docstrings.
- Verdicts from each delegated agent.
- Open TODOs.
