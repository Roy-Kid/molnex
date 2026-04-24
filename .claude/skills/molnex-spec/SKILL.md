---
name: molnex-spec
description: Convert natural language requirements into a detailed technical specification with literature grounding. Use before implementing new features.
argument-hint: <natural language description>
user-invocable: true
---

Generate a detailed technical spec for: $ARGUMENTS

**Step 1 — Understand Intent**
Parse the description to identify: what capability is added/changed, which packages are affected (molix, molrep, molpot, molzoo), what new nn.Modules or potentials are needed.

**Step 2 — Literature Search**
For any physical model or algorithm:
- Search arXiv and relevant journals for the original paper
- Extract key equations, parameters, physical constraints
- Identify reference implementations
- Document expected symmetry properties

**Step 3 — Codebase Analysis**
Read relevant existing code to understand:
- Module dependency graph and where the new code fits
- Existing similar implementations for consistency
- Dict-based data flow (`MoleculeSample`/`MoleculeBatch` keys)
- cuEquivariance integration patterns
- Pydantic config patterns

**Step 4 — Generate Spec**
Produce a structured markdown document:

```markdown
# Spec: <Feature Name>

## Summary
One-paragraph description.

## Scientific Basis
- Paper: [Author et al., "Title", Venue Year](arXiv link)
- Key equations (LaTeX)
- Physical symmetries to preserve
- Known approximations

## Design

### New Types
nn.Module subclasses, Pydantic configs, TypedDicts. Include tensor shapes for all inputs/outputs.

### Modified Types
Existing types that need changes (before/after).

### Module Changes
| File | Action | Description |
|------|--------|-------------|

### Interface Contracts
Dict keys consumed/produced. cuEquivariance irreps used.

## Tensor Flow
Shape diagram through computation graph.

## Dependencies
Internal modules, external packages, PyTorch version requirements (torch>=2.6).

## Testing Strategy
Unit tests (shapes), numerical validation (reference values), symmetry tests, edge cases.

## Performance Considerations
torch.compile compatibility, memory scaling, GPU utilization.
```

**Step 5 — Save or Present**
Save the spec as `docs/specs/<kebab-case-name>.md` for future reference.
Otherwise present the spec inline for review.
