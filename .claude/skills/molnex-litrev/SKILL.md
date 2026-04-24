---
name: molnex-litrev
description: Literature review to verify scientific basis before implementation. Use before implementing any physical model, potential, or operator.
argument-hint: <method or topic name>
user-invocable: true
---

Perform a literature review for: $ARGUMENTS

**Step 1 — Search Literature**
Search arXiv, journals, and code repositories:
- `"$ARGUMENTS" arxiv molecular dynamics`
- `"$ARGUMENTS" machine learning interatomic potential`
- `"$ARGUMENTS" equivariant neural network`
Find the original publication and follow-up papers.

**Step 2 — Extract Key Information**
From the primary paper:
- **Equations**: All key mathematical formulations
- **Parameters**: Default hyperparameters and valid ranges
- **Symmetries**: Guaranteed invariances/equivariances
- **Complexity**: Computational scaling with system size
- **Approximations**: What is approximated and why

**Step 3 — Find Reference Implementations**
Search for existing code:
- Official repository (usually linked in paper)
- PyTorch Geometric / e3nn implementations
- NVIDIA cuEquivariance examples

**Step 4 — Identify Validation Targets**
From papers or reference code:
- Published benchmark results (energies, forces on standard datasets)
- Known numerical values for simple test systems
- Expected accuracy ranges

**Step 5 — Report**
Output:

```markdown
# Literature Review: <Method Name>

## Primary Reference
- **Paper**: Author et al., "Title", Venue Year
- **arXiv**: URL
- **DOI**: URL

## Key Equations
[equations from the paper]

## Physical Properties
- Invariances, equivariances, conservation laws, approximations

## Reference Implementations
- [repo URL] — [framework, notes]

## Validation Targets
- Dataset: MAE energy/force values
- Simple test: reproducible numerical check

## Known Limitations

## Recommendations for MolNex
- Which molrep building blocks to reuse
- Numerical stability concerns
- cuEquivariance integration notes
```

If no credible paper is found, report clearly and ask the user for a reference.
