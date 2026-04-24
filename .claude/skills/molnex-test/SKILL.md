---
name: molnex-test
description: Test coverage analysis, gap identification, and scientific validation audit. Use after implementing a feature or before release.
argument-hint: "[path or module]"
user-invocable: true
---

Analyze test coverage for: $ARGUMENTS

If no path given, analyze the entire project.

**Step 1 — Run Tests**

```bash
# Full project
python -m pytest tests/ --cov=src --cov-report=term-missing -v

# Specific package
python -m pytest tests/test_<pkg>/ --cov=src/<pkg> --cov-report=term-missing -v
```

**Step 2 — Coverage Analysis**

Report coverage by package. Flag any module below 80%.

**Step 3 — Scientific Test Audit**

For each nn.Module implementing a physical model (in molzoo, molpot/potentials, molrep):
- [ ] Unit test for forward pass with correct output shape
- [ ] Numerical validation against reference values
- [ ] Rotation invariance/equivariance test (where applicable)
- [ ] Translation invariance test (where applicable)
- [ ] Permutation equivariance test (where applicable)
- [ ] Edge cases (single atom, empty graph)

**Step 4 — Report**

```
TEST COVERAGE REPORT

Overall: XX% (target: 80%)

By Package:
  molix.core:        XX% ✅/⚠️
  molrep.embedding:  XX% ✅/⚠️
  molpot.potentials: XX% ✅/⚠️
  molzoo:            XX% ✅/⚠️

Scientific Validation:
  ✅ <model>: <tests present>
  ❌ <model>: <missing tests>

Suggested Tests:
  1. <file>: <what to test>
```
