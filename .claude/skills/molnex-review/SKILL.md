---
name: molnex-review
description: Comprehensive code review aggregating architecture, performance, documentation, scientific correctness, and quality checks. Use after writing code or during PR review.
argument-hint: "[path or module]"
user-invocable: true
---

Review code for: $ARGUMENTS

If no path given, review all files modified in `git diff --name-only HEAD`.

**Invoke all dimensions in parallel:**

1. **Architecture** → invoke `/molnex-arch` on $ARGUMENTS
2. **Performance** → invoke `/molnex-perf` on $ARGUMENTS
3. **Documentation** → invoke `/molnex-docs` on $ARGUMENTS
4. **Scientific Correctness** (for molrep/, molpot/, molzoo/):
   - Equations match cited paper (arXiv/DOI)
   - Rotation/translation/permutation symmetries preserved
   - Autograd forces: F = -dE/dx via torch.autograd.grad
   - Cutoff functions smooth and continuous
   - Normalization consistent with paper convention
   - Units documented (eV, Å, etc.)
5. **Code Quality** (inline):
   - Functions < 50 lines, files < 800 lines
   - No deep nesting (> 4 levels)
   - No hardcoded magic numbers
   - Type annotations on all public APIs
   - Google-style docstrings with tensor shapes
   - Dict-first data flow (no TensorDict)
6. **Immutability** (inline):
   - No in-place tensor operations on inputs
   - Dict mutation creates new dicts
   - New tensors for transformed data

**Severity levels**:
- CRITICAL — must fix (architecture violations, scientific errors)
- HIGH — should fix (missing tests, performance issues)
- MEDIUM — fix when possible (style, documentation gaps)
- LOW — nice to have

**Output**: Merged report:
```
CODE REVIEW: <path>
ARCHITECTURE: ✅/❌ per check
PERFORMANCE: ✅/⚠️ per check
DOCUMENTATION: ✅/⚠️ per check
SCIENTIFIC CORRECTNESS: ✅/❌ per check
CODE QUALITY: ✅/⚠️ per check
IMMUTABILITY: ✅/❌ per check
SUMMARY: N CRITICAL, N HIGH, N MEDIUM, N LOW
```
