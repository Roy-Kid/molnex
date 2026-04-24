---
name: molnex-arch
description: Validate code against MolNex module dependency rules and architecture patterns. Use before code review, after refactoring imports, or when adding cross-package dependencies.
argument-hint: "[path or module]"
user-invocable: true
---

Validate architecture for: $ARGUMENTS

If no path given, check all files modified in `git diff --name-only HEAD`.

**Dependency Rules**

```
ALLOWED (← = can be imported by):
  molix.config          ← molrep, molpot, molzoo
  molrep.embedding      ← molrep.interaction, molzoo
  molrep.interaction    ← molrep.readout, molzoo
  molrep.readout        ← molpot.heads, molzoo
  molzoo                ← application code only
  molpot.potentials     ← molpot.composition
  molpot.heads          ← molpot.composition
  molpot.composition    ← application code only
  molix.core            ← application code only
  molix.data            ← application code, molix.datasets

FORBIDDEN:
  molrep → molpot, molzoo
  molpot → molzoo
  molzoo → molpot (encoder-only)
  molix.data → molix.core
```

**Checks**

1. **Import direction**: Scan `.py` files for import violations against the rules above.
2. **Dict-first compliance**: Flag custom data container classes — all molecular data must be `dict[str, Tensor]`.
3. **Pydantic config**: Every `nn.Module` with >3 constructor args should have a companion `BaseModel` config.
4. **Type annotations**: All `forward()` methods must have full type annotations.
5. **Tensor shapes**: Docstrings must document tensor shapes using ``(n_nodes, dim)`` notation.
6. **Immutability**: Flag in-place tensor ops (`add_()`, `mul_()`) and input dict mutation.
7. **Encoder-only**: molzoo modules must not contain readout logic (no EnergyHead, ForceHead).
8. **BasePotential**: All molpot potentials must inherit `BasePotential`.

**Output format**:
```
ARCHITECTURE VALIDATION: <path>

✅ Import directions: OK
❌ <violation description> (file:line)
⚠️ <warning description> (file:line)

N ERRORS, M WARNINGS
```
