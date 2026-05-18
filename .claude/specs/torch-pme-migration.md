---
slug: torch-pme-migration
title: Migrate torch-pme into molnex as molelec
status: approved
created: 2026-05-17
revised: 2026-05-17
scope_layer: molelec (new top-level package)
conflict_decision: independent
---

## Summary

Port the [torch-pme](https://github.com/lab-cosmo/torch-pme) library — a PyTorch
implementation of Particle-Mesh Ewald (PME), P3M, and Ewald summation for
long-range interactions — into MolNex as the `molelec` package. Migrate all
source code, all tests, and adapt to MolNex code style (Google docstrings, no
TorchScript decorators, modern PyTorch APIs, `torch.compile` compatible).

## Domain basis

torch-pme provides differentiable, GPU-accelerated computation of long-range
electrostatic and dispersion interactions via:

- **Ewald summation** — O(N²) exact reciprocal-space sum
- **PME (Particle-Mesh Ewald)** — O(N log N) via FFT with Lagrange interpolation
- **P3M (Particle-Particle Particle-Mesh)** — O(N log N) with optimized Green's function

These are foundational for building ML potentials that respect periodic boundary
conditions. MolNex currently lacks a long-range electrostatics module; `molelec`
fills this gap and sits alongside `molrep` (representation) and `molpot`
(potentials) in the architecture.

### References

- Ewald, P. *Ann. Phys.* 369, 253–287 (1921)
- Darden, T. et al. *J. Chem. Phys.* 98, 10089–10092 (1993) — PME
- Hockney, R.W. & Eastwood, J.W. *Computer Simulation Using Particles* (1988) — P3M
- Deserno, M. & Holm, C. *J. Chem. Phys.* 109, 7678–7693 (1998) — P3M error estimates
- [torch-pme GitHub](https://github.com/lab-cosmo/torch-pme)

## Design

### Package layout

```
src/molelec/
├── __init__.py           # public re-exports
├── _utils.py             # parameter validation
├── prefactors.py         # physical unit conversion factors
├── potentials/
│   ├── __init__.py
│   ├── potential.py      # Potential base class (nn.Module)
│   ├── coulomb.py        # CoulombPotential (1/r)
│   ├── inversepowerlaw.py # InversePowerLawPotential (1/r^p)
│   ├── spline.py         # SplinePotential
│   ├── combined.py       # CombinedPotential (linear combination)
│   └── potential_dipole.py # PotentialDipole (dipole-dipole)
├── calculators/
│   ├── __init__.py
│   ├── calculator.py     # Calculator base class (nn.Module)
│   ├── ewald.py          # EwaldCalculator
│   ├── pme.py            # PMECalculator
│   ├── p3m.py            # P3MCalculator
│   └── calculator_dipole.py # CalculatorDipole
├── lib/
│   ├── __init__.py
│   ├── kvectors.py       # reciprocal-space vector generation
│   ├── kspace_filter.py  # KSpaceFilter, P3MKSpaceFilter
│   ├── mesh_interpolator.py # MeshInterpolator (Lagrange + P3M)
│   ├── splines.py        # CubicSpline, CubicSplineReciprocal
│   └── math.py           # gamma, exp1, gammaincc_over_powerlaw
├── tuning/
│   ├── __init__.py
│   ├── tuner.py          # TunerBase, GridSearchTuner, TuningTimings
│   ├── ewald.py          # tune_ewald, EwaldErrorBounds
│   ├── pme.py            # tune_pme, PMEErrorBounds
│   └── p3m.py            # tune_p3m, P3MErrorBounds
└── metatensor/
    ├── __init__.py
    ├── calculator.py     # metatensor Calculator wrapper
    ├── ewald.py          # metatensor EwaldCalculator
    ├── pme.py            # metatensor PMECalculator
    └── p3m.py            # metatensor P3MCalculator
```

### Style adaptations (from torch-pme → molelec)

1. **Remove `@torch.jit.export`** — all decorators stripped. MolNex targets
   `torch.compile`, not TorchScript.
2. **Remove `torch.jit.script` compatibility workarounds** — no more
   `# type: ignore` comments for TorchScript type narrowing, no
   `self.smearing = None` then `register_buffer` pattern (just use
   `Optional[Tensor]` attributes directly).
3. **Google-style docstrings** — replace RST-style `:param:` with
   `Args:`/`Returns:` blocks, tensor shape annotations as
   ``(N, 3)``.
4. **Imports** — flat `from molelec.potentials import ...`, no nested
   relative imports beyond one level.
5. **`__init__.py` re-exports** — match MolNex pattern: each subpackage
   re-exports its public symbols.
6. **`_utils.py`** — keep parameter validation at package level (was
   `_utils.py` in torch-pme).
7. **metatensor module** — guarded import (`try: import metatensor`)
   as optional subpackage; skip if metatensor-torch not installed.
8. **`dtype=torch.float64`** — keep float64 default for potentials
   (scientific correctness requires it); no forced dtype coercion.

### What stays unchanged

- Core algorithm implementations (Ewald sum, PME mesh interpolation, P3M
  influence function, spline math, error bounds)
- Public API surface (class names, method signatures, parameter names)
- Test logic and test parameter grids

### Integration points

- `molelec` is a standalone package with no dependencies on `molix`,
  `molrep`, `molpot`, or `molzoo`
- Downstream: `molpot` potentials/future heads may consume `molelec`
  calculators for long-range energy terms
- The `molix.data` neighbor list convention (source→target edges,
  `bond_diff = pos[target] - pos[source]`) is compatible with
  torch-pme's convention where `neighbor_indices[:, 0]` = source (atom i)
  and `[:, 1]` = target (atom j)

## Files

| File | Action | Notes |
|------|--------|-------|
| `src/molelec/__init__.py` | Create | Public re-exports |
| `src/molelec/_utils.py` | Port | From `torchpme/_utils.py` |
| `src/molelec/prefactors.py` | Port | From `torchpme/prefactors.py` |
| `src/molelec/potentials/__init__.py` | Create | Re-exports |
| `src/molelec/potentials/potential.py` | Port + adapt | Remove @torch.jit.export, Google docstrings |
| `src/molelec/potentials/coulomb.py` | Port + adapt | Same |
| `src/molelec/potentials/inversepowerlaw.py` | Port + adapt | Same |
| `src/molelec/potentials/spline.py` | Port + adapt | Same |
| `src/molelec/potentials/combined.py` | Port + adapt | Same |
| `src/molelec/potentials/potential_dipole.py` | Port + adapt | Same |
| `src/molelec/calculators/__init__.py` | Create | Re-exports |
| `src/molelec/calculators/calculator.py` | Port + adapt | Remove @torch.jit.export, Google docstrings |
| `src/molelec/calculators/ewald.py` | Port + adapt | Same |
| `src/molelec/calculators/pme.py` | Port + adapt | Same |
| `src/molelec/calculators/p3m.py` | Port + adapt | Same |
| `src/molelec/calculators/calculator_dipole.py` | Port + adapt | Same |
| `src/molelec/lib/__init__.py` | Create | Re-exports |
| `src/molelec/lib/kvectors.py` | Port + adapt | Google docstrings |
| `src/molelec/lib/kspace_filter.py` | Port + adapt | Remove @torch.jit.export |
| `src/molelec/lib/mesh_interpolator.py` | Port + adapt | Google docstrings |
| `src/molelec/lib/splines.py` | Port + adapt | Google docstrings |
| `src/molelec/lib/math.py` | Port + adapt | Google docstrings |
| `src/molelec/tuning/__init__.py` | Create | Re-exports |
| `src/molelec/tuning/tuner.py` | Port + adapt | Google docstrings |
| `src/molelec/tuning/ewald.py` | Port + adapt | Same |
| `src/molelec/tuning/pme.py` | Port + adapt | Same |
| `src/molelec/tuning/p3m.py` | Port + adapt | Same |
| `src/molelec/metatensor/__init__.py` | Create | Guarded import |
| `src/molelec/metatensor/calculator.py` | Port + adapt | Google docstrings |
| `src/molelec/metatensor/ewald.py` | Port + adapt | Same |
| `src/molelec/metatensor/pme.py` | Port + adapt | Same |
| `src/molelec/metatensor/p3m.py` | Port + adapt | Same |
| `tests/test_molelec/__init__.py` | Create | Empty |
| `tests/test_molelec/test_init.py` | Port | From `tests/test_init.py` |
| `tests/test_molelec/test_prefactors.py` | Port | From `tests/test_prefactors.py` |
| `tests/test_molelec/test_potentials.py` | Port + adapt | Update imports |
| `tests/test_molelec/helpers.py` | Port | From `tests/helpers.py` |
| `tests/test_molelec/calculators/__init__.py` | Create | Empty |
| `tests/test_molelec/calculators/test_calculator.py` | Port + adapt | Update imports |
| `tests/test_molelec/calculators/test_workflow.py` | Port + adapt | Update imports |
| `tests/test_molelec/calculators/test_padding.py` | Port + adapt | Update imports |
| `tests/test_molelec/calculators/test_values_direct.py` | Port + adapt | Update imports |
| `tests/test_molelec/calculators/test_values_dipole.py` | Port + adapt | Update imports |
| `tests/test_molelec/calculators/test_values_ewald.py` | Port + adapt | Update imports |
| `tests/test_molelec/lib/__init__.py` | Create | Empty |
| `tests/test_molelec/lib/test_kspace_filter.py` | Port + adapt | Update imports |
| `tests/test_molelec/lib/test_splines.py` | Port + adapt | Update imports |
| `tests/test_molelec/lib/test_kvectors.py` | Port + adapt | Update imports |
| `tests/test_molelec/lib/test_mesh_interpolator.py` | Port + adapt | Update imports |
| `tests/test_molelec/lib/test_math.py` | Port + adapt | Update imports |
| `tests/test_molelec/tuning/__init__.py` | Create | Empty |
| `tests/test_molelec/tuning/test_error_bounds.py` | Port + adapt | Update imports |
| `tests/test_molelec/tuning/test_tuning.py` | Port + adapt | Update imports |
| `tests/test_molelec/tuning/test_timer.py` | Port + adapt | Update imports |
| `tests/test_molelec/metatensor/__init__.py` | Create | Empty |
| `tests/test_molelec/metatensor/test_workflow_metatensor.py` | Port + adapt | Update imports |
| `tests/test_molelec/metatensor/test_calculator_metatensor.py` | Port + adapt | Update imports |

## Tasks

- [ ] Create `src/molelec/__init__.py` with public re-exports
- [ ] Port `_utils.py` (parameter validation)
- [ ] Port `prefactors.py` (physical conversion factors)
- [ ] Port `potentials/` subpackage (6 files)
- [ ] Port `calculators/` subpackage (5 files)
- [ ] Port `lib/` subpackage (5 files)
- [ ] Port `tuning/` subpackage (4 files)
- [ ] Port `metatensor/` subpackage (4 files, optional import guard)
- [ ] Port all test files (20 files)
- [ ] Verify `import molelec` works
- [ ] Run full test suite: `python -m pytest tests/test_molelec/ -v`
- [ ] Verify `torch.compile` compatibility on core modules

## Testing

All existing torch-pme tests are migrated with import paths updated from
`torchpme` → `molelec`. Test coverage should match or exceed the original:

- **Unit tests**: potentials, splines, kvectors, mesh interpolation, math
  functions, kspace filter, error bounds
- **Integration tests**: calculator workflows, value correctness (direct,
  Ewald, dipole), padding behavior
- **Metatensor tests**: metatensor workflow and calculator (skipped if
  metatensor-torch not installed)

## Out of scope

- Rewriting algorithms for performance
- Adding new potential types or calculators
- Integration with `molpot` or `molix` pipelines
- CUDA kernel implementations
- Adding dipole/P3M support beyond what torch-pme already provides
- Changing the metatensor dependency version requirements
