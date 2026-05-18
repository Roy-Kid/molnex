---
slug: torch-pme-migration
title: 将 torch-pme 迁移到 molpot.elec
status: done
created: 2026-05-17
revised: 2026-05-17
scope_layer: molpot.elec (molpot 子包)
conflict_decision: independent
---

## Summary

将 [torch-pme](https://github.com/lab-cosmo/torch-pme) 库移植到 MolNex 的
`molpot.elec` 子包下。迁移全部源代码和全部测试，适配 MolNex 代码风格（Google
docstrings，去除 TorchScript 装饰器，兼容 `torch.compile`）。

不迁移 metatensor 模块（MolNex 不使用 metatensor）。

## Domain basis

torch-pme 提供可微分、GPU 加速的长程静电和色散相互作用计算：

- **Ewald summation** — O(N²) 精确倒空间求和
- **PME (Particle-Mesh Ewald)** — O(N log N) FFT + Lagrange 插值
- **P3M (Particle-Particle Particle-Mesh)** — O(N log N) 优化 Green's 函数

### References

- Ewald, P. *Ann. Phys.* 369, 253–287 (1921)
- Darden, T. et al. *J. Chem. Phys.* 98, 10089–10092 (1993) — PME
- Hockney, R.W. & Eastwood, J.W. *Computer Simulation Using Particles* (1988) — P3M
- Deserno, M. & Holm, C. *J. Chem. Phys.* 109, 7678–7693 (1998) — P3M error estimates
- [torch-pme GitHub](https://github.com/lab-cosmo/torch-pme)

## Design

### Package layout

```
src/molpot/elec/
├── __init__.py              # 公开导出
├── _utils.py                # 参数校验
├── prefactors.py            # 物理单位转换因子
├── potentials/
│   ├── __init__.py
│   ├── potential.py         # Potential 基类 (nn.Module)
│   ├── coulomb.py           # CoulombPotential (1/r)
│   ├── inversepowerlaw.py   # InversePowerLawPotential (1/r^p)
│   ├── spline.py            # SplinePotential
│   ├── combined.py          # CombinedPotential (线性组合)
│   └── potential_dipole.py  # PotentialDipole (偶极-偶极)
├── calculators/
│   ├── __init__.py
│   ├── calculator.py        # Calculator 基类 (nn.Module)
│   ├── ewald.py             # EwaldCalculator
│   ├── pme.py               # PMECalculator
│   ├── p3m.py               # P3MCalculator
│   └── calculator_dipole.py # CalculatorDipole
├── lib/
│   ├── __init__.py
│   ├── kvectors.py          # 倒空间向量生成
│   ├── kspace_filter.py     # KSpaceFilter, P3MKSpaceFilter
│   ├── mesh_interpolator.py # MeshInterpolator (Lagrange + P3M)
│   ├── splines.py           # CubicSpline, CubicSplineReciprocal
│   └── math.py              # gamma, exp1, gammaincc_over_powerlaw
└── tuning/
    ├── __init__.py
    ├── tuner.py             # TunerBase, GridSearchTuner, TuningTimings
    ├── ewald.py             # tune_ewald, EwaldErrorBounds
    ├── pme.py               # tune_pme, PMEErrorBounds
    └── p3m.py               # tune_p3m, P3MErrorBounds
```

### 风格适配 (torch-pme → molpot.elec)

1. **去除 `@torch.jit.export`** — 全部删除。MolNex 使用 `torch.compile`。
2. **去除 TorchScript 兼容代码** — 删除 `# type: ignore` for TorchScript，
   删除 `self.smearing = None` then `register_buffer` 模式。
3. **Google 风格 docstrings** — 用 `Args:`/`Returns:` 替代 `:param:`，
   张量形状标注用 ``(N, 3)``。
4. **导入路径** — `from molpot.elec.potentials import ...`，不使用深层相对导入。
5. **`__init__.py` 重导出** — 每个子包导出公开符号。
6. **metatensor 模块不迁移**。

### 保持不变

- 核心算法实现（Ewald sum, PME mesh 插值, P3M influence function, spline 数学, error bounds）
- 公开 API（类名、方法签名、参数名）
- 测试逻辑和参数网格

### 与现有代码的关系

- `molpot.elec` 是 `molpot` 的子包，与 `molpot.potentials`（经典力场项）互补
- 不与 `molix`/`molrep`/`molzoo` 耦合
- 下游：`molpot.composition.PotentialComposer` 可使用 `molpot.elec` 计算器作为长程能量项
- `molix.data` 的边约定（source→target, `bond_diff = pos[target] - pos[source]`）
  与 torch-pme 的 `neighbor_indices[:, 0]` = source 兼容

## Files

| File | Action | Notes |
|------|--------|-------|
| `src/molpot/elec/__init__.py` | Create | Public re-exports |
| `src/molpot/elec/_utils.py` | Port | From `torchpme/_utils.py` |
| `src/molpot/elec/prefactors.py` | Port | From `torchpme/prefactors.py` |
| `src/molpot/elec/potentials/__init__.py` | Create | Re-exports |
| `src/molpot/elec/potentials/potential.py` | Port + adapt | Remove @torch.jit.export, Google docstrings |
| `src/molpot/elec/potentials/coulomb.py` | Port + adapt | Same |
| `src/molpot/elec/potentials/inversepowerlaw.py` | Port + adapt | Same |
| `src/molpot/elec/potentials/spline.py` | Port + adapt | Same |
| `src/molpot/elec/potentials/combined.py` | Port + adapt | Same |
| `src/molpot/elec/potentials/potential_dipole.py` | Port + adapt | Same |
| `src/molpot/elec/calculators/__init__.py` | Create | Re-exports |
| `src/molpot/elec/calculators/calculator.py` | Port + adapt | Remove @torch.jit.export, Google docstrings |
| `src/molpot/elec/calculators/ewald.py` | Port + adapt | Same |
| `src/molpot/elec/calculators/pme.py` | Port + adapt | Same |
| `src/molpot/elec/calculators/p3m.py` | Port + adapt | Same |
| `src/molpot/elec/calculators/calculator_dipole.py` | Port + adapt | Same |
| `src/molpot/elec/lib/__init__.py` | Create | Re-exports |
| `src/molpot/elec/lib/kvectors.py` | Port + adapt | Google docstrings |
| `src/molpot/elec/lib/kspace_filter.py` | Port + adapt | Remove @torch.jit.export |
| `src/molpot/elec/lib/mesh_interpolator.py` | Port + adapt | Google docstrings |
| `src/molpot/elec/lib/splines.py` | Port + adapt | Google docstrings |
| `src/molpot/elec/lib/math.py` | Port + adapt | Google docstrings |
| `src/molpot/elec/tuning/__init__.py` | Create | Re-exports |
| `src/molpot/elec/tuning/tuner.py` | Port + adapt | Google docstrings |
| `src/molpot/elec/tuning/ewald.py` | Port + adapt | Same |
| `src/molpot/elec/tuning/pme.py` | Port + adapt | Same |
| `src/molpot/elec/tuning/p3m.py` | Port + adapt | Same |
| `tests/test_molpot/test_elec/__init__.py` | Create | Empty |
| `tests/test_molpot/test_elec/test_init.py` | Port | From `tests/test_init.py` |
| `tests/test_molpot/test_elec/test_prefactors.py` | Port | From `tests/test_prefactors.py` |
| `tests/test_molpot/test_elec/test_potentials.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/helpers.py` | Port | From `tests/helpers.py` |
| `tests/test_molpot/test_elec/calculators/__init__.py` | Create | Empty |
| `tests/test_molpot/test_elec/calculators/test_calculator.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/calculators/test_workflow.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/calculators/test_padding.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/calculators/test_values_direct.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/calculators/test_values_dipole.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/calculators/test_values_ewald.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/lib/__init__.py` | Create | Empty |
| `tests/test_molpot/test_elec/lib/test_kspace_filter.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/lib/test_splines.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/lib/test_kvectors.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/lib/test_mesh_interpolator.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/lib/test_math.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/tuning/__init__.py` | Create | Empty |
| `tests/test_molpot/test_elec/tuning/test_error_bounds.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/tuning/test_tuning.py` | Port + adapt | Update imports |
| `tests/test_molpot/test_elec/tuning/test_timer.py` | Port + adapt | Update imports |

## Tasks

- [ ] Create `src/molpot/elec/__init__.py` with public re-exports
- [ ] Port `_utils.py` (parameter validation)
- [ ] Port `prefactors.py` (physical conversion factors)
- [ ] Port `potentials/` subpackage (6 files)
- [ ] Port `calculators/` subpackage (5 files)
- [ ] Port `lib/` subpackage (5 files)
- [ ] Port `tuning/` subpackage (4 files)
- [ ] Port all test files (17 test files)
- [ ] Verify `from molpot.elec import PMECalculator` works
- [ ] Run full test suite: `python -m pytest tests/test_molpot/test_elec/ -v`
- [ ] Verify `torch.compile` compatibility on core modules

## Testing

所有 torch-pme 测试均迁移，导入路径从 `torchpme` 更新为 `molpot.elec`：

- **单元测试**: potentials, splines, kvectors, mesh interpolation, math, kspace filter, error bounds
- **集成测试**: calculator workflows, value correctness (direct, Ewald, dipole), padding

## Out of scope

- metatensor 模块（不迁移）
- 算法性能重写
- 新增 potential 类型或 calculator
- 与 `molpot.composition.PotentialComposer` 的集成
- CUDA kernel 实现
