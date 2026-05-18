---
slug: torch-pme-migration
title: 将 torch-pme 迁移到 molpot.elec — acceptance criteria
status: code-complete
spec: torch-pme-migration.md
---

## Acceptance Criteria

### type: code

- [x] **AC-01**: `src/molpot/elec/__init__.py` 导出 `Calculator`, `EwaldCalculator`,
  `PMECalculator`, `P3MCalculator`, `CoulombPotential`, `Potential`,
  `InversePowerLawPotential`, `SplinePotential`, `CombinedPotential`,
  `PotentialDipole`, `CalculatorDipole`
  → verified 2026-05-17
- [x] **AC-02**: 所有 `src/molpot/elec/` 下的源文件使用 Google 风格 docstrings，
  包含 `Args:`/`Returns:` 块和张量形状标注
  → verified 2026-05-17
- [x] **AC-03**: 无 `@torch.jit.export` 装饰器残留
  → verified 2026-05-17
- [x] **AC-04**: 无 `torchpme` 导入残留；所有内部导入使用 `molpot.elec` 前缀
  （如 `from molpot.elec.potentials import Potential`）
  → verified 2026-05-17
- [x] **AC-05**: 无 metatensor 相关代码
  → verified 2026-05-17
- [x] **AC-06**: 17 个测试文件位于 `tests/test_molpot/test_elec/`，导入全部更新为
  `molpot.elec`
  → verified 2026-05-17
- [x] **AC-07**: `torch.compile` 在 `Potential`, `CoulombPotential`,
  `Calculator`, `EwaldCalculator`, `PMECalculator`, `KSpaceFilter`,
  `MeshInterpolator` 上通过
  → verified 2026-05-17

### type: runtime

- [x] **AC-08**: `python -m pytest tests/test_molpot/test_elec/ -v` 零失败
  → verified 2026-05-17 (1382 passed, 0 failed, 8 skipped)
- [x] **AC-09**: `python -c "from molpot.elec import PMECalculator"` 成功
  → verified 2026-05-17
- [x] **AC-10**: Coulomb potential SR/LR 拆分数值测试通过
  （SR + LR = 完整 1/r，误差在机器精度内）
  → verified 2026-05-17 (test_sr_lr_split passes)
- [x] **AC-11**: PME calculator 对简单的 2 原子 NaCl 体系产生有限、非 NaN 输出
  → verified 2026-05-17 (workflow tests pass)
