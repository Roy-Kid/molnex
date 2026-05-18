---
title: Sonata-05-MD — manifest-driven short MD driver for bulk water
status: approved
created: 2026-05-11
---

# Sonata-05-MD — manifest-driven short MD driver for bulk water

## Summary

为 bulk-water RPBE-D3 端到端链路引入首批 in-tree MD 代码：在 `src/molix/md/` 下新建一个侧层子包（与 `molix.profiler` 同级，**训练代码永不导入**），实现一个 flat `MDState`、Velocity-Verlet NVE / NVT 积分器、Langevin BAOAB 恒温器、以及 `run_md` 主循环；并在 `src/molix/bench/drivers/md.py` 下提供 `run_md_stage(manifest)` 驱动，它从 `sonata-05-train` 阶段产物中加载 checkpoint，对一帧 `WaterLESSource(split="test")` 帧做 Maxwell-Boltzmann 初速分配，跑 NVE（10–20 ps）+ NVT（50–100 ps），把 `trajectory.xyz`（经 `molpy.io.trajectory.XYZTrajectoryWriter`）和 `md_log.json`（每帧 step / time / E_total / E_kin / T / max|F| / NaN-count / 可选 Sonata 多极矩）写入 `manifest.to_artifact_dir()`，供 `sonata-05-obs` 消费。

## Domain basis

- **Velocity-Verlet**（kick-drift-kick）：第一步 `v ← v + 0.5·a·dt`，第二步 `r ← r + v·dt`（drift 后施加最小镜像折回 `cell`），第三步重新评估力 `a ← F(r)/m`，第四步 `v ← v + 0.5·a·dt`。Toy-harmonic 验证中，float64 下 1000 步 `dt=0.5 fs` 的总能漂移 < 1e-6（相对单位）。
- **Langevin BAOAB**：Leimkuhler & Matthews, *J. Chem. Phys.* 138:174102 (2013), doi:10.1063/1.4802990。每步把 OU 半步 `v ← e^{-γ·dt/2}·v + √((1-e^{-γ·dt})·k_B·T/m)·ξ`（ξ ~ N(0,1)）放在 drift 前与 kick 后，使配置分布对 dt 二阶精确。Default `gamma = 1.0 / ps`。
- **能量漂移工程门**：NVE 上 < 1 meV/atom/ps 为良好，< 10 meV/atom/ps 可接受；此值非文献钉死，是 MLIP-MD 通用经验（GROMACS 论坛口径）。spec 与 acceptance 中明确标注 "engineering rule of thumb, not literature-pinned"。
- **时间步**：水（带 H、无键约束）默认 `dt = 0.5 fs`；`dt > 1.0 fs` 触发 `ValueError`，防止 fs/ps 单位混淆。
- **初条件**：Maxwell-Boltzmann at `T_init = T_target`，从 `WaterLESSource(split="test")[0]` 取盒子与坐标；NVT 跑前 20 ps 等化，后段做诊断统计。
- **单位**：内部统一 eV / Å / fs / amu / K，BAOAB 中 `k_B = 8.617333262e-5 eV/K`，速度由 `v = √(k_B·T/m)` 标度。
- **轨迹格式**：`molpy.io.trajectory.xyz.XYZTrajectoryWriter` 仅写 `n_atoms / Lattice-in-comment / element x y z`；能量/力/电荷不进 XYZ，全部走 `md_log.json` 并行流。
- **参考**：Cheng B., *Latent Ewald summation*, npj Comput. Mater. 11:80 (2025), doi:10.1038/s41524-025-01577-7（200 ps NVT 先例）。

## Design

新增侧层子包 `molix.md`，依赖闭包 `{molix.data.types, molix.data.collate, molix.nn.locality, molpot.*, molpy.*}`，**不被** `molix.core`、`molix.data`、`molix.bench.train` 等训练侧代码导入；CI 在 `molix.core.__init__` 等顶层增加显式 import-time 反查（test）保证侧层不回流。

实体与符号：

- `MDState`（`state.py`）：`@dataclass` 持有张量 `Z (N,) long / pos (N,3) f64 / velocities (N,3) f64 / forces (N,3) f64 / masses (N,) f64`、`cell (3,3) f64`、`time: float`、`pbc: bool = True`。Dataclass 本身 frozen=False（张量原地更新），但替换张量字段时显式经 `replace_field`，避免 hooks 别名复用。`to_graph_batch()` 走 `molix.data.collate.collate_molecules` 走出 `GraphBatch`（nested TensorDict）。
- `VelocityVerletNVE(nn.Module)`：构造 `__init__(model, dt_fs: float, pbc: bool = True)`；`step(state) -> state` 完成 KDK；`dt > 1.0 fs` 触发 `ValueError("dt > 1 fs unsafe for water without bond constraints")`。最小镜像由 `_wrap_into_cell(pos, cell)` 工具完成。
- `VelocityVerletNVT(nn.Module)`：`__init__(model, dt_fs, thermostat)`；`step` = `thermostat.apply(B/2)` → `kick` → `drift+wrap` → `compute_forces` → `kick` → `thermostat.apply(B/2)`。
- `LangevinThermostat(nn.Module)`：`__init__(T_target_K, gamma_per_ps, seed=None)`；`apply(state, dt) -> state` 实现 BAOAB 的 O-step。`gamma_per_ps` 内部换算成 `1/fs`。
- `MDLog`（`loop.py`）：`@dataclass`，字段 `step / time_fs / e_total_eV / e_kin_eV / temperature_K / max_force_eV_per_A / nan_count / dipole_norm? / charge_sum? / quadrupole_trace?`。`append(frame_dict)` 与 `to_json(path)`。
- `run_md(model, init_state, n_steps, dt_fs, integrator, T_target=None, gamma=None, write_every, writer=None) -> MDLog`：循环调用 `integrator.step`，每 `write_every` 步把 `state` 转 `molpy.Frame` 经 `writer` 写出一帧，并 append 一条 `MDLog`。NaN 检测在每步上做（`torch.isnan(state.forces).sum()`）。
- `mdstate_to_molpy_frame` / `molpy_frame_to_mdstate`（`io.py`）：纯函数；前者只放 `element / pos / Lattice`，后者把 molpy.Frame 转回 `MDState`，由 `Z` 与 `masses` 显式注入（molpy XYZ 不解析质量）。
- `run_md_stage(manifest) -> Path`（`bench/drivers/md.py`）：装配 model + checkpoint → 取测试集首帧 → MB 抽样初速 → 跑 NVE（`manifest.md.nve_ps`）→ 跑 NVT（`manifest.md.nvt_ps`）→ 写 `trajectory.xyz` + `md_log.json`，返回 artifact 目录 Path。

生命周期与所有权：`MDState` 由调用方持有；`run_md` 不复制状态对象，仅原地更新张量；`LangevinThermostat` 的 RNG 状态作为 `nn.Module` 的 buffer 持久化，便于复现。

## Files to create or modify

- `src/molix/md/__init__.py` (new)
- `src/molix/md/state.py` (new)
- `src/molix/md/integrator.py` (new)
- `src/molix/md/loop.py` (new)
- `src/molix/md/io.py` (new)
- `src/molix/bench/drivers/__init__.py` (new)
- `src/molix/bench/drivers/md.py` (new)
- `tests/test_molix/test_md/__init__.py` (new)
- `tests/test_molix/test_md/conftest.py` (new)
- `tests/test_molix/test_md/test_integrator_nve.py` (new)
- `tests/test_molix/test_md/test_thermostat_langevin.py` (new)
- `tests/test_molix/test_md/test_io_xyz_roundtrip.py` (new)
- `tests/test_molix/test_md/test_side_tier_isolation.py` (new)
- `tests/test_molix/test_bench/test_md_driver.py` (new)
- `tests/test_molix/test_bench/test_md_bench.py` (new) — `pytest.mark.bench_md`，夜间门控。

## Tasks

- [ ] Write failing tests for `MDState` and `to_graph_batch()` (tests/test_molix/test_md/test_integrator_nve.py)
- [ ] Implement `MDState` dataclass and `to_graph_batch` in src/molix/md/state.py
- [ ] Write failing NVE harmonic-toy energy-conservation test in float64, 1000 steps, atol=1e-6 (tests/test_molix/test_md/test_integrator_nve.py)
- [ ] Implement `VelocityVerletNVE` + `_wrap_into_cell` and `dt>1 fs` guard in src/molix/md/integrator.py
- [ ] Write failing tests for `LangevinThermostat` BAOAB target-T convergence within 5 % over 50 ps on harmonic toy (tests/test_molix/test_md/test_thermostat_langevin.py)
- [ ] Implement `LangevinThermostat` and `VelocityVerletNVT` in src/molix/md/integrator.py
- [ ] Write failing tests for molpy XYZ writer/reader round-trip on a 64-water frame (tests/test_molix/test_md/test_io_xyz_roundtrip.py)
- [ ] Implement `mdstate_to_molpy_frame`, `molpy_frame_to_mdstate`, `MDLog`, and `run_md` in src/molix/md/io.py and src/molix/md/loop.py
- [ ] Write failing tests for `run_md_stage` smoke (100 NVE steps on water fixture, asserts files written + no NaN) and side-tier import isolation (tests/test_molix/test_bench/test_md_driver.py, tests/test_molix/test_md/test_side_tier_isolation.py)
- [ ] Implement `run_md_stage` manifest driver in src/molix/bench/drivers/md.py and wire `src/molix/md/__init__.py` + `src/molix/bench/drivers/__init__.py`
- [ ] Add nightly bench test (`pytest.mark.bench_md`) for 25 ps NVE + 50 ps NVT, asserting drift < 10 meV/atom/ps, NaN==0, max|F|<50 eV/Å, T within ±15 % over last 25 ps (tests/test_molix/test_bench/test_md_bench.py)
- [ ] Run full check + test suite

## Testing strategy

- **Happy path**: harmonic-toy NVE conserves energy to atol=1e-6 over 1000 steps in float64; `run_md` smoke writes `trajectory.xyz` (N frames present, `Lattice=` in comment line) and `md_log.json` (length == n_frames, all fields finite); `run_md_stage` returns the artifact path and both files exist on disk.
- **Edge cases**: `dt=1.5 fs` raises `ValueError`; XYZ round-trip preserves `Z`, `pos`, and `cell` exactly (positions float-exact within 1e-4 Å due to %.6f formatting in molpy XYZ); NaN injection into model forces is caught by `MDLog.nan_count > 0` without crashing the loop; minimum-image wrap idempotent (applying twice gives identical `pos`).
- **Side-tier isolation**: an `importlib.import_module("molix.md")` from a fresh interpreter does NOT trigger `molix.core` import; conversely, `molix.core.__init__` does not import `molix.md`. Assert with `sys.modules` introspection.
- **Domain validation**: Langevin BAOAB target-T convergence within 5 % over 50 ps on harmonic toy (analytic equipartition check); nightly Sonata-MD bar: 25 ps NVE drift < 10 meV/atom/ps, NaN==0, max|F| sustained < 50 eV/Å, NVT temperature within ±15 % over last 25 ps. Drift/force/T bars are **engineering rules of thumb, not literature-pinned for RPBE-D3 64-water**.

## Out of scope

- Nosé–Hoover-chain thermostat (Langevin BAOAB suffices for plumbing; NHC tracked for sonata-06).
- NPT / barostat (no production observable in this round needs box fluctuation; alternative considered: Berendsen for cell pre-equilibration — rejected as inconsistent with NVT-only obs stage).
- Constraint algorithms (SHAKE / RATTLE) — bare 0.5 fs Verlet is the established MLIP-MD path; constraints would require redesigning `MDState` to carry topology.
- Multi-frame initial conditions / replica exchange (single-frame init from `WaterLESSource(split="test")[0]` is the minimal validation surface).
- Writing forces / charges into the XYZ file — molpy writer doesn't support extended-XYZ Properties; diagnostics live in `md_log.json` by design.
- Distributed / multi-GPU MD (single-rank only this round).
