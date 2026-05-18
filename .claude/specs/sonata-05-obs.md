---
title: Sonata 05 — Observables sub-package (RDF + dipole ACF + multipole stability)
status: approved
created: 2026-05-11
---

# Sonata 05 — Observables sub-package (RDF + dipole ACF + multipole stability)

## Summary

为 Sonata 端到端水盒验证链路收尾，新增一个不在训练路径上的 side-tier 子包 `src/molix/observables/`，与 `molix.profiler` 同级。该子包提供三类原位可观测量计算器：周期性最小镜像 RDF、系统总偶极自相关函数、Sonata 输出的多极矩稳定性诊断。配套的 `molix.bench.drivers.obs.run_obs_stage` 流水线读取 `sonata-05-md` 产生的 `trajectory.xyz`，按帧 lazy 解析（复用 `molpy.io.trajectory.xyz.XYZTrajectoryReader` 与 `molix.datasets._extxyz.parse_comment_line` 提取 cell），按帧重建 `GraphBatch` 喂入由 `manifest.checkpoint_path` 重新加载的模型，得到逐帧 `q_perm / mu_perm / theta_perm`，输出 `observables.json`、`rdf.npz`、`dipole_acf.npz`、`multipole_stability.json` 到 `manifest.to_artifact_dir()`。Allegro 基线无多极通道时仅跑 RDF 并写入显式 "skipped" 占位。

## Domain basis

- **RDF**: Allen M. P. & Tildesley D. J., *Computer Simulation of Liquids*, 2nd ed. (Oxford, 2017), Algorithm 22。直方图归一化使用理想气体参考密度 `ρ = N / V`，归一因子 `g(r) = hist(r) / (N · ρ · 4π r² Δr)`，按对类型分别累加。周期边界采用最小镜像约定 `Δr -= L · round(Δr / L)`（cubic / orthorhombic cell）。
  - 协议参数：r-range `0.0–6.0 Å`，`n_bins = 600`（Δr = 0.01 Å）。
  - 比较锚点（非容差硬绑定，工程判断）：RPBE-D3 AIMD 水 O–O 一峰约 `2.73–2.78 Å`；实验 X-ray 参考 `~2.80 Å`，Skinner L. B. et al., *J. Chem. Phys.* 138:074506 (2013), doi:10.1063/1.4790861。

- **偶极–偶极自相关**: `C(τ) = ⟨μ_total(0) · μ_total(τ)⟩`，单位 `(e·Å)²`（当 `μ` 以 e·Å 为单位时）。系统总偶极由原子分解 `μ_total(t) = Σ_i μ_perm_i(t) + q_i(t) · r_i(t)`。窗口化平均通过 `molpy.compute.compute_acf` 复用，不在本子包内重写。
  - 物理判别意义：Cheng B., *Latent Ewald summation*, npj Comput. Mater. 11:80 (2025), doi:10.1038/s41524-025-01577-7，§III Fig. 4 — Sonata 的长程项恢复了 SR-only 模型缺失的低 k 纵向结构。论文原始判别量为伸长盒中的 `⟨|m̃_z(k)|²⟩`；本 sub-spec 实现的是 cubic-box 标准实空间 `⟨μ(0)·μ(τ)⟩`（用户显式命名的 "dipole correlation"）。Fourier 形式 `m̃_z(k)` 因需要伸长盒几何，**显式不在本轮范围内**。

- **多极矩稳定性约束（来自 chain-level brief，非文献硬钉容差，工程下界）**：
  - 净电荷漂移 `|Σ q_i| / N_atoms < 1e-4`（启用 `constrain_total_charge=True` 时仅留机器精度残差）。
  - 单原子电荷上界 `max |q_perm| < 5 e/atom`（软上界；超出即非物理 attractor 信号）。
  - 电相互作用与短程项比例 `|E_elec| / |E_short| < 1`（运行平均健康区间）。
  - 以上三项均明确标注 "engineering bound, not literature-pinned" 于代码 docstring。

## Design

side-tier 子包，**不被训练路径 import**，纯离线诊断（与 `molix.profiler` 同模式）。

- `molix.observables` 暴露三组「计算器 + Result dataclass」对，无 `Recorder` Protocol（与 `molix.io` 的 YAGNI 取舍一致：仅一个生产者时不抽象）：
  - `RDF(cutoff: float, n_bins: int, pair_types: list[tuple[int, int]])` — 内部维护 `dict[tuple[int,int], Tensor]` 直方图与帧计数。`.update(frame: molpy.core.Frame)` 累积当前帧（读取 `frame.Z`, `frame.pos`, `frame.cell`），周期最小镜像下做对距离统计。`.result() -> RDFResult` 一次性归一化并返回。
  - `DipoleDipoleACF(max_lag_ps: float, dt_ps: float, dim: str = "z")` — 持有一个 `μ_total(t)` 列表；`.update(mu_total: Tensor (3,))` 追加；`.result() -> DipoleACFResult` 内部转 `(T, 3)` Tensor 后调用 `molpy.compute.compute_acf`，返回 `(lag_ps, acf)`。`dim` 选项保留为未来 `m̃_z(k)` 升级钩子，当前默认全 3 维点积。
  - `MultipoleStability()` — 持有 `q_avg, mu_avg, theta_avg` 三个 `molpy.compute.TimeAverage`，以及每帧最大 `|q|` 与 `Σq` 标量序列。`.update(q_perm, mu_perm, theta_perm)` 推进。`.result() -> MultipoleStabilityResult` 返回 mean / std / min / max 标量与必要的直方图。

- `Result` dataclass 全部带 `.save(path: Path)`：JSON for scalar summary，`numpy.savez_compressed` for arrays。命名与 `molix.profiler.module.ModuleResult` 风格一致（`@dataclass`，不放在 `__init__.py`）。

- `molix.bench.drivers.obs.run_obs_stage(manifest: BenchManifest) -> Path` —— 单一公开入口：
  1. 用 `molix.bench.factory.model_from_manifest(manifest)` 重载模型；从 `manifest.checkpoint_path` 加载 state_dict。
  2. 用 `XYZTrajectoryReader(manifest.runs_root / manifest.system / manifest.model / "md" / "trajectory.xyz")` 流式 yield 帧；用 `parse_comment_line` 提取 `cell`。
  3. 每帧：RDF 总是 update；若 `manifest.model == "sonata"`（或更通用：模型 forward 输出含 `q_perm`/`mu_perm`/`theta_perm` 键），重建 `MDState → GraphBatch`（使用 `manifest` 描述的 cutoff/邻居列表配置）→ `model.forward(batch, compute_forces=False)` → 累计 ACF 与 stability。
  4. 写入 `manifest.to_artifact_dir() / {"observables.json","rdf.npz","dipole_acf.npz","multipole_stability.json"}`。Allegro 路径下 ACF 与 stability 产物为带 `"skipped": "model has no multipole channel"` 的占位 JSON / 空 NPZ。
  5. 返回 artifact 目录 `Path`。

所有权 / 生命周期：计算器对象在驱动函数栈帧内即建即销，不持有跨调用全局状态；模型 checkpoint 一次性加载、整轮共享；`Frame` 对象每帧释放（lazy reader 保证）。

## Files to create or modify

- `src/molix/observables/__init__.py` (new)
- `src/molix/observables/rdf.py` (new)
- `src/molix/observables/correlation.py` (new)
- `src/molix/observables/multipole_stability.py` (new)
- `src/molix/bench/drivers/obs.py` (new)
- `tests/test_molix/test_observables/__init__.py` (new)
- `tests/test_molix/test_observables/test_rdf.py` (new)
- `tests/test_molix/test_observables/test_correlation.py` (new)
- `tests/test_molix/test_observables/test_multipole_stability.py` (new)
- `tests/test_molix/test_observables/test_obs_driver.py` (new)

## Tasks

- [ ] Write failing tests for RDF on FCC lattice fixture (tests/test_molix/test_observables/test_rdf.py)
- [ ] Implement RDF + RDFResult with periodic minimum-image histogram in src/molix/observables/rdf.py
- [ ] Write failing tests for DipoleDipoleACF on synthetic exponential-decay series (tests/test_molix/test_observables/test_correlation.py)
- [ ] Implement DipoleDipoleACF + DipoleACFResult delegating to molpy.compute.compute_acf in src/molix/observables/correlation.py
- [ ] Write failing tests for MultipoleStability on deterministic q/μ/Θ fixture (tests/test_molix/test_observables/test_multipole_stability.py)
- [ ] Implement MultipoleStability + MultipoleStabilityResult via molpy.compute.TimeAverage in src/molix/observables/multipole_stability.py
- [ ] Wire up observables __init__ re-exports in src/molix/observables/__init__.py
- [ ] Write failing tests for run_obs_stage on 10-frame synthetic trajectory (tests/test_molix/test_observables/test_obs_driver.py)
- [ ] Implement run_obs_stage driver with lazy XYZ reader + per-frame inference + artifact writeback in src/molix/bench/drivers/obs.py
- [ ] Run full check + test suite

## Testing strategy

- **Happy path**: 三个计算器各自在确定性 fixture 上 update→result→save 往返一次；驱动函数在 10 帧合成轨迹（含 cell、Z、pos）上跑完 Sonata 与 Allegro 两条分支，均产出预期 artifact 集合。
- **Edge cases**:
  - RDF 在小盒子（cutoff > L/2）下抛 `ValueError`（最小镜像不再适用）；空 pair_types 抛 `ValueError`。
  - DipoleDipoleACF 在 `len(series) < max_lag/dt + 1` 时优雅降级（warn + 截断），不崩。
  - MultipoleStability 在 `N_atoms == 0` 帧上跳过（不污染 running stats）。
  - 驱动函数遇到无 `q_perm` 输出键的模型（Allegro）写显式 skipped 占位 JSON 而非崩溃。
- **Domain validation**（per `$META.science.required`）:
  - FCC-lattice RDF 应在 `r_nn`, `√2·r_nn`, `√3·r_nn` 处出现 δ-like 峰，相对峰位误差 < 1 %（解析可推导）。
  - 合成 `μ(t) = exp(-t/τ) · μ_0 + noise` 序列，恢复 `τ_recovered` 在 5 % 以内（1000 步序列）。
  - 真实 Sonata-MD 水盒运行下 O–O 一峰位置 ∈ `[2.73, 2.78] Å`（容差 ±0.05 Å，工程判断而非文献硬钉）。
  - Sonata 与 Allegro 同轨迹运行下 dipole ACF 在 τ=1 ps 处可区分（差异 > 2× 噪声地板）— Cheng 2025 Fig 4 的负载承重物理主张。
  - Sonata 运行的多极稳定性约束（`|Σq|/N < 1e-4`、`max|q| < 5e`、`|E_elec|/|E_short| < 1`）整轮成立。

## Out of scope

- 伸长盒 Fourier-空间 `⟨|m̃_z(k)|²⟩` 判别（需独立伸长盒数据集，单独 sub-spec）。
- 角度相关 RDF / 三体相关函数。
- 红外谱 / 介电常数频域提取（基于 ACF 但需额外加窗 + FFT 协议）。
- Non-orthorhombic（triclinic）cell 的最小镜像（本轮仅支持 cubic / orthorhombic）。
- 在线（训练中）观测量 hook —— 本子包明确为离线 side-tier。
