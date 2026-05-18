---
title: Sonata 05 训练驱动 — manifest-driven 端到端训练 (bulk water)
status: approved
created: 2026-05-11
---

# Sonata 05 训练驱动 — manifest-driven 端到端训练 (bulk water)

## Summary

为 Sonata HPC 端到端验证链（`sonata-05-*`）落地训练阶段驱动 `run_train(manifest: BenchManifest) -> Path`，置于 `src/molix/bench/drivers/train.py`。驱动从 `BenchManifest`（由 sub-spec `sonata-05-bench` 提供）出发，依次完成：用 `model_from_manifest` 构造模型（`model="allegro"` 与 `model="sonata"` 对称）、用 `WaterLESSource`（由 sub-spec `sonata-05-data` 提供）构造 train/val/test 三个数据子集并经 `Pipeline → NeighborList(symmetry=True) → PackedCache → MmapDataset → DataModule` 流水线落盘缓存、按 manifest 装配优化器（`Adam(lr=hyperparams.lr)`）与损失（`molix.core.losses.energy_force_mse`）、注册一组 hooks（`MetricsHook` 训练/评估各持深拷贝实例、`CheckpointHook` 监控 `eval/force_rmse` 最小化、可选 `ScalarHook`/`TensorBoardHook`、Sonata 分支额外注册新的 `MultipoleDiagnosticHook`）、调用 `Trainer.train()`，最后在 test split 上跑一遍评估并把最终标量写入 `<artifact_dir>/metrics.json`，返回 `artifact_dir`。驱动通过 `wrap_for_ddp`（由 `sonata-05-hpc` 提供）兼容 `torchrun --nproc_per_node=N`：`WORLD_SIZE>1` 时模型在喂给 `Trainer` 前先包成 DDP；rank-0 守卫由现有的 `CheckpointHook` / `JournalHook` / `TensorBoardHook` 自带承担。`MultipoleDiagnosticHook` 是只读 hook，写 `state["train"][…]` 本地标量，**不**做跨 rank all-reduce（与 `sonata-05-hpc` § Out-of-scope 的 eval-reduction 策略一致）。smoke 模式必须能在 4-water/100-step 的 micro fixture 上无网络运行，覆盖 CI。

## Domain basis

驱动本身是工程编排（trainer/optimizer/hook 装配），物理验收门由 `MultipoleDiagnosticHook` 与 test-set metrics 共同承担。基线与判据来自 scientist 输出，逐字保留：

- **Allegro force-RMSE 基线（RPBE-D3 bulk water）**：≈ 32.1 meV·Å⁻¹（*J. Chem. Phys.* **163**:104102, 2025，<https://pubs.aip.org/aip/jcp/article/163/10/104102/3361757/>）；本规范的 `performance` 类验收使用 "Sonata force RMSE ≤ baseline × 1.20" —— **20 % 容差为工程判断，非文献锚定**。
- **能量 MAE 软上界**：Allegro 级 MLIP 在 bulk water 上典型 ≲ 1 meV/atom；本规范作 sanity check，不作硬门。
- **q_perm / μ_perm / Θ_perm 健康范围**：Sonata 的 `q_perm` 是从能量/力梯度训练出的*潜变量*，**不要求**与 Bader 参考电荷（|q_O| ≈ 1.0–1.2 e，|q_H| ≈ 0.5–0.6 e）一致；坍缩判据 `max|q_perm| < 5 e` 是工程判断，**非文献锚定**。
- **净电荷漂移**：Sonata 配 `constrain_total_charge=True` 时 `|Σ qᵢ| / N_atoms < 1e-4 e`（float32 求和漂移上界，**非文献锚定**）。
- **E_elec / E_short 比值**：Cheng 2025 §III 指出长程贡献在 bulk water 总键能中占比小，健康范围 `|E_elec|/|E_short| ≪ 1`；本规范取 `< 1` 作为非崩溃门，**非文献锚定确切比值**。

参考文献：

- Cheng B., *Latent Ewald summation for machine-learning potentials*, **npj Comput. Mater.** 11:80 (2025), doi:10.1038/s41524-025-01577-7.
- *J. Chem. Phys.* **163**:104102 (2025), <https://pubs.aip.org/aip/jcp/article/163/10/104102/3361757/>（Allegro-class MLIP on RPBE-D3 liquid water；force-RMSE 基线来源）。

单位：能量 eV、力 eV·Å⁻¹、长度 Å；`metrics.json` 中以 meV 与 meV·Å⁻¹ 落盘（×1000 转换在 `run_train` 末段完成）。

## Design

新增子包 `src/molix/bench/drivers/`，对外只暴露 `run_train`：

```
src/molix/bench/drivers/__init__.py          # re-export run_train
src/molix/bench/drivers/train.py             # run_train + 内部 helpers
src/molix/hooks/multipole_diagnostic.py      # MultipoleDiagnosticHook
```

`run_train(manifest: BenchManifest) -> Path` 的执行顺序：

1. **路径与日志**：`artifact_dir = manifest.to_artifact_dir()`；`mkdir -p artifact_dir/checkpoints`；rank-0 守卫由 `CheckpointHook.on_train_start` 内置承担（`sonata-05-hpc` 已落地）。
2. **模型**：`model = model_from_manifest(manifest)`；分支选择透明走 sub-spec 02 的工厂，驱动本身对两个分支对称。
3. **数据流水线**（每个 split 一份）：
   - `source = WaterLESSource(root=manifest.dataset_root, split=split)`，`split ∈ {"train","val","test"}`。
   - `cache = Pipeline().add(NeighborList(cutoff=manifest.hyperparams["r_max"], symmetry=True)).build().cache(source, cache_dir=artifact_dir / "cache" / split)`，返回 `PackedCache`。
   - `dataset = MmapDataset(cache)`（**遵循 `feedback_dataset_base_class.md` —— DataModule 必须接 `dataset_cls=MmapDataset` 而不能 hard-code `CachedDataset`**）。
   - 三个 `dataset` 喂入 `DataModule(dataset_cls=MmapDataset, train=..., val=..., test=..., batch_size=manifest.hyperparams.get("batch_size", 4), collate_fn=collate_molecules)`。
4. **DDP wrap**（**先于** Trainer 构造）：`model, dist_env = wrap_for_ddp(model, device=manifest.device)`；`WORLD_SIZE=1` 时 `wrap_for_ddp` 走 single-process fast path。
5. **Hooks 装配**：
   - `MetricsHook(metrics={"energy_mae","force_rmse"})` —— 训练侧与评估侧各一份独立实例（用 `copy.deepcopy` 复制，符合 CLAUDE.md § State namespace contract 第 4 条 "Hooks must not share mutable buffers between train and val"）。
   - `CheckpointHook(dirpath=artifact_dir / "checkpoints", monitor="eval/force_rmse", mode="min", save_top_k=1, save_last=True)`。
   - 若 `manifest.hyperparams.get("tensorboard", False)`：注册 `TensorBoardHook(log_dir=artifact_dir / "tb")`；smoke 默认关。
   - 若 `manifest.model == "sonata"`：额外注册 `MultipoleDiagnosticHook(window_frac=0.10)` —— 见下文。
6. **优化器与 trainer**：`optimizer_factory = lambda p: torch.optim.Adam(p, lr=manifest.hyperparams.get("lr", 1e-3))`；`trainer = Trainer(model=model, loss_fn=energy_force_mse, optimizer_factory=optimizer_factory, hooks=[...], device=manifest.device); trainer.train(datamodule, max_epochs=manifest.hyperparams["max_epochs"])`。
7. **Test-set 评估**：`trainer._run_eval_phase(test_datamodule)` 复用现有 eval 路径写入 `state["eval"][...]`；驱动从 `state` 拿 `energy_mae` / `force_rmse` 并额外计算 long-range slice 上的 force RMSE（mask `bond_dist > r_max` 的远场原子，**只在 sonata 分支**计算 `force_rmse_long_range`，allegro 分支这一项写 `None`）。
8. **写盘 + 返回**（rank-0 守卫）：在 `is_rank_zero()` 下 `json.dump({"energy_mae_meV_per_atom": ..., "force_rmse_meV_per_A": ..., "force_rmse_long_range_meV_per_A": ...}, artifact_dir / "metrics.json")`；调用 `teardown_ddp(dist_env)`；返回 `artifact_dir`。

`MultipoleDiagnosticHook`（新文件 `src/molix/hooks/multipole_diagnostic.py`）：

- 形状参照 `GPUMemoryHook`（`src/molix/hooks/gpu.py:11-81`）；继承 `ScalarHook`；`scalar_keys` 返回固定的 `(("train","q_perm_mean"), ("train","q_perm_max_abs"), ("train","q_net_abs_per_atom"), ("train","mu_perm_mean"), ("train","mu_perm_max_abs"), ("train","theta_perm_mean"), ("train","theta_perm_max_abs"), ("train","E_short"), ("train","E_elec"), ("train","E_elec_over_short"))`。
- `on_train_batch_end(trainer, state, batch, outputs)`：仅在 `outputs is not None and ("atoms","q_perm") in batch` 时执行；从 `batch[("atoms","q_perm")]`（shape `(N,)`）/ `batch[("atoms","mu_perm")]`（shape `(N,3)`）/ `batch[("atoms","theta_perm")]`（shape `(N,3,3)`）/ `batch[("graphs","E_short")]` / `batch[("graphs","E_elec")]` 读取张量，全部 `.detach()` 后计算标量并写入 `state["train"][…]`。
- 窗口聚合策略：hook 内部用环形 buffer 保存最后 `window_frac × total_steps` 步的 per-step 标量；`on_train_end` 把窗口均值复制到 `state["train"]["q_perm_mean_window"]` 等键，验收准则在窗口均值上断言。
- **不**做 `all_reduce`；每个 rank 的标量为本 rank 局部值，与 `sonata-05-hpc` § Out-of-scope（eval-reduction 策略）一致；hook 文档段标注 "rank-local — no all-reduce in this round"。

数据契约（与 CLAUDE.md § Two-tier data contract 对齐）：post-collate `GraphBatch` 嵌套；hook 用 tuple-key 读 `batch[("atoms","q_perm")]`；驱动级 source/sample 读写仍用 flat dict。

## Files to create or modify

- (new) `src/molix/bench/__init__.py` — 占位 namespace（若不存在）。
- (new) `src/molix/bench/drivers/__init__.py` — `from .train import run_train`；`__all__ = ["run_train"]`。
- (new) `src/molix/bench/drivers/train.py` — `run_train(manifest) -> Path` 主体 + `_build_datamodule` / `_build_hooks` / `_write_metrics_json` 三个内部 helpers。
- (new) `src/molix/hooks/multipole_diagnostic.py` — `MultipoleDiagnosticHook` 实现，形状参照 `src/molix/hooks/gpu.py`。
- `src/molix/hooks/__init__.py` — append `from .multipole_diagnostic import MultipoleDiagnosticHook` 并扩展 `__all__`。
- (new) `tests/test_molix/test_bench/__init__.py`。
- (new) `tests/test_molix/test_bench/test_drivers_train.py` — `run_train` 单元测试（smoke 模式，allegro & sonata 两分支，4-water fixture，100 步，asserts `metrics.json` 落盘 + 返回 Path）。
- (new) `tests/test_molix/test_hooks/test_multipole_diagnostic.py` — `MultipoleDiagnosticHook` 行为单元测试（scalar_keys 形状、`state["train"]` 写入、窗口均值、batch 缺键时 graceful skip）。
- (new) `tests/test_molix/test_bench/conftest.py` — manifest fixture（指向 sub-spec 1 的 micro extxyz 与 sub-spec 2 的 manifest pydantic 构造器）。

## Tasks

- [ ] Write failing tests for MultipoleDiagnosticHook scalar contract in tests/test_molix/test_hooks/test_multipole_diagnostic.py
- [ ] Implement MultipoleDiagnosticHook in src/molix/hooks/multipole_diagnostic.py and export from src/molix/hooks/__init__.py
- [ ] Write failing tests for run_train smoke mode (allegro + sonata branches, 4-water fixture, 100 steps, metrics.json existence and key set) in tests/test_molix/test_bench/test_drivers_train.py
- [ ] Implement run_train in src/molix/bench/drivers/train.py with model_from_manifest + Pipeline+NeighborList+PackedCache+MmapDataset+DataModule wiring + wrap_for_ddp + Trainer.train + test-eval + metrics.json write
- [ ] Wire src/molix/bench/__init__.py and src/molix/bench/drivers/__init__.py to re-export run_train
- [ ] Add Google-style docstrings with units to run_train and MultipoleDiagnosticHook (Å, eV, eV·Å⁻¹, meV, meV·Å⁻¹) and tensor shapes per CLAUDE.md § Docstring Convention
- [ ] Verify Sonata test-set force-RMSE ≤ 1.20 × 32.1 meV·Å⁻¹ baseline on the bulk-water full-mode run (manifest hyperparams supplied by the implementer; not part of CI smoke)
- [ ] Run full check + test suite (ruff check src/ && ruff format --check src/ && python -m pytest tests/ -v)

## Testing strategy

- **happy path (unit, CI)**: `MultipoleDiagnosticHook.on_train_batch_end` 在一个手工构造的 `GraphBatch`（含 `("atoms","q_perm")` 等 5 张量）下写入十个 `state["train"][…]` 键；`scalar_keys` 返回严格的元组列表（顺序稳定，可被 `ScalarHook` 框架消费）。
- **happy path (unit, CI)**: `run_train` 用 sub-spec 1 的 4-water micro fixture + `model="allegro"` smoke manifest（`max_epochs=1`, `max_steps=100`, `batch_size=2`, `lr=1e-3`）能跑通；返回值是 `Path`，`(artifact_dir / "metrics.json").exists()` 为真，JSON 含 `{energy_mae_meV_per_atom, force_rmse_meV_per_A, force_rmse_long_range_meV_per_A}` 三键；`force_rmse_long_range_meV_per_A` 在 allegro 分支为 `None`。
- **happy path (unit, CI)**: 同上换 `model="sonata"`；额外断言 `MultipoleDiagnosticHook` 在 `state["train"]` 写下十个键；`force_rmse_long_range_meV_per_A` 为 float。
- **edge case**: `WORLD_SIZE` 未设 / `="1"` 时 `run_train` 走 single-process fast path，不调 `init_process_group`（monkeypatch 验证）。
- **edge case**: hook 在 batch **缺少** `("atoms","q_perm")` 键（如 allegro 分支）时 graceful skip，**不**抛 `KeyError`、**不**写 `state["train"][…]` 中的多极相关键。
- **edge case**: `MultipoleDiagnosticHook` 的窗口 buffer 在 step 数小于 `window_frac × total_steps` 时仍能给出有限值（不抛 `ZeroDivisionError` 或 `nan`）。
- **edge case**: hook 在 `compute_forces=True` 路径下读取的张量已 `.detach()`，不污染 autograd graph（断言 hook 调用前后 `q_perm.requires_grad` 与 `q_perm.grad_fn` 等价）。
- **domain validation**: 在一个 50-step run 末段（`MultipoleDiagnosticHook.on_train_end` 后）断言：(a) `state["train"]["q_net_abs_per_atom"] < 1e-4`；(b) `state["train"]["q_perm_max_abs"] < 5.0`；(c) `state["train"]["E_elec_over_short"] < 1.0`；三个都是窗口均值。
- **domain validation (full mode, off-CI)**: 在真实 bulk water RPBE-D3 数据上完成完整训练后，test-split 上 `force_rmse_meV_per_A ≤ 32.1 × 1.20`（Allegro 基线 + 20 % 容差，scientist 输出）；该判据为 `performance` 类验收。
- **integration (CI)**: 全套 `ruff check src/ && ruff format --check src/ && python -m pytest tests/ -v` 绿。

## Out of scope

- `BenchManifest` Pydantic 类与 `model_from_manifest` 工厂 —— 属 sub-spec 2（`sonata-05-bench`）；本规范只**调用**它们。
- `WaterLESSource` 实现 —— 属 sub-spec 1（`sonata-05-data`）。
- `wrap_for_ddp` / rank-0 hook 守卫 —— 已落地（`sonata-05-hpc`）。
- MD 驱动 / 观测量驱动 —— 属 sub-spec 4 / 5（`sonata-05-md` / `sonata-05-obs`）。
- 多节点 DDP rendezvous / SLURM 模板 —— 不进本仓库（同 `sonata-05-hpc` § Out-of-scope）。
- Eval 阶段跨 rank `all_reduce` 聚合 —— 与 `sonata-05-hpc` 一致；本规范的 hook 只写 rank-local 标量。
- ChargedDimer 数据路径 —— bulk water-only chain；如未来纳入则独立 spec。
- `bm_sonata.py`（既有 sonata-03-bench 入口）的弃用 —— legacy 入口保留，本驱动是 manifest-driven 后继；用户切换由 release notes 注释。
- 训练侧 metric all-reduce —— 与 eval 同政策；rank-local 即可，HPC 端到端用户自行 reduce。
- `force_rmse_long_range` 在 allegro 分支的定义 —— allegro 无 `E_elec` 分量，long-range slice 的 force RMSE 无意义，统一记 `None`；如未来需要短程 baseline 的远场切片，独立 spec。
- 超参扫描 —— 一次 `run_train` 一个 manifest；扫描由调用方循环组织。
- 早停（early stopping）—— `CheckpointHook(monitor=..., mode="min")` 自带 best-tracker 已够 sonata-05-md 消费；如未来需要 patience-based stop，独立 hook。
