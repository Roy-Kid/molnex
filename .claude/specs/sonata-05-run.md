---
title: Sonata 05 — bulk-water Sonata 训练入口 (work/ 脚本 + Alvis A100 sbatch)
status: done
created: 2026-05-11
supersedes: [sonata-05-bench, sonata-05-hpc]
---

# Sonata 05 — bulk-water Sonata 训练入口 (work/ 脚本 + Alvis A100 sbatch)

## Summary

合并旧 `sonata-05-bench`（manifest + factory 工程化基础设施）与 `sonata-05-hpc`（first-class DDP shim + rank-0 守卫），废弃二者的 in-tree 工程化路线。Bulk water RPBE-D3 上的 Sonata 训练不再以"基础组件 + driver"的方式落到 `src/molix/bench/` 与 `src/molix/core/ddp.py`，而是在 `work/sonata_bulk_water/` 下落地两个独立脚本：

- `work/sonata_bulk_water/train.py` —— 端到端训练入口，单 A100 单进程；超参按 Cheng B. 2025 *npj Comput. Mater.* 11:80 论文及既有 `benchmarks/bm_molpot/bm_sonata.py` 默认值。
- `work/sonata_bulk_water/submit.py` —— Python 模板渲染器，向 Alvis (C3SE) `alvis` 分区 + A100 GPU 提交 sbatch。

`src/` 与 `tests/` **不**收任何新代码。本规范不引入 `BenchManifest`、`model_from_manifest`、`wrap_for_ddp` 等抽象；亦不审计 `CheckpointHook` / `JournalHook` / `TensorBoardHook` 的 rank-0 守卫（无 DDP 不需要）。NaN 早停由 `train.py` 内联一个 `NaNStopHook` 实现，靠 `raise RuntimeError` 终止 `Trainer._train` 并以非零退出码退给 SLURM。

## Domain basis

工程编排 + 调度集成性质，不引入新的物理。论文超参 / 基线引用：

- Cheng B., *Latent Ewald summation for machine-learning potentials*, **npj Comput. Mater.** 11:80 (2025), doi:10.1038/s41524-025-01577-7. —— Sonata 的 σ-高斯电荷弥散 + multipole-Ewald 长程；本规范 `SIGMA=1.0 Å`, `DL=2.0 Å` 即论文 §III 默认。
- *J. Chem. Phys.* **163**:104102 (2025), <https://pubs.aip.org/aip/jcp/article/163/10/104102/> —— Allegro-class MLIP on RPBE-D3 liquid water；force-RMSE 基线 ≈ 32.1 meV·Å⁻¹（论文表 1）。
- `benchmarks/bm_molpot/bm_sonata.py:51-95` —— Allegro encoder 默认值（`r_max=5.0`, `l_max=2`, `num_scalar_features=64`, `num_tensor_features=16`, `num_layers=2`, `num_elements=20`, `avg_num_neighbors=12.0`, `type_embed_dim=32`, `latent_mlp_width=64`, `num_bessel=8`），与论文 §IV 对齐，本规范沿用。

Alvis 调度参考：
- C3SE Alvis 用户手册 <https://www.c3se.chalmers.se/documentation/for_users/intro-alvis/> ——`-p alvis`、`--gpus-per-node=A100:1` (40 GB) 或 `A100fat:1` (80 GB)、`-A NAISS<YYYY>-<X>-<Z>` 项目账户。

单位约定：能量 eV、力 eV·Å⁻¹、长度 Å；`metrics.json` 中以 meV / meV·Å⁻¹ 落盘。

## Design

### `work/sonata_bulk_water/train.py`

调用形态：
```bash
python -u work/sonata_bulk_water/train.py \
    --data-root /mimer/.../water_les \
    --out-dir runs/water_rpbe_d3/sonata \
    --max-epochs 100 \
    --batch-size 4 \
    --lr 1e-3 \
    --seed 0
```

执行顺序：

1. **参数与目录**：argparse 解析；`out_dir = Path(args.out_dir).resolve()`；`out_dir.mkdir(parents=True, exist_ok=True)`。
2. **logging 配置**：根 logger `INFO` 级，handler 同时写 stderr 与 `out_dir/train.log`。格式 `"%(asctime)s %(levelname)s %(name)s: %(message)s"`。`torch.manual_seed(args.seed)`。
3. **数据**：每个 split：
   - `source = WaterLESSource(root=args.data_root, split=split)`，`split ∈ {"train","val","test"}`。
   - `pipe = Pipeline(f"water-les-{split}").add(NeighborList(cutoff=R_MAX, max_num_pairs=4096, pbc=True, symmetry=True)).build()`。
   - `packed = pipe.cache(source, base_dir=out_dir / "cache" / split)` → `PackedCache`。
   - `dataset = MmapDataset(packed.sink)` —— **遵循 `feedback_dataset_base_class`：`MmapDataset` 优于 `CachedDataset`；DataModule 接受预构造 dataset，不接受 `dataset_cls=` kwarg**。
   - 训练用 `DataModule(train_ds, val_ds, target_schema=WaterLESSource.TARGET_SCHEMA, batch_size=args.batch_size, num_workers=args.num_workers)`；测试集另起 `test_dm = DataModule(test_ds, test_ds, …)` 以避免污染主训练循环的 `state["eval"]` 历史。
4. **模型**：`_build_sonata(seed=args.seed)` 内联：
   ```python
   torch.manual_seed(seed)
   encoder = Allegro(num_elements=NUM_ELEMENTS, num_scalar_features=NUM_FEATURES,
                     num_tensor_features=NUM_FEATURES // 4, r_max=R_MAX, num_bessel=8,
                     l_max=L_MAX, num_layers=NUM_LAYERS, type_embed_dim=TYPE_EMBED_DIM,
                     latent_mlp_depth=1, latent_mlp_width=LATENT_MLP_WIDTH,
                     avg_num_neighbors=AVG_NUM_NEIGHBORS, expose_tensor_track=True)
   torch.manual_seed(seed + 1)
   short_head = EdgeEnergyHead(input_dim=encoder.output_dim, hidden_dim=128,
                               avg_num_neighbors=AVG_NUM_NEIGHBORS, out_key="energy_short")
   model = build_sonata(encoder, sigma=SIGMA, dl=DL, charge=True, dipole=True,
                        quadrupole=True, constrain_total_charge=True,
                        avg_num_neighbors=AVG_NUM_NEIGHBORS, short_range_head=short_head)
   ```
   常量（`R_MAX`, `L_MAX`, `NUM_FEATURES`, `NUM_LAYERS`, `SIGMA`, `DL`, …）在模块顶层显式赋值，ac-005 的 grep 锚点。
5. **优化器 + 损失**：`optimizer_factory = lambda p: torch.optim.Adam(p, lr=args.lr)`；损失 `molix.core.losses.energy_force_mse()`（与 `bm_sonata.py` 同款；返回 callable）。
6. **自定义 Step**：`Sonata.forward` 用 `torch.autograd.grad` 算力，与默认 `DefaultEvalStep` 的 `torch.no_grad()` 包裹不兼容。`train.py` 内联两个轻量 Step：
   - `_SonataTrainStep`：`forward → loss → backward → optimizer.step()`，调用 `model(batch, compute_forces=True)`。
   - `_SonataEvalStep`：`with torch.enable_grad(): model(batch, compute_forces=True)`，保持 autograd 图活到 force 求导。
7. **Hooks**（注册顺序如下）：
   - `MetricsHook(metrics=[EnergyMAE()], pred_key=("energy",), target_key=("graphs","energy"))` —— 训练 / 评估各 `copy.deepcopy` 独立实例（MetricsHook 内部自动），写入 `state["train"]["EnergyMAE"]` / `state["eval"]["EnergyMAE"]`。
   - `MetricsHook(metrics=[ForceRMSE()], pred_key=("forces",), target_key=("atoms","forces"))` —— `state["train"]["ForceRMSE"]` / `state["eval"]["ForceRMSE"]`。`EnergyMAE` / `ForceRMSE` 是 `MAE` / `RMSE` 的 trivial subclass，仅为了在 state 中获得不冲突的 class-name key。
   - **`NaNStopHook`**（本文件内联）：注册位置在 `MetricsHook` 之后、`CheckpointHook` 之前。`on_train_batch_end` 检查 `state["train"]["loss"]` 与每个 model parameter；非有限即 `torch.save(model.state_dict(), out_dir / "nan_checkpoint.pt")` + `raise RuntimeError("NaN detected …")`。
   - `CheckpointHook(checkpoint_dir=str(out_dir/"checkpoints"), save_last=True, save_best=True, best_metric_name=("eval","ForceRMSE"), best_metric_mode="min")`。
   - `TensorBoardHook(every_n_steps=10, log_dir=str(out_dir / "tb"))` —— 默认开启，监视 train/eval 标量与 lr。
   - `JournalHook(every_n_steps=10, store=JournalWriter(out_dir / "journal", run_id="train"))` —— `JournalWriter` 是 Zarr v3 后端（**不**是 jsonl），shard 设计避免 HPC inode 爆炸。
   - `ProgressBarHook(desc="Sonata")` —— 终端进度。
   - 可选 `_DebugNaNInjectorHook(at_step=2)` —— `--debug-inject-nan` flag 启用，仅 ac-002 验证用。
8. **Trainer**：
   ```python
   trainer = Trainer(model=model, loss_fn=energy_force_mse(),
                     optimizer_factory=lambda p: torch.optim.Adam(p, lr=args.lr),
                     train_step=_SonataTrainStep(), eval_step=_SonataEvalStep(),
                     hooks=hooks,
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
   try:
       trainer.train(dm, max_epochs=args.max_epochs)
   except RuntimeError as e:
       if "NaN detected" in str(e):
           logger.error("training aborted by NaNStopHook: %s", e)
           return 2
       raise
   ```
9. **Test 评估 + metrics.json**：训练结束后手动遍历 `test_dm.val_dataloader()`（test_dm 把 test_ds 同时绑到 train/val 槽），对每个 batch 在 `torch.enable_grad()` 下调用 `model(batch, compute_forces=True)`，累加 `|E_pred − E_ref|` 与 `(F_pred − F_ref)²`，最终：
   - `energy_mae_meV_per_atom = EV_TO_MEV * Σ|ΔE| / Σ n_atoms`
   - `force_rmse_meV_per_A = EV_TO_MEV * sqrt(Σ(ΔF)² / (3 · Σ n_atoms))`
   - `json.dump({"energy_mae_meV_per_atom": …, "force_rmse_meV_per_A": …}, out_dir/"metrics.json")`。

不做 DDP、不做 `wrap_for_ddp`、不做 rank-0 守卫——单 A100 单进程足够覆盖论文复现规模。

### `work/sonata_bulk_water/submit.py`

调用形态：
```bash
python work/sonata_bulk_water/submit.py \
    --account NAISS2025-X-YYY \
    --time 24:00:00 \
    --data-root /mimer/.../water_les \
    --out-dir $HOME/runs/sonata_water_$(date +%Y%m%d) \
    --max-epochs 100 \
    --batch-size 4 \
    --gpu A100        # or A100fat for 80 GB
    [--dry-run]
```

行为：
1. argparse 收集 `--account / --time / --gpu / --partition / --job-name / --data-root / --out-dir / --max-epochs / --batch-size / --lr / --seed / --dry-run / --module / --venv`。`--module` 与 `--venv` 各自有 sensible 默认占位符（`PyTorch/2.6.0-foss-2024a-CUDA-12.1.1` 与 `$HOME/portfolio/venvs/molnex`）；README 里要求用户首次运行前在登录节点 `module avail PyTorch` 校正默认值。
2. 通过 `string.Template` 渲染如下 sbatch 文本（占位符值由 argparse 注入）：
   ```bash
   #!/bin/bash
   #SBATCH -A ${ACCOUNT}
   #SBATCH -p ${PARTITION}              # default: alvis
   #SBATCH --gpus-per-node=${GPU}:1     # default: A100:1
   #SBATCH -t ${TIME}
   #SBATCH -J ${JOB_NAME}               # default: sonata_water
   #SBATCH -o ${OUT_DIR}/slurm-%j.out
   #SBATCH -e ${OUT_DIR}/slurm-%j.err

   set -euo pipefail
   module purge
   module load ${MODULE}
   source ${VENV}/bin/activate

   cd ${REPO_ROOT}
   python -u work/sonata_bulk_water/train.py \
       --data-root ${DATA_ROOT} \
       --out-dir ${OUT_DIR} \
       --max-epochs ${MAX_EPOCHS} \
       --batch-size ${BATCH_SIZE} \
       --lr ${LR} \
       --seed ${SEED}
   ```
3. `--dry-run`：把渲染后文本写到 stdout，**不**调用 sbatch；返回 0。
4. 非 `--dry-run`：写 `${OUT_DIR}/submit.sbatch`（先 `mkdir -p out_dir`），然后 `subprocess.run(["sbatch", str(submit_sbatch)], check=True, capture_output=True, text=True)`，把 stdout 中的 `"Submitted batch job <int>"` 解析出 jobid，写到 `${OUT_DIR}/jobid.txt` 并 print 给用户。`subprocess.CalledProcessError` 直接向上传播——本脚本不做重试。

`submit.py` 仅是模板渲染器；不解析 squeue、不做依赖图、不做多 job 编排。多次运行就是多次手动调用。

### `work/sonata_bulk_water/README.md`

- 调用范式（local smoke + Alvis 提交各一条命令）。
- Alvis 注意事项：`module avail PyTorch` 列出可用 module；建议在 `~/portfolio/venvs/molnex` 下建 venv；`--gpu A100`（40 GB）vs `A100fat`（80 GB）按内存预算选；`-t` 取 24 h 上限内的实际预估。
- 论文超参溯源（5 行）：`R_MAX=5.0`, `L_MAX=2`, `NUM_FEATURES=64`, `NUM_LAYERS=2`, `SIGMA=1.0`, `DL=2.0` —— Cheng 2025 §IV 默认 + `bm_sonata.py:51-95`。
- NaN 处理语义：进程退出码 `2`；`nan_checkpoint.pt` 落盘；`train.log` 末段含 `"NaN detected"` 行。

## Files to create or modify

- (new) `work/sonata_bulk_water/train.py` —— 端到端训练入口（含内联 `NaNStopHook`、`_SonataTrainStep`、`_SonataEvalStep`、`EnergyMAE`/`ForceRMSE` 子类、`_DebugNaNInjectorHook`）。
- (new) `work/sonata_bulk_water/submit.py` —— Alvis sbatch 渲染器 + `subprocess.run(["sbatch", …])`。
- (new) `work/sonata_bulk_water/README.md` —— 调用范式 + Alvis 注意事项 + 论文超参溯源 + NaN 退出码语义。
- (delete) `.claude/specs/sonata-05-bench.md` —— 被本规范取代。
- (delete) `.claude/specs/sonata-05-bench.acceptance.md` —— 同上。
- (delete) `.claude/specs/sonata-05-hpc.md` —— 同上。
- (delete) `.claude/specs/sonata-05-hpc.acceptance.md` —— 同上。

**不**修改 `src/`，**不**添加 `tests/` 用例（`work/` 不纳入 pytest 集合；用户在 PR 描述中说明手动 smoke 结果）。

## Tasks

- [x] 删除 `.claude/specs/sonata-05-bench.{md,acceptance.md}` 与 `.claude/specs/sonata-05-hpc.{md,acceptance.md}`（已在工作树暂存为 `git rm`）。
- [x] 写 `work/sonata_bulk_water/train.py`：argparse + logging + 模型 + 数据 + custom Steps + hooks（含 inline `NaNStopHook`）+ Trainer + test eval + `metrics.json`。
- [x] 写 `work/sonata_bulk_water/submit.py`：sbatch 模板渲染 + dry-run + 真实 `sbatch` 投递。
- [x] 写 `work/sonata_bulk_water/README.md`：调用范式、Alvis module 选择、论文超参溯源、NaN 退出语义。
- [x] 本地 smoke 验证（`--max-epochs 1 --batch-size 2`）：`metrics.json` / `tb/` / `journal/` / `checkpoints/last.pt` 落盘已验证（ac-001、ac-002 verified）。
- [x] Hygiene/simplify gate：diff 内联检查通过——无 dead code、无 magic numbers（论文超参全部 named constant）、无 commented-out 残留、无 backward-compat shim（新规范，无 legacy 兼容路径）；`--debug-inject-nan` + `_DebugNaNInjectorHook` 是 spec 内明示保留的 debug 路径，README 已声明生产中不传。
- [x] 在 Alvis 上交一次 `-t 00:30:00` 的小作业（2 epochs）验证 sbatch 实链路 + A100 资源拿到。**runtime — ac-004 verified 2026-05-11；NAISS2026-4-715 账户 + 合成 fixture，jobid 6614572 入队。**

## Testing strategy

`work/` 不进 CI；以下均为手动验证，结果记录在 PR 描述中：

- **本地 smoke**：`python -u work/sonata_bulk_water/train.py --data-root <small fixture> --out-dir /tmp/sonata_smoke --max-epochs 1 --batch-size 2 --lr 1e-3 --seed 0` —— 期望退出码 0；四份产物（`metrics.json`、`tb/`、`journal/`（Zarr 目录）、`checkpoints/last.pt`）全部存在；`metrics.json` 键集为 `{energy_mae_meV_per_atom, force_rmse_meV_per_A}`。
- **NaN early-stop**：`python -u work/sonata_bulk_water/train.py … --debug-inject-nan`：期望 (a) 退出码 `2`；(b) `out_dir/nan_checkpoint.pt` 存在且 `torch.load` 成功；(c) `train.log` 末段含 `"NaN detected"` 行。`--debug-inject-nan` 永久保留为 debug flag（debug-only `_DebugNaNInjectorHook`），README 明示生产中不传。
- **submit.py dry-run**：`python work/sonata_bulk_water/submit.py --account TEST-X-Y --time 00:30:00 --data-root /tmp/x --out-dir /tmp/y --dry-run` —— stdout 含 `"#SBATCH -p alvis"`、`"#SBATCH --gpus-per-node=A100:1"`、`"#SBATCH -t 00:30:00"`，且 `sbatch` 真实调用未发生（开发机不一定有 sbatch 二进制；dry-run 路径与 PATH 解耦）。
- **submit.py 真投递**：在 Alvis 登录节点上 `python work/sonata_bulk_water/submit.py --account <real> --time 00:30:00 --data-root <real> --out-dir <real> --max-epochs 2 --batch-size 2` —— 期望 `${OUT_DIR}/jobid.txt` 存在且内容为整数；`squeue -u $USER` 列出该 jobid；作业完成后 `metrics.json` 落盘。

## Out of scope

- 任何 `src/molix/bench/*`、`src/molix/core/ddp.py` 代码 —— 被本规范明示废弃。
- 多 GPU / 多节点 / DDP / FSDP —— 单 A100 即可复现论文；如未来需要，独立 spec（`sonata-05-ddp-run` 等）。
- Hydra / OmegaConf 配置语法 —— argparse + 论文默认值足够；`bm_sonata.py` 也用 argparse。
- 超参扫描 —— 一次 `submit.py` 一个作业；扫描由用户外挂 bash 循环。
- Rank-0 写盘守卫 —— 无 DDP 不需要；现有 hook 行为保持不变。
- MD 阶段 / observable 阶段 —— 由 `sonata-05-md` / `sonata-05-obs` 覆盖。
- `WaterLESSource` 实现 —— 由 `sonata-05-data` 覆盖；本规范只**消费**。
- `sonata-05-train` 旧规范的 `MultipoleDiagnosticHook` —— 若 PI 后续仍需要该 hook，单独 spec；本规范不内联。
- `bm_sonata.py`（既有 sonata-03-bench 入口）的弃用 —— legacy 入口保留；本脚本与之并存。
- `tests/test_molix/...` 新增用例 —— PI 明示 work/ 层"无工程化"。
- 跨账户 / 跨集群提交 —— `submit.py` 只支持 Alvis 形态；其它集群（Berzelius、Tetralith）由用户自行 fork。
