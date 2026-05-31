---
title: MolRecSource — labeled-configuration dataset over molrs.MolRec
status: code-complete
created: 2026-05-31
---

# MolRecSource — labeled-configuration dataset over molrs.MolRec

## Summary

为多教师蒸馏与标注构象数据提供一个新的具体数据源 `MolRecSource`，它从一个 molrs `MolRec` 的 zarr 记录中读取「带标签的构象数据集」。一个标注构象数据集就是一条**标准的、未经改动的** `MolRec` 记录：构象保存在 `trajectory` 帧中，教师标签保存在扁平的 `observables` 映射里（按 `"<teacher_id>.<name>"` 命名约定），教师溯源保存在自由的 `method` JSON 树下。`MolRecSource` 在构造时选定一个 `teacher_id`，急加载（eager-load）该记录，对外暴露 `data/source.py` 里的 `DataSource` Protocol：`__len__` 为构象数，`__getitem__(idx)` 返回扁平样本 `{Z, pos, (box), targets={...}}`，其中 `targets` 由带该教师前缀的 observables 键**动态**构建。整个约定不要求 molrs 有任何代码改动——它完全建立在 `MolRec` 既有的通用原语之上。

## Domain basis

本 spec 不引入任何新方程；科学正确性的关注点是**保真度（fidelity）**而非推导。「写入 MolRec → 通过 `MolRecSource` 读回」的往返必须满足以下保真不变量，并由回归门（见 Testing strategy）强制：

- **dtype 不变量**：QM9 / rMD17 等数据集以 `float32` 存储。往返**不得静默上转**（例如 `float32 → float64`）。读回张量的 dtype 必须与写入时逐字一致。
- **单位不变量**：可观测量携带的 `unit`（例如能量 eV、力 eV/Å）必须随 observable 一路传递，不得丢弃或改写；教师溯源单位记录在 `method["<teacher_id>"]["units"]`。
- **力的形状不变量**：逐原子力以 VECTOR observable 存储，`axes=["timestep","atom","component"]`，形状 `[n_frames, n_atoms, 3]`；读回后单帧力形状必须为 `[n_atoms, 3]`。
- **标量数值不变量**：QM9 的 15 个标量属性（`A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv`，即 `molix.datasets.qm9._QM9_GRAPH_TARGETS`）写入后逐值读回必须无损（值相等且 dtype 相等）。

参考数据集 DOI（写入模块 docstring，遵守科学正确性规则）：
- QM9 — Ramakrishnan et al., *Scientific Data* 1, 140022 (2014). https://doi.org/10.1038/sdata.2014.22
- rMD17 — Christensen & von Lilienfeld, *MLST* 1 (2020). https://doi.org/10.1088/2632-2153/abba6f

## Design

**实体与所有权。** 新增公共类 `MolRecSource`，位于 `src/molix/datasets/molrec.py`，从 `src/molix/datasets/__init__.py` 重新导出。它**遵从**（conform to）`molix.data.source.DataSource` Protocol（`runtime_checkable`），但**不是子类**——与 `RevMD17Source` 一致采用鸭子类型。

**多教师命名约定（molnex 约定，零 molrs 改动）。**
- 构象 → `MolRec` 的 `trajectory` 帧，每帧 `atoms{Z, pos}` + `box`。构象来源（复用已有数据集几何 vs 用户提供几何）在下游**不可区分**——两者都只是 `trajectory` 帧。
- 教师标签 → `MolRec` 扁平 `observables: BTreeMap<String, ObservableRecord>` 映射中的条目，键遵循扁平命名约定 `"<teacher_id>.<name>"`（如 `teacherA.energy`、`teacherA.forces`）。标量用 scalar observable；逐原子力用 VECTOR observable（`axes=["timestep","atom","component"]`，形状 `[n_frames, n_atoms, 3]`）。标签是任意的——**无固定 target schema**。
- 教师溯源 → 自由嵌套在 `MolRec` 任意的 `method` JSON 树下，如 `method = {"teacherA": {theory_level, model_id, checkpoint_hash, units}, ...}`。molrs 对此无 schema，纯自由 JSON。

**构造与急加载。** `__init__(record_path, teacher_id, ...)` 通过 `molrs.MolRec.read_zarr` 急加载记录（镜像 `RevMD17Source`：`Z` 作为跨帧共享量持有一次，逐构象 `pos` 堆叠）。当 `teacher_id` 不匹配任何 observable 键前缀时，**快速失败**并给出清晰错误信息，列出记录中可用的教师前缀集合。

**动态 per-instance TargetSchema（对现有源的有意偏离）。** 现有 3 个源（QM9 / RevMD17 / ThreeBPA / WaterLES）使用冻结的类属性 `TARGET_SCHEMA`。`MolRecSource` **不**附带冻结的 `TARGET_SCHEMA` 类属性；它在 `__init__` 里从发现到的、带 `"<teacher_id>."` 前缀的 observable 键**动态**构建一个 per-instance `TargetSchema`：scalar observable → `graph_level`，vector observable（按 kind/axes 推断）→ `atom_level`。剥离前缀后的名字即为 target 名。此偏离是有意的，需在 docstring 中点明。

**`__getitem__` 契约。** `__getitem__(idx)` 返回扁平 dict `{Z, pos, (box), targets={...}}`，`targets` 由前缀匹配的 observables 动态填充（剥离 `"<teacher_id>."` 前缀作为 target 名）。形状契约与 `RevMD17Source` 对齐：scalar target `(1,)`，atom-level force target `(N, 3)`。

**`source_id` 与 per-teacher 缓存失效。** 镜像 `RevMD17Source.source_id` 的 `f"<name>:size=<bytes>:n=<n>"`（含可选 `:total=`），并**追加** `:teacher=<teacher_id>`，使下游 `PackedCache` 按教师维度失效。这是 per-teacher 缓存失效**唯一需要的杠杆**——`source.source_id` 是缓存键链的根（librarian 已确认），`pipeline.cache_key` / `PackedCache` 无需任何改动。若同一记录暴露的 target 集合可变，参照 `SubsetSource.source_id` 的 SHA256 习惯将其折入 `source_id`。

**与现有层的关系（显式声明）。**
- 持久、可共享的数据集 = `MolRec` / zarr。
- 临时、每次运行的预处理 scratch = 下游既有的 `PackedCache` / `MmapDataset`。
- 二者是**不同的层，无冲突**：`MolRecSource` 在上游产出原始样本，`PackedCache`/`MmapDataset` 在下游做预处理与缓存。`MmapDataset` 保持 teacher-agnostic（教师维度仅经由 `source_id` 进入缓存键）。

**弃用指令（记录在案）。** molpy 的 `MolStore`（`molpy/src/molpy/io/store/`）为弃用/死命名；molnex 独占使用 `molrs.MolRec`。molnex 当前无 `MolStore` 使用——此处仅记录方向性指令。

## Files to create or modify

- `src/molix/datasets/molrec.py` (new) — `MolRecSource` 实现，模块 docstring 携带 QM9 / rMD17 DOI，`__all__` 置于文件末尾。
- `src/molix/datasets/__init__.py` — 重新导出 `MolRecSource`，加入 `__all__`。
- `tests/test_molix/test_datasets/test_molrec.py` (new) — `MolRecSource` 单元测试 + 往返回归门。
- `tests/test_molix/test_datasets/conftest.py` — 新增写出小型 `MolRec` zarr 记录的 fixture（QM9 切片 + 含力数据集切片），镜像现有 `water_les_root` / `charged_dimers_root` 风格。

## Tasks

- [x] Pin the `molrs.MolRec` API against the installed molrs wheel (`read_zarr` / `write_zarr` signatures, `trajectory` frame layout, `observables` map + `ObservableRecord` kind/axes typing, `method` JSON access) and record findings inline in the module docstring
- [x] Add a conftest fixture writing a tiny QM9-like MolRec zarr record (Z, pos, 15 float32 scalar observables under `teacherA.` prefix) and a force-bearing MolRec record (energy scalar + per-atom forces vector observable) in `tests/test_molix/test_datasets/conftest.py`
- [x] Write failing tests for `MolRecSource` construction, Protocol conformance, dynamic TargetSchema, and `__getitem__` flat-sample contract (`tests/test_molix/test_datasets/test_molrec.py`)
- [x] Write failing tests for `source_id` per-teacher invalidation and for fail-fast on unknown `teacher_id` (`tests/test_molix/test_datasets/test_molrec.py`)
- [x] Write failing roundtrip regression-gate tests: QM9 lossless (values + dtype) and force-bearing lossless (per-atom force shape + values) (`tests/test_molix/test_datasets/test_molrec.py`)
- [x] Implement `MolRecSource` in `src/molix/datasets/molrec.py` (eager `read_zarr` load, shared `Z`, stacked `pos`, dynamic per-instance `TargetSchema`, dynamic `targets` in `__getitem__`, `source_id` with `:teacher=` suffix, fail-fast unknown teacher)
- [x] Re-export `MolRecSource` from `src/molix/datasets/__init__.py` and add to `__all__`
- [x] Add Google-style docstrings with units and tensor shapes to `MolRecSource` and its public methods, carrying QM9 / rMD17 DOIs in the module docstring and noting the dynamic-TargetSchema departure
- [x] Verify against the fidelity invariants (dtype preserved, units carried, force shape `[n_atoms,3]`, 15 QM9 scalars lossless) via the roundtrip gate
- [x] Run full check + test suite (`ruff check src/ && ruff format --check src/` and `python -m pytest tests/test_molix/test_datasets/test_molrec.py -v`)

## Testing strategy

**Happy path.**
- 构造 `MolRecSource(record_path, teacher_id="teacherA")`，断言 `len(source) == n_conformers`，`isinstance(source, DataSource)` 为真（Protocol 鸭子类型）。
- `source[0]` 返回扁平 dict，含 `Z`、`pos (N,3)`、可选 `box`、`targets`；`targets` 键为剥离前缀后的 target 名（如 `energy`、`forces`）。
- 动态 per-instance `TargetSchema`：scalar observable 归入 `graph_level`，vector force observable 归入 `atom_level`。

**Edge cases.**
- 未知 `teacher_id` → 快速失败，错误信息列出记录中可用的教师前缀。
- 记录含多个教师前缀（`teacherA.`、`teacherB.`）时，所选教师的 targets 与另一教师隔离。
- `source_id` 在不同 `teacher_id` 下产出不同字符串（含 `:teacher=<teacher_id>` 后缀），证明下游 `PackedCache` 按教师维度失效。
- 仅标量、无力的记录 → `targets` 仅含 graph-level 项，`atom_level` 为空。

**Domain validation（科学正确性，强制）—— 往返回归门。**
- (a) **QM9 全量保真**：构造 QM9 切片（Z、pos、15 个标量 targets）→ `write_zarr` 写成 MolRec 记录 → 经 `MolRecSource` 读回 → 断言无损：15 个标量逐值相等且 dtype 为 `float32`（不上转），pos/Z dtype 保持。
- (b) **含力数据集保真**：构造 rMD17 或 3BPA 切片（energy + 逐原子力）→ `write_zarr` → 读回 → 断言逐原子力形状 `[n_atoms, 3]`、值无损、dtype 保持，单位经 observable `unit` 传递。
- 该门证明扁平命名 + vector observable 约定确实能往返穿过 molrs zarr。

## Out of scope

- **运行教师模型以生产记录的 writer/generator**：本 spec 仅覆盖**读取**一条已存在的记录 + 往返门（往返门为测试目的写出夹具记录，不构成生产用 writer）。生产侧的标注/蒸馏管线另立 spec。
- **任何 molrs 代码改动**：`MolRec` 是通用且**有意保持不变**的——`method`/`meta`/`parameters` 为任意 JSON，`observables` 键与 `axes` 自由格式。不得将领域键硬编码进 molrs。
- **下游 `PackedCache` / `pipeline.cache_key` / `MmapDataset` 改动**：per-teacher 失效完全经由 `source_id` 实现，下游零改动。
- **molpy `MolStore` 的迁移或清理**：仅记录其为弃用命名；molnex 当前无使用。
