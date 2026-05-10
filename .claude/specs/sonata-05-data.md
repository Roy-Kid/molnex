---
title: Sonata 05 数据层 — 真实周期数据加载器（bulk water + charged dimers）
status: approved
created: 2026-05-10
---

# Sonata 05 数据层 — 真实周期数据加载器（bulk water + charged dimers）

## Summary

为 Sonata HPC 科学正确性验证链（`sonata-05-*`）落地数据基础设施：在
`src/molix/datasets/` 下新增两个 `Source` 类——`WaterLESSource`（液态水
RPBE-D3，64 H₂O / 192 atoms，周期立方盒）和 `ChargedDimersSource`（6 类
带电分子二聚体，5–15 Å 分离距离，30 Å 周期立方盒）。两者均严格对齐上游
`ChengUCB/les_fit/data-benchmark/` 的 extxyz 文件格式与 train/val/test 切
分约定，发布 `TARGET_SCHEMA`、`source_id`、可选 `download()` 和 periodic
cell 支持，并在 `tests/test_molix/test_datasets/` 下补齐固定校验和（fixed
checksum）的子集回归测试。该 sub-spec 不引入任何训练驱动、不接入 DDP、不
重新证明 kernel 正确性——DDP 由后续 `sonata-05-hpc` 负责；HPC 训练驱动
（`bm_sonata_hpc.py`、诊断核 `S_d(k)` / dimer dissociation 等）作为
out-of-tree 用户脚本由具体使用者各自维护，**不进本仓库**。

## Domain basis

数据源与切分约定（来自上游 `les_fit` 仓库 + scientist 输出 §A）：

| Pick | Canonical name | 上游路径 | Split | 单位 | n_atoms |
|---|---|---|---|---|---|
| 1 | `train-H2O_RPBE-D3.xyz` / `test-H2O_RPBE-D3.xyz`（液态水 RPBE-D3） | `https://github.com/ChengUCB/les_fit/tree/main/data-benchmark` | 上游 YAML：`train-H2O_RPBE-D3.xyz` 取 0.95 / 0.05 train/val；`test-H2O_RPBE-D3.xyz` 独立 test 文件 | eV / eV·Å⁻¹ / Å (extxyz) | 64 H₂O = 192 atoms / config (Cheng 2024 §III.2) |
| 2 | charged molecular dimers（C₃N₃H₁₀⁺ / C₂O₂H₃⁻ 等 6 类） | Huguenin-Dumittan et al. 2023，arXiv:2412.15455 §III 引用，**下载 URL 上游论文与代码均未明确锚定**——必须在实施期通过 `Add data acquisition prerequisite` 任务向论文作者或 LES 作者落实 | 10 train（5–12 Å 分离） / 3 test（12–15 Å 分离）——故意分布偏移以暴露长程失效 | eV / eV·Å⁻¹ | ≲ 30 atoms / config，30 Å 立方盒周期 |

参考文献（verbatim DOI / arXiv，与 `sonata-04-docs` 一致）：

- Cheng B., *Latent Ewald summation for machine-learning potentials*,
  npj Comput. Mater. **11**:80 (2025), doi:10.1038/s41524-025-01577-7。
- Kim D. et al., *Machine learning of charges and long-range
  interactions from energies and forces*, Nat. Commun. **16**:8763
  (2025), doi:10.1038/s41467-025-63852-x。注意第一作者为 **Kim D.**，
  非 King D. S.。
- 上游代码 + 数据：`https://github.com/ChengUCB/les`、
  `https://github.com/ChengUCB/les_fit`。
- Huguenin-Dumittan et al. 2023（charged dimer 数据集来源），通过
  arXiv:2412.15455 §III 二级引用获取。

单位与坐标系：所有 extxyz 文件能量为 eV，力为 eV·Å⁻¹，位置与 cell 为
Å；本仓库内部约定一致，无需单位转换。坐标皆已包于 `cell` 内（minimum-
image 由下游 `NeighborList` 在 Pipeline 阶段处理，不在 `Source` 层
做）。

## Design

新增两个 `Source` 类，形状严格镜像 `RevMD17Source` /
`ThreeBPASource`：

- `WaterLESSource`（`src/molix/datasets/water_les.py`，新文件）
  - 字段：`root: Path`、`split: Literal["train", "val", "test"]`、
    `download: bool = False`（默认 `False`，因为上游为公共 GitHub 仓
    库 raw 文件而非 figshare 包，由 `acquisition.md` 列示手动 fetch
    指令；`download=True` 时使用 `requests.get` 抓取 raw URL，命名
    与上游一致）。
  - `TARGET_SCHEMA = TargetSchema(graph_level=frozenset({"energy"}),
    atom_level=frozenset({"forces"}))`。
  - 每个 sample 输出 flat `dict`（pre-collate 形状）：
    `{"Z": (N,) long, "pos": (N, 3) float, "cell": (3, 3) float,
    "energy": float, "forces": (N, 3) float}`。`cell` 字段是周期数据
    集的关键差异点——`RevMD17Source` 不带 `cell`，本类必须带。
  - `source_id` 包含文件路径、文件大小（bytes）、`split` 名、样本
    数；用于 PackedCache 键。
  - 0.95 / 0.05 train/val 切分通过 deterministic 顺序切片实现（不
    使用 random shuffle）以与上游 YAML 完全一致。
- `ChargedDimersSource`（`src/molix/datasets/charged_dimers.py`，新
  文件）
  - 字段：`root: Path`、`split: Literal["train", "test"]`、
    `dimer_class: Literal["C3N3H10+", "C2O2H3-", ...]`（6 类枚举，
    每类独立 source 实例；多类合训通过 `ConcatDataset` 在
    Pipeline 阶段拼接，本层不 hard-wire）、`download: bool = False`。
  - `TARGET_SCHEMA` 同上。
  - sample 形状同上（带 `cell`）。
  - 切分按上游分布偏移约定：`split="train"` 返回 5–12 Å 子集，
    `split="test"` 返回 12–15 Å 子集；这两个 split 不打乱，按分离
    距离升序排列以便 out-of-tree 训练驱动的 dissociation curve
    诊断核按行索引消费。

extxyz 解析器设计：
- 复用 ASE（`ase.io.read(..., format="extxyz", index=":")`）。`pyproject.toml`
  已经在 dev extras 中含 `ase`；如未含则 sub-spec 作为依赖添加项处
  理（验证步骤里硬性 check）。
- 解析后 in-memory 持有 `list[Sample]`；PackedCache 由调用方按既有
  Pipeline pattern 构造（不在 Source 层 cache）。

数据获取前置任务：在 `src/molix/datasets/_data_acquisition.md`（新文
件，markdown 而非 Python）写明：
- 上游 raw URL（针对 water）；
- 上游 issue / 邮件链接（针对 charged dimers，用以让用户自行解决数据
  落盘后 `root` 指向哪里）；
- 校验和（SHA-256）锚定每个 extxyz 文件的预期内容；如下载脚本得到不
  同文件则 `Source.__init__` 抛出 `ValueError(f"checksum mismatch:
  expected {expected}, got {actual}")`。

文件命名与归属：两个 source 类暴露在 `molix.datasets`（即在
`src/molix/datasets/__init__.py` 增加两行 `from .water_les import
WaterLESSource` / `from .charged_dimers import ChargedDimersSource`，
并 append 到 `__all__`）。

## Files to create or modify

- (new) `src/molix/datasets/water_les.py` — `WaterLESSource` 实现，
  含 extxyz 解析、`split` 切分、`source_id`、可选 `download()`、
  checksum 校验。
- (new) `src/molix/datasets/charged_dimers.py` — `ChargedDimersSource`
  实现，含 6-class 枚举、deliberate distribution shift 切分、按分
  离距离升序排序逻辑、`source_id`、可选 `download()`、checksum 校
  验。
- (new) `src/molix/datasets/_data_acquisition.md` — 数据获取说明文
  档（raw URL / 上游 issue / SHA-256 列表）。
- `src/molix/datasets/__init__.py` — 增加两行 `from .* import *Source`
  并扩展 `__all__`。
- (new) `tests/test_molix/test_datasets/test_water_les.py` — 针对
  `WaterLESSource` 的单元测试（fixture 用迷你 extxyz，4 帧）。
- (new) `tests/test_molix/test_datasets/test_charged_dimers.py` —
  针对 `ChargedDimersSource` 的单元测试（fixture 用迷你 extxyz，
  6 类 × 2 帧）。
- (new) `tests/test_molix/test_datasets/conftest.py`（如不存在则新
  建；存在则追加 fixture）— 提供两个 `tmp_path` 下的 micro-extxyz
  fixture 工厂。

## Tasks

- [ ] Add data acquisition prerequisite in `src/molix/datasets/_data_acquisition.md` (raw URL for water, upstream issue link for charged dimers, SHA-256 list for both).
- [ ] Write failing tests for `WaterLESSource` in `tests/test_molix/test_datasets/test_water_les.py` covering: extxyz parse, `cell` propagation, `TARGET_SCHEMA`, split shape (0.95 / 0.05 train/val deterministic slice, separate test file), `source_id` stability, checksum mismatch raises `ValueError`.
- [ ] Implement `WaterLESSource` in `src/molix/datasets/water_les.py` per § Design.
- [ ] Write failing tests for `ChargedDimersSource` in `tests/test_molix/test_datasets/test_charged_dimers.py` covering: 6-class enum, deliberate-distribution-shift split (5–12 vs 12–15 Å), sort-by-separation invariant, `cell` propagation, `source_id` stability.
- [ ] Implement `ChargedDimersSource` in `src/molix/datasets/charged_dimers.py` per § Design.
- [ ] Add `tests/test_molix/test_datasets/conftest.py` micro-extxyz fixtures (4-frame water box, 6-class × 2-frame dimers) so the two new test files run hermetically without network access.
- [ ] Wire both classes into `src/molix/datasets/__init__.py` (`from … import …` + `__all__`).
- [ ] Add docstrings per Google style with units (`(N,) long`, `(N, 3) Å`, `(3, 3) Å`, `eV`, `eV·Å⁻¹`) on every public method of both classes; verify with `ruff check src/`.
- [ ] Verify against upstream LES YAML: open `https://github.com/ChengUCB/les_fit/.../water/.../lr_r45_nlayer3_lmax2.yaml`, confirm split ratios (0.95 / 0.05) and file basenames (`train-H2O_RPBE-D3.xyz`, `test-H2O_RPBE-D3.xyz`) match `_MOLECULES`-equivalent constants in `water_les.py`.
- [ ] Run full check + test suite (`ruff check src/ && ruff format --check src/ && python -m pytest tests/ -v`).

## Testing strategy

- **happy path（unit）**：micro-extxyz fixture（4 H₂O 帧）经
  `WaterLESSource(root, split="train")` 加载；断言 `len(src)`、
  `src[0]["Z"].shape`、`src[0]["pos"].shape`、`src[0]["cell"].shape ==
  (3, 3)`、`src[0]["energy"]` 是 float、`src[0]["forces"].shape ==
  (N, 3)`、`src.source_id` 含 `"water_les"` 与文件大小。
- **happy path（unit）**：dimer fixture 经
  `ChargedDimersSource(root, split="train", dimer_class="C3N3H10+")`
  加载；断言每个 sample 按分离距离升序、`split="test"` 全部 ≥ 12 Å、
  `split="train"` 全部 ≤ 12 Å。
- **edge case**：`WaterLESSource(root, split="val")` 与
  `WaterLESSource(root, split="train")` 的 indices 不重合（用
  fixture 的可数样本验证 0.95 / 0.05 deterministic slice）。
- **edge case**：checksum mismatch fixture（被故意写错一个字节的
  extxyz）触发 `ValueError`，错误消息包含 expected 与 actual SHA-
  256。
- **edge case**：未知 `dimer_class`（如 `"H2O"`）抛
  `ValueError`，列出 6 个合法名。
- **edge case**：`download=False` 且文件缺失抛
  `FileNotFoundError`，错误消息含 raw URL / 上游 issue 链接。
- **domain validation**（`$META.science.required` = true）：fixture
  样本能量数值必须以 eV 为单位（不是 Hartree / kcal·mol⁻¹）。验证
  方式：fixture extxyz 的 `Properties=...` 与 `energy=...` 头注释
  显式标记 `[eV]`；测试中读出后断言数值范围合理（液态水 RPBE-D3
  典型 192-atom 总能量在 −2000 到 −1500 eV 之间——micro fixture 用
  缩放过的 stub 值，但单位 tag 必须 eV）。
- **domain validation**：`cell` 字段在 fixture 中为 30 Å 立方对角；
  `WaterLESSource` 与 `ChargedDimersSource` 不得隐式截断或重新缩
  放 cell；断言 `torch.allclose(loaded["cell"], expected_cell,
  atol=1e-6)`。
- **domain validation**：force 字段单位为 eV·Å⁻¹（不是 Hartree·Bohr⁻¹）。
  fixture 头注释标记 `[eV/Ang]`；测试断言 `forces.dtype` 为 float
  且数值规模合理（≤ 50 eV·Å⁻¹）。

## Out of scope

- HPC 训练驱动 `bm_sonata_hpc.py`——out-of-tree 用户脚本，**不进本仓库**。
- DDP shim 与 `molix.core.Trainer` 多 GPU 集成——属于 `sonata-05-hpc`。
- 诊断核 `S_d(k)` / dimer dissociation evaluator——属于 out-of-tree
  训练驱动；如果未来证明可复用，可单独立 spec 提升到
  `molix.diagnostics`，但**不在本链落地**。
- sbatch / slurm 模板——本仓库无先例（librarian §interaction_points
  确认）；除非未来有第二个 HPC 用户也需要，否则不引入。
- 多节点 DDP——`sonata-05-hpc` 也只承诺单节点多 GPU。
- 分子液 NaCl 数据集——neither paper publishes Allegro+LES bars on
  it（scientist §A 推荐删除）。
- 重新证明 kernel 正确性——已由
  `tests/test_molpot/test_les_parity/test_forward.py` gating（atol=1e-6
  rtol=1e-6 in float64）。
- arXiv:2408.15165 figure-inset 数值转写——文本不可检索。
- `multipole-layer.md` 状态升迁——独立的 `multipole-layer-promote`
  spec 处理。
- 在 `Source` 层做 `NeighborList` / minimum-image wrap——这是
  Pipeline `SampleTask` 的职责，不属于数据集加载层。
- PackedCache 构造与持久化——调用方使用既有的
  `Pipeline().add(NeighborList(...)).build().cache(...)` 模式即可，
  本 sub-spec 不重复实现。
