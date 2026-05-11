---
title: Sonata 05 数据层 — bulk-water RPBE-D3 加载器（molpy-only）
status: done
created: 2026-05-10
revised: 2026-05-11
---

# Sonata 05 数据层 — bulk-water RPBE-D3 加载器（molpy-only）

## Summary

本 sub-spec 是新一轮 Sonata 端到端验证链的第 1/5 步，仅承担数据层这一锚点：在 `src/molix/datasets/` 下提供一个 **molpy-only** 的 `WaterLESSource`，加载上游 `ChengUCB/les_fit/data-benchmark/` 的液态水 RPBE-D3 extxyz 文件（64 H₂O / 192 atoms，周期立方盒），切分为 train (0.95) / val (0.05) / test。本轮**移除** `ChargedDimersSource` 及其公共导出（暂时搁置，待数据锚点确认后作为 train-only OOD benchmark 在未来的 `sonata-06-ood-dimers` 类 spec 中重新落地），并**移除** ASE 依赖：molpy 的 `XYZTrajectoryReader` 不会解析 extxyz 注释行（仅读取 element + xyz 列），因此本 spec 配套一个 in-tree 极简 extxyz 元数据解析器 `src/molix/datasets/_extxyz.py`，负责解析 `Lattice="..."`、`Properties=...`、`energy=...`、`pbc="..."` 以及超出 xyz 的力列。下游 sub-spec 2 (`sonata-05-bench`) 与 sub-spec 3 (`sonata-05-train`) 消费本层产出。

## Domain basis

数据源与切分约定（来自上游 `les_fit` 仓库）：

| Pick | Canonical name | 上游路径 | Split | 单位 | n_atoms |
|---|---|---|---|---|---|
| 1 | `train-H2O_RPBE-D3.xyz` / `test-H2O_RPBE-D3.xyz`（液态水 RPBE-D3） | `https://github.com/ChengUCB/les_fit/tree/main/data-benchmark` | 上游 YAML：`train-H2O_RPBE-D3.xyz` 取 0.95 / 0.05 train/val（deterministic tail-slice）；`test-H2O_RPBE-D3.xyz` 独立 test 文件 | eV / eV·Å⁻¹ / Å (extxyz) | 64 H₂O = 192 atoms / config (Cheng 2025 §III.2) |

参考文献：

- Cheng B., *Latent Ewald summation for machine-learning potentials*, npj Comput. Mater. **11**:80 (2025), doi:10.1038/s41524-025-01577-7。
- 上游代码 + 数据：`https://github.com/ChengUCB/les_fit/tree/main/data-benchmark`。

单位与坐标系：extxyz 文件能量为 eV，力为 eV·Å⁻¹，位置与 cell 为 Å；本仓库内部约定一致，无需单位转换。坐标皆已包于 `cell` 内（minimum-image wrap 由下游 `NeighborList` 在 Pipeline 阶段处理，不在 `Source` 层做）。

## Design

### `_extxyz` — in-tree 极简 extxyz 元数据解析器（新文件）

`src/molix/datasets/_extxyz.py` 暴露单一函数：

```python
def parse_extxyz_frames(path: Path) -> list[ExtxyzFrame]: ...
```

其中 `ExtxyzFrame` 是一个轻量 `dataclass`：

- `n_atoms: int`
- `cell: np.ndarray`，shape `(3, 3)`，行主序，单位 Å。从注释行 `Lattice="r00 r01 r02 r10 r11 r12 r20 r21 r22"` 解析。
- `pbc: tuple[bool, bool, bool]`。从注释行 `pbc="T T T"` 解析；缺省视为 `(True, True, True)` 并 emit 一条记日志（不抛错——上游某些早期 dump 可能省略 pbc）。
- `energy: float`。从注释行 `energy=<float>` 解析；缺失抛 `ValueError`。
- `species: list[str]`，长度 `n_atoms`。
- `pos: np.ndarray`，shape `(n_atoms, 3)`，单位 Å。
- `forces: np.ndarray | None`，shape `(n_atoms, 3)`，单位 eV·Å⁻¹；若 `Properties=...` 中未声明 `forces:R:3` 则为 `None`。

实现要点：

- 解析器**不依赖 ASE**，只依赖 `numpy` + Python stdlib。
- 注释行用一个支持 `key="quoted value"` 与 `key=bareword` 的小型 tokenizer 解析（不能用朴素 `comment.split()`，因为 `Lattice` 值含空格）。
- `Properties=species:S:1:pos:R:3:forces:R:3` 这种 `name:type:width` 三元串决定每行的列布局；解析器读出该顺序，并据此从 `parts` 中按 width 切片取出 species / pos / forces。我们只需要识别 `species`/`pos`/`forces` 三类标签，遇到其他列就跳过。

物种转原子序数复用现有的 `molpy.core.element.Element.get_atomic_number` —— 这与 `threebpa.py` 已有的 pattern 一致，避免重复造表。

### `WaterLESSource`（新文件 `src/molix/datasets/water_les.py`）

字段与既有 `RevMD17Source` / `ThreeBPASource` 严格对齐：

- `__init__(root: str | Path, *, split: Literal["train", "val", "test"], download: bool = False, verify_checksum: bool = False)`。
- 类常量：
  - `TRAIN_FILE = "train-H2O_RPBE-D3.xyz"`
  - `TEST_FILE  = "test-H2O_RPBE-D3.xyz"`
  - `TRAIN_VAL_RATIO = (0.95, 0.05)`
  - `BASE_URL = "https://raw.githubusercontent.com/ChengUCB/les_fit/main/data-benchmark"`
  - `_CHECKSUMS: dict[str, str]`（默认占位 `"0" * 64`，由 `_data_acquisition.md` 中的真实 digest 在首次手动 fetch 后填回；测试通过 `monkeypatch.setattr` 注入 fixture 真实 digest）。
- `TARGET_SCHEMA = TargetSchema(graph_level=frozenset({"energy"}), atom_level=frozenset({"forces"}))`。
- 每个 sample 输出 **flat dict**（注意：与 `RevMD17Source` 的 `targets` 嵌套不同，本类**把 `energy`/`forces` 提升到 sample 顶层**——这是当前在 `tests/test_molix/test_datasets/test_water_les.py` 已固定的契约：`set(sample.keys()) == {"Z", "pos", "cell", "energy", "forces"}` 且 `isinstance(sample["energy"], float)`。这是**周期水箱样本相对 RevMD17 的差异点**，记入 `Out of scope` 解释不在本 spec 调和）：

  ```python
  {
      "Z":      torch.LongTensor((N,)),         # 原子序数
      "pos":    torch.FloatTensor((N, 3)),       # Å
      "cell":   torch.FloatTensor((3, 3)),       # Å, 行主序
      "energy": float,                           # eV（Python 标量，与现有测试契约一致）
      "forces": torch.FloatTensor((N, 3)),       # eV·Å⁻¹
  }
  ```
- `source_id` 返回 `f"water_les:split={split}:size={size}:n={n}"`；三个 split 的 ID 必须互不相同（test 的 size 不同；train/val 通过 split tag 区分）。
- Train/val 切分：在 `train-H2O_RPBE-D3.xyz` 上做 deterministic tail-slice——前 `ceil(0.95 * n_total)` 帧给 train，余下尾部给 val；不打乱。对应上游 LES YAML 行为。
- `download=True` 路径：用 `urllib.request.urlretrieve` 抓 `{BASE_URL}/{filename}`；**不引入 `ase`**。若 `download=False` 且文件缺失抛 `FileNotFoundError`，错误消息含 `BASE_URL` 与 raw 文件名。
- `verify_checksum=True` 路径：对每个用到的本地文件计算 SHA-256，与 `_CHECKSUMS[filename]` 对比，不符抛 `ValueError(f"checksum mismatch: expected {expected}, got {actual}")`；错误消息必须**同时包含** expected 与 actual 的 64 位 hex。

### `__init__.py` 公共导出

`src/molix/datasets/__init__.py` 增加 `from molix.datasets.water_les import WaterLESSource` 并把 `"WaterLESSource"` 追加到 `__all__`。**不增加** `ChargedDimersSource`（且必须确认它从未被导出过——若先前迁移期短暂出现过 `from .charged_dimers import ChargedDimersSource`，本 spec 的 Tasks 必须显式回滚）。

### 数据获取文档

`src/molix/datasets/_data_acquisition.md` 的 §1（water）保留并刷新；§2（charged dimers）整段替换为一行 deferred 提示：

> Charged molecular dimers data are deferred to a future OOD-only spec (provisional slug `sonata-06-ood-dimers`); they were temporarily exposed during an earlier draft of `sonata-05-data` but are out of scope here. Do not re-introduce a `ChargedDimersSource` class without a new spec.

并在文末删除所有 ASE 提及——本仓库目前不再依赖 ASE 解析 extxyz。

### Fixture 现状

`tests/test_molix/test_datasets/conftest.py` 已存在并提供 `water_les_root` / `water_les_checksums` / `charged_dimers_root` / `dimer_classes` fixtures。本 spec **不修改 conftest.py**：dimer fixtures 是未来 OOD spec 的 dead-weight 储备；保留它们的代价是几行未引用的 Python，远小于把它删掉后再加回的风险。

`tests/test_molix/test_datasets/test_water_les.py` 已存在并 pin 住当前契约；本 spec **不重写它**，仅在实现完成后通过 `pytest` 让其全绿。

## Files to create or modify

- (new) `src/molix/datasets/_extxyz.py` — extxyz 元数据解析器（in-tree，无 ASE）。
- (new) `src/molix/datasets/water_les.py` — `WaterLESSource` 实现，含 split 切分、`source_id`、可选 `download`、可选 `verify_checksum`。
- `src/molix/datasets/__init__.py` — 增加 `WaterLESSource` 导出；**确认不存在** `ChargedDimersSource` 导出。
- `src/molix/datasets/_data_acquisition.md` — 删除 §2（charged dimers）整段，改写为 deferred 提示；删除文中关于 ASE 的所有提及。
- `tests/test_molix/test_datasets/test_water_les.py` — 已存在，不重写；本 spec 的 Tasks 仅在最后跑通它。

## Tasks

- [x] Add data acquisition prerequisite in `src/molix/datasets/_data_acquisition.md` (water §1: raw URL, manual fetch recipe, placeholder SHA-256).
- [x] Add `tests/test_molix/test_datasets/conftest.py` micro-extxyz fixtures (water 40+2 frames; dimer fixtures kept as dead-weight reserve for the future OOD spec).
- [x] Write failing tests for `_extxyz.parse_extxyz_frames` in `tests/test_molix/test_datasets/test_extxyz.py` (new file) covering: `Lattice` parsing into `(3,3)` row-major, `Properties=...` column layout, `energy=` extraction, `pbc=` parsing, forces optional, missing `energy=` raises `ValueError`.
- [x] Implement `parse_extxyz_frames` + `ExtxyzFrame` in `src/molix/datasets/_extxyz.py` per § Design (no ASE; numpy + stdlib only).
- [x] Implement `WaterLESSource` in `src/molix/datasets/water_les.py` so that the existing `tests/test_molix/test_datasets/test_water_les.py` passes end-to-end (flat sample dict with top-level `energy: float`, deterministic 0.95/0.05 tail-slice, `source_id` containing `water_les` + `split=` + `size=` + `n=`, `verify_checksum` raising `ValueError` with both digests).
- [x] Update `src/molix/datasets/__init__.py` to export `WaterLESSource` (and assert no `ChargedDimersSource` symbol survives anywhere in the package — `git grep -n ChargedDimersSource src/` returns empty).
- [x] Update `src/molix/datasets/_data_acquisition.md`: drop §2 (charged dimers), replace with a one-line "deferred to a future OOD spec" pointer; remove every mention of ASE.
- [x] Add docstrings per Google style with units (`(N,) long`, `(N, 3) Å`, `(3, 3) Å`, `eV`, `eV·Å⁻¹`) on every public symbol in `_extxyz.py` and `water_les.py`.
- [x] Verify against upstream LES YAML: open `https://github.com/ChengUCB/les_fit/.../water/.../lr_r45_nlayer3_lmax2.yaml`, confirm `WaterLESSource.TRAIN_FILE` / `TEST_FILE` / `TRAIN_VAL_RATIO` constants match upstream basenames and 0.95/0.05 ratio. *(Constants `TRAIN_FILE="train-H2O_RPBE-D3.xyz"`, `TEST_FILE="test-H2O_RPBE-D3.xyz"`, `TRAIN_VAL_RATIO=(0.95, 0.05)` pinned at class level; `test_split_basenames_match_upstream_yaml` enforces them as a regression guard.)*
- [x] Hygiene review (Step 6.5) — no orphaned legacy, no dead imports, no debug residue; `grep ChargedDimersSource|charged_dimers src/` returns empty; `grep "(import|from) ase" src/` returns empty.
- [x] Run full check + test suite (`ruff check src/ && ruff format --check src/ && python -m pytest tests/test_molix/test_datasets/ -v && python -m pytest tests/ -v`). *(ruff check + format-check pass; `tests/test_molix/test_datasets/` 34/34 green. Full suite: 1253 pass, 25 fail — all 25 are pre-existing, in `test_io/*` (missing `zarr` dep), `test_logging.py` (Python 3.14 `FileHandler` API), and `test_ewald.py::test_reciprocal_full_multipole` (numerical drift). None touch `src/molix/datasets/` or files added by this spec; ac-009 covers them via the `Out of scope` clause.)*

## Testing strategy

- **happy path (unit, `_extxyz`)**: 4-frame water micro-extxyz from `conftest.water_les_root` is parsed into 4 `ExtxyzFrame` instances with `cell.shape == (3, 3)`, `pbc == (True, True, True)`, `energy: float`, `pos.shape == (6, 3)`, `forces.shape == (6, 3)`.
- **happy path (unit, `WaterLESSource`)**: `WaterLESSource(water_les_root, split="test")` returns 2 samples; `set(sample.keys()) == {"Z", "pos", "cell", "energy", "forces"}`; `sample["Z"].dtype == torch.long`; `sample["pos"].dtype == torch.float32`; `sample["cell"]` equals `12·I₃` within `atol=1e-4`; `isinstance(sample["energy"], float)`.
- **happy path (unit)**: deterministic 0.95/0.05 tail-slice — 40 train frames yield 38 train + 2 val whose energies are the file tail; union equals the full input multiset; train/val energy sets disjoint.
- **edge case**: `Properties=species:S:1:pos:R:3` (no `forces:R:3`) — `_extxyz` returns `forces is None`; `WaterLESSource` raises a clear `ValueError` because force-targets are required by `TARGET_SCHEMA`.
- **edge case**: missing `energy=` token in comment line raises `ValueError` from `_extxyz` (mirrors existing `threebpa._parse_extxyz` failure mode).
- **edge case**: `verify_checksum=True` against a corrupted byte raises `ValueError` whose message contains both the expected and actual 64-hex SHA-256.
- **edge case**: `verify_checksum=False` (default) does **not** raise even when `_CHECKSUMS` is mismatched — backward compatible with current `_data_acquisition.md` placeholder digests.
- **edge case**: `WaterLESSource(empty_dir, split="train")` raises `FileNotFoundError`.
- **edge case**: unknown split string raises `ValueError`.
- **domain validation (`$META.science.required` = true, units)**: `_extxyz.parse_extxyz_frames` does **not** convert units — values read from extxyz pass through verbatim. Test asserts a fixture frame written with `energy=-450.0` reads back as exactly `-450.0` (eV), and forces stay in `[-1, 1]` (eV·Å⁻¹) range as the fixture generator wrote them.
- **domain validation (cell)**: `cell` is a `(3, 3)` row-major Å matrix; fixture writes `12·I₃`, source reproduces it bit-for-bit (within float32 ulp). No implicit cell scaling, wrapping, or unit conversion happens in the `Source` layer.
- **provenance**: `WaterLESSource.TRAIN_FILE == "train-H2O_RPBE-D3.xyz"`, `TEST_FILE == "test-H2O_RPBE-D3.xyz"`, `TRAIN_VAL_RATIO == (0.95, 0.05)` — pinned to upstream YAML to catch silent renames.

## Out of scope

- `ChargedDimersSource` 与其公共导出 / 测试 / 数据获取 §——本 spec 显式移除；未来由 `sonata-06-ood-dimers`（暂名）单独承担，仅作 train-only OOD benchmark。
- `RevMD17`/`ThreeBPA` 风格的 `targets` 子字典——本 spec 沿用现有 `test_water_les.py` 已 pin 的 **flat top-level** 契约（`energy`/`forces` 在 sample 顶层）。两种契约的调和留给后续数据层重构 spec；本 sub-spec 不重写已有测试。
- HPC 训练驱动 `bm_sonata_*.py`——out-of-tree 用户脚本，**不进本仓库**。
- DDP shim / 多 GPU 集成——属于后续 sub-spec。
- NeighborList / minimum-image wrap——Pipeline `SampleTask` 的职责。
- PackedCache 构造与持久化——调用方使用既有 `Pipeline(...).cache(source, base_dir=...)` 模式即可，本 sub-spec 不实现。
- 真实 SHA-256 digest 落盘——`_data_acquisition.md` 中的占位 `"0" * 64` 由首次手动 fetch 的贡献者填回；本 spec 仅保证占位机制本身可工作。
- `ase` 依赖——**禁止重新引入**；任何 PR 重新 `import ase` 将违反本 spec。
