---
slug: sonata-05-data
criteria:
  - id: ac-001
    summary: WaterLESSource parses extxyz and propagates periodic cell
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/test_water_les.py::test_parse_and_cell -v`
      passes; loaded sample has keys {"Z", "pos", "cell", "energy",
      "forces"}, `cell.shape == (3, 3)`, and `pos`/`forces` are
      float32 with shape `(N, 3)` for N == 192 (or fixture-scaled).
    status: pending
  - id: ac-002
    summary: WaterLESSource split is deterministic and matches upstream YAML
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/test_water_les.py::test_split_deterministic -v`
      passes; train/val ratio is 0.95/0.05 by deterministic slice
      (no shuffle); `train-H2O_RPBE-D3.xyz` and
      `test-H2O_RPBE-D3.xyz` map to disjoint splits matching the
      upstream YAML basenames in `les_fit`.
    status: pending
  - id: ac-003
    summary: WaterLESSource raises on checksum mismatch
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/test_water_les.py::test_checksum_mismatch -v`
      passes; a corrupted-byte fixture triggers `ValueError` whose
      message contains both expected and actual SHA-256.
    status: pending
  - id: ac-004
    summary: ChargedDimersSource enforces distribution-shift split
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/test_charged_dimers.py::test_distribution_shift -v`
      passes; for any of the 6 dimer classes, every `split="train"`
      sample has nearest inter-fragment separation ≤ 12 Å, every
      `split="test"` sample has separation ≥ 12 Å, and samples
      within a split are sorted by ascending separation.
    status: pending
  - id: ac-005
    summary: ChargedDimersSource validates dimer_class enum
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/test_charged_dimers.py::test_unknown_class -v`
      passes; `ChargedDimersSource(..., dimer_class="H2O")` raises
      `ValueError` listing all 6 legal names.
    status: pending
  - id: ac-006
    summary: Both sources expose stable source_id strings
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/ -k source_id -v`
      passes; `source_id` contains the dataset tag, file size in
      bytes, split name, and sample count for both classes; calling
      `source_id` twice on the same instance returns identical
      strings.
    status: pending
  - id: ac-007
    summary: Both sources publish TARGET_SCHEMA matching downstream contract
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/ -k target_schema -v`
      passes; both classes expose
      `TARGET_SCHEMA = TargetSchema(graph_level=frozenset({"energy"}),
      atom_level=frozenset({"forces"}))` as a class-level attribute
      consumable by `DataModule(target_schema=...)`.
    status: pending
  - id: ac-008
    summary: Units and cell shape preserve domain-physics contract
    type: scientific
    evaluator_hint: domain-units-check
    pass_when: |
      For both micro fixtures, the extxyz header declares units
      `[eV]` for energy and `[eV/Ang]` for forces; the loaded
      `cell` tensor matches the fixture-declared 30 Å cubic
      diagonal to atol=1e-6; loaded force magnitudes are bounded
      by 50 eV·Å⁻¹; loaded energy values are finite and float.
      The check is wired as a pytest test under
      `tests/test_molix/test_datasets/test_units_contract.py` and
      runs in CI (no HPC manual step required).
    status: pending
  - id: ac-009
    summary: Public exports landed in molix.datasets
    type: code
    pass_when: |
      `python -c "from molix.datasets import WaterLESSource,
      ChargedDimersSource"` exits 0; both names appear in
      `molix.datasets.__all__`.
    status: pending
  - id: ac-010
    summary: Repo lint and full test suite green
    type: runtime
    pass_when: |
      `ruff check src/ && ruff format --check src/ && python -m
      pytest tests/ -v` exits 0 on the implementer's machine after
      all sub-spec edits land.
    status: pending
---

# Acceptance criteria

ac-001 / ac-002 / ac-003 — `WaterLESSource` 的解析、切分与校验和三大不变量。
ac-004 / ac-005 — `ChargedDimersSource` 的分布偏移与枚举校验。
ac-006 / ac-007 — 两个 source 的 `source_id` 稳定性与 `TARGET_SCHEMA` 暴露。
ac-008 — 单位与周期 cell 的 domain-physics 契约（CI 内运行，非 HPC 手动）。
ac-009 — 公共导出落地 `molix.datasets`。
ac-010 — 仓库 lint + 全量测试通过。

provenance：所有 ac-001 至 ac-007 的字段（split 比例 0.95 / 0.05、dimer 5–12 / 12–15 Å 切分）来自 `les_fit/MLIPs/Allegro-LES/water/.../lr_r45_nlayer3_lmax2.yaml` 与 arXiv:2412.15455 §III；ac-008 的单位约定来自 Cheng 2025（npj Comput. Mater. 11:80, 2025）+ extxyz format spec；ac-010 来自 `CLAUDE.md` § Build & Development（`build.check` + `build.test`）。
