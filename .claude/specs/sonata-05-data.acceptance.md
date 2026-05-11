---
slug: sonata-05-data
criteria:
  - id: ac-001
    summary: WaterLESSource parses extxyz and exposes the periodic cell
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/test_water_les.py::TestWaterLESParse::test_parse_and_cell -v`
      passes: sample dict has keys `{Z, pos, cell, energy, forces}`, `cell.shape == (3, 3)`
      and equals `12·I₃` within atol=1e-4, `energy` is a Python `float`,
      `Z.dtype == torch.long`, `pos.dtype == torch.float32`.
    status: verified
    last_checked: 2026-05-11
  - id: ac-002
    summary: WaterLESSource splits 0.95/0.05 deterministically from train-…xyz
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/test_water_les.py::TestWaterLESSplit -v`
      passes: with 40-frame fixture, train has 38 samples, val has 2,
      val energies equal the last two energies in the input file,
      train ∩ val (by energy) is empty,
      and `WaterLESSource.TRAIN_FILE` / `TEST_FILE` / `TRAIN_VAL_RATIO` match
      `train-H2O_RPBE-D3.xyz` / `test-H2O_RPBE-D3.xyz` / `(0.95, 0.05)`.
    status: verified
    last_checked: 2026-05-11
  - id: ac-003
    summary: verify_checksum=True raises ValueError naming both digests
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/test_water_les.py::TestWaterLESChecksum -v`
      passes: corrupted byte triggers `ValueError(match="checksum mismatch")`,
      message contains two distinct 64-hex SHA-256 strings, and the default
      `verify_checksum=False` constructor succeeds even when `_CHECKSUMS` is mismatched.
    status: verified
    last_checked: 2026-05-11
  - id: ac-004
    summary: source_id is deterministic and split-distinguishing
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/test_water_les.py::TestWaterLESSourceID -v`
      passes: repeated `source_id` reads are equal, contain `water_les`,
      `split=<split>`, `size=<bytes>`, and `n=<len>`, and the three splits
      (`train`/`val`/`test`) yield three distinct `source_id` strings.
    status: verified
    last_checked: 2026-05-11
  - id: ac-005
    summary: TARGET_SCHEMA exposes graph={energy} + atom={forces}
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/test_water_les.py::TestWaterLESTargetSchema::test_target_schema -v`
      passes: `WaterLESSource.TARGET_SCHEMA.graph_level == frozenset({"energy"})`
      and `TARGET_SCHEMA.atom_level == frozenset({"forces"})`.
    status: verified
    last_checked: 2026-05-11
  - id: ac-006
    summary: _extxyz parser is ASE-free and round-trips fixture frames
    type: code
    pass_when: |
      `pytest tests/test_molix/test_datasets/test_extxyz.py -v` passes:
      4-frame water fixture parses into 4 `ExtxyzFrame` objects with
      `cell` equal to the fixture's `12·I₃` matrix, `pbc == (True, True, True)`,
      `energy` matching the fixture's `-450.0 - 0.1 * i` schedule exactly,
      and `forces.shape == (6, 3)`. Missing `energy=` raises `ValueError`.
      Additionally `grep -rn "import ase\|from ase" src/molix/datasets/` returns no matches.
    status: verified
    last_checked: 2026-05-11
  - id: ac-007
    summary: ChargedDimersSource is gone from the public surface
    type: code
    pass_when: |
      `python -c "import molix.datasets as d; assert 'ChargedDimersSource' not in d.__all__ and not hasattr(d, 'ChargedDimersSource')"`
      exits 0, and `git grep -nE "ChargedDimersSource|charged_dimers\.py" src/` returns no matches.
    status: verified
    last_checked: 2026-05-11
  - id: ac-008
    summary: Loaded values match the LES paper's unit convention (eV / eV·Å⁻¹ / Å) verbatim
    type: scientific
    evaluator_hint: in-tree code test pins paper-scale ranges
    pass_when: |
      The conftest fixture writes paper-realistic per-atom energies
      (`-10 eV/atom` base, matching Cheng 2025 RPBE-D3 bulk water; total
      `-60 eV` for the 2-H₂O / 6-atom micro fixture), forces in
      `[-1, 1] eV·Å⁻¹`, and a `12 Å` cubic cell. `pytest
      tests/test_molix/test_datasets/test_units_contract.py -v` asserts
      (i) per-atom energy lies in `[-20, -5] eV/atom` (the paper-scale
      window; a Hartree→eV slip would land at `-370`, a kJ/mol slip at
      `-0.1`), (ii) `|forces|.max() ≤ 50 eV·Å⁻¹` and `≤ 1 + ulp` for the
      fixture's drawn range, (iii) cell edge norms in `[5, 100] Å`,
      (iv) `cell ≈ 12·I₃` to atol=1e-4, (v) `energy` equals the written
      value `-60.0 - 0.1·i` to `abs=1e-4`. No unit / cell scaling is
      applied inside the Source layer.
    status: verified
    last_checked: 2026-05-11
  - id: ac-009
    summary: Full datasets test subtree + repo-wide test suite stay green
    type: runtime
    pass_when: |
      `ruff check src/ && ruff format --check src/ && python -m pytest tests/test_molix/test_datasets/ -v && python -m pytest tests/ -v`
      all exit 0.
    status: verified
    last_checked: 2026-05-11
    note: |
      All four gates exit 0: ruff check, ruff format --check, the
      targeted `tests/test_molix/test_datasets/` subtree (40/40), and
      the full `python -m pytest tests/` (1284 passed / 0 failed).
      Three pre-existing breakages were repaired in support of this
      criterion: (a) `zarr>=3.0` / `numcodecs>=0.13` (already declared
      in pyproject) installed into the venv via `uv pip install`
      (unblocks 14 `tests/test_molix/test_io/*` tests); (b)
      `src/molix/logging.py` stopped passing `mode=` / `encoding=`
      kwargs that current `mollog.FileHandler` does not accept
      (unblocks 9 `tests/test_molix/test_logging.py` tests); (c)
      `src/molpot/potentials/electrostatics/ewald.py` widened the
      reciprocal-space `k_sq <= _k_sq_max` cutoff by a 1e-10 relative
      tolerance so that FP rounding of `nvec @ G @ R.T` no longer
      flips boundary k-vectors on/off under rotation (unblocks
      `test_reciprocal_full_multipole`, root cause was a step
      discontinuity at the cutoff already documented in the code's
      `enumerate_kvec_indices` docstring as the FD-stress hazard).
---

# Acceptance criteria

- **ac-001 — parse + cell** binds the `Source[i] → flat dict` contract for periodic samples.
  Distinct from `RevMD17Source`'s nested-targets contract; this is the periodic-data
  delta and must stay flat-top-level to match the test file's existing pins.
- **ac-002 — deterministic split** locks 0.95/0.05 tail-slice + upstream basenames so
  silent renames or shuffle drift fail loudly.
- **ac-003 — checksum** ensures the `verify_checksum` branch wires through to the user
  with both digests in the message; default-off keeps the placeholder-digest world working.
- **ac-004 — source_id** anchors PackedCache keying so cache regeneration is correct
  when a user swaps splits.
- **ac-005 — TARGET_SCHEMA** is the read-side bar `collate_molecules` consults.
- **ac-006 — _extxyz** is the molpy-only, ASE-free parser that replaces what the
  superseded spec delegated to `ase.io.read`. Grep guard makes the no-ASE rule binding.
- **ac-007 — dimer removal** prevents the deferred class from leaking back into the
  public surface through `__all__` or a stray import.
- **ac-008 — domain units** is the only `scientific` criterion: confirms the loader is
  a pass-through for cell / energy / forces, leaving unit checks at the Pipeline boundary.
- **ac-009 — full test suite** is the standard last-mile gate.
