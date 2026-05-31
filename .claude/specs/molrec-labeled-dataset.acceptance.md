---
slug: molrec-labeled-dataset
criteria:
  - id: ac-001
    summary: molrs.MolRec API pinned against installed wheel before coding
    type: docs
    pass_when: |
      src/molix/datasets/molrec.py module docstring records the verified
      molrs.MolRec API surface used (read_zarr/write_zarr, trajectory frame
      layout, observables map + ObservableRecord kind/axes, method JSON
      access), and the code imports only names that exist in the installed
      molrs wheel.
    status: pending

  - id: ac-002
    summary: MolRecSource conforms to DataSource Protocol without subclassing
    type: code
    pass_when: |
      isinstance(MolRecSource(record_path, teacher_id="teacherA"), DataSource)
      is True at runtime, and MolRecSource is not a subclass of any class in
      molix.data.source.
    status: verified
    last_checked: 2026-05-31

  - id: ac-003
    summary: __getitem__ returns flat sample with dynamic prefix-stripped targets
    type: code
    pass_when: |
      source[0] returns a dict with keys Z, pos, targets (and optional box);
      pos has shape (N, 3); targets keys equal the observable keys prefixed
      "teacherA." with the prefix stripped (e.g. "energy", "forces").
    status: verified
    last_checked: 2026-05-31

  - id: ac-004
    summary: dynamic per-instance TargetSchema built from observable kinds
    type: code
    pass_when: |
      source.target_schema (or equivalent per-instance attribute) places
      scalar observables in graph_level and vector force observables in
      atom_level; no frozen class-attr TARGET_SCHEMA is defined on
      MolRecSource.
    status: verified
    last_checked: 2026-05-31

  - id: ac-005
    summary: source_id appends :teacher= for per-teacher cache invalidation
    type: code
    pass_when: |
      MolRecSource(..., teacher_id="teacherA").source_id ends with
      ":teacher=teacherA" and differs from the source_id for teacher_id=
      "teacherB" on the same record; the prefix matches the RevMD17Source
      "<name>:size=<bytes>:n=<n>[:total=]" shape.
    status: verified
    last_checked: 2026-05-31

  - id: ac-006
    summary: unknown teacher_id fails fast listing available teacher prefixes
    type: runtime
    pass_when: |
      Constructing MolRecSource with a teacher_id absent from the record
      raises an error whose message enumerates the available teacher
      prefixes discovered in the observables map.
    status: verified
    last_checked: 2026-05-31

  - id: ac-007
    summary: QM9 roundtrip lossless for 15 scalars with float32 preserved
    type: scientific
    pass_when: |
      Writing a QM9 slice (Z, pos, 15 scalar targets) to a MolRec zarr and
      reading it back via MolRecSource yields the 15 scalar values bit-equal
      (or within float32 eps) and dtype == float32 (no upcast); test passes
      in tests/test_molix/test_datasets/test_molrec.py.
    status: pending

  - id: ac-008
    summary: force-bearing roundtrip preserves per-atom force shape/values
    type: scientific
    pass_when: |
      Writing a force-bearing slice (energy + per-atom forces vector
      observable, axes timestep/atom/component) and reading back yields
      per-frame forces of shape [n_atoms, 3] with values lossless and dtype
      preserved; test passes in test_molrec.py.
    status: pending

  - id: ac-009
    summary: public docstrings carry units, shapes, DOIs and schema departure note
    type: docs
    pass_when: |
      MolRecSource and its public methods have Google-style docstrings with
      tensor shapes and units; module docstring cites QM9 (10.1038/sdata.2014.22)
      and rMD17 (10.1088/2632-2153/abba6f) DOIs and notes the dynamic
      per-instance TargetSchema departure from the other sources.
    status: pending

  - id: ac-010
    summary: full lint + format + test suite passes
    type: runtime
    pass_when: |
      `ruff check src/ && ruff format --check src/` exits 0 and
      `python -m pytest tests/test_molix/test_datasets/test_molrec.py -v`
      passes with no failures.
    status: verified
    last_checked: 2026-05-31
---

# Acceptance criteria

- **ac-001** (docs): API pin precedes coding — the docstring is the durable trace that the molrs surface was verified against the wheel, not assumed.
- **ac-002 / ac-003 / ac-004 / ac-005** (code): the Protocol-conformance, flat-sample, dynamic-schema, and per-teacher-`source_id` contracts that `/mol:impl` verifies via the unit tests.
- **ac-006** (runtime): fail-fast UX requires observing the raised error message at runtime.
- **ac-007 / ac-008** (scientific): the two roundtrip fidelity gates — the load-bearing proof that the flat-name + vector-observable convention survives a molrs zarr roundtrip without dtype/shape/value drift.
- **ac-009** (docs): scientific-correctness + docstring-convention compliance.
- **ac-010** (runtime): the standard check + test gate.
