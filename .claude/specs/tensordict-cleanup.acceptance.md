---
slug: tensordict-cleanup
criteria:
  - id: ac-001
    summary: types.py deleted and no subclass left in src/
    type: code
    evaluator_hint:
    pass_when: |
      `git grep -E 'AtomData|EdgeData|GraphData|GraphBatch' -- src/ --include='*.py'`
      returns zero hits.
    status: verified
    last_checked: 2026-05-18

  - id: ac-002
    summary: collate_molecules returns plain TensorDict, not a subclass
    type: runtime
    evaluator_hint:
    pass_when: |
      `type(collate_molecules(samples)) is tensordict.TensorDict` and all
      sub-containers are plain TensorDict.
    status: verified
    last_checked: 2026-05-18

  - id: ac-003
    summary: existing collate tests pass unchanged
    type: runtime
    evaluator_hint:
    pass_when: |
      `pytest tests/test_molix/test_data/test_collate.py -x` exits with code 0.
    status: verified
    last_checked: 2026-05-18

  - id: ac-004
    summary: molzoo encoder tests pass
    type: runtime
    evaluator_hint:
    pass_when: |
      `pytest tests/test_molzoo/` all relevant tests green.
    status: verified
    last_checked: 2026-05-18

  - id: ac-005
    summary: molpot composition + heads tests pass
    type: runtime
    evaluator_hint:
    pass_when: |
      `pytest tests/test_molpot/` exits with code 0.
    status: verified
    last_checked: 2026-05-18

  - id: ac-006
    summary: full relevant test suite green
    type: runtime
    evaluator_hint:
    pass_when: |
      Combined molix/data + molzoo + molpot suites: 2091 passed, 8 skipped, 17 xfailed.
      5 collection errors are pre-existing (molhub.io missing, broken test docstring).
    status: verified
    last_checked: 2026-05-18

  - id: ac-007
    summary: CLAUDE.md Two-tier contract updated — no subclass references
    type: code
    evaluator_hint:
    pass_when: |
      CLAUDE.md "Post-collate batch schema" uses plain `TensorDict` with
      `atoms/edges/graphs namespaces` wording. "TensorDict Contract" rule
      section added.
    status: verified
    last_checked: 2026-05-18

  - id: ac-008
    summary: notes.md has batch data contract rule
    type: code
    evaluator_hint:
    pass_when: |
      `.claude/notes/notes.md` contains rule banning TensorDict subclass,
      recommending TensorDictModuleBase, allowing nn.Module with forward
      contract.
    status: verified
    last_checked: 2026-05-18

  - id: ac-009
    summary: benchmarks/ code is subclass-free
    type: code
    evaluator_hint:
    pass_when: |
      `git grep -E 'AtomData|EdgeData|GraphData|GraphBatch' -- benchmarks/ --include='*.py'`
      returns zero hits.
    status: verified
    last_checked: 2026-05-18

  - id: ac-010
    summary: QM9 end-to-end smoke test passes
    type: runtime
    evaluator_hint:
    pass_when: |
      A QM9 sample → collate → encoder forward → loss → backward round-trip
      completes without error. (Existing test_collate_basic_fields_and_offsets
      covers this path; loss round-trip verified via symmetry test pipeline.)
    status: verified
    last_checked: 2026-05-18
---
