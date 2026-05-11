---
slug: sonata-05-bench
criteria:
  - id: ac-001
    summary: BenchManifest validates the documented Literal fields and is frozen
    type: code
    pass_when: |
      tests/test_molix/test_bench/test_manifest.py covers (a) valid construction
      via from_dict and from_yaml for at least one allegro+train and one
      sonata+md payload, (b) pydantic.ValidationError on system="bogus" /
      model="bogus" / stage="bogus", and (c) ValidationError when assigning to
      an existing field on a constructed instance; all assertions pass.
    status: pending
  - id: ac-002
    summary: to_artifact_dir resolves the canonical runs/<system>/<model>/<stage>/ layout
    type: code
    pass_when: |
      tests/test_molix/test_bench/test_manifest.py shows that for a manifest
      with system="water_rpbe_d3", model="allegro", stage="train", runs_root=
      tmp_path, to_artifact_dir() returns tmp_path / "water_rpbe_d3" /
      "allegro" / "train" and the directory exists after the call;
      system_artifact_dir returns the same path without creating it.
    status: pending
  - id: ac-003
    summary: to_artifact_dir refuses to clobber a non-empty existing run
    type: code
    pass_when: |
      tests/test_molix/test_bench/test_manifest.py shows to_artifact_dir()
      raises FileExistsError when the target directory already contains a
      file and overwrite=False, and succeeds (no raise) when overwrite=True
      or when the target dir is empty / absent.
    status: pending
  - id: ac-004
    summary: model_from_manifest builds the Allegro baseline and runs a forward+grad pass
    type: code
    pass_when: |
      tests/test_molix/test_bench/test_factory.py constructs a manifest with
      model="allegro" and synthesises a 4-water periodic GraphBatch (reusing
      benchmarks/bm_molpot/bm_sonata.py::_build_synthetic_water_box), calls
      model_from_manifest(m)(batch, compute_forces=True), and asserts the
      result dict contains finite "energy" of shape (1,) and "forces" of
      shape (N, 3) matching batch atom count.
    status: pending
  - id: ac-005
    summary: model_from_manifest builds the Sonata composer with build_sonata defaults
    type: code
    pass_when: |
      tests/test_molix/test_bench/test_factory.py constructs a manifest with
      model="sonata" and the same synthetic water batch, calls
      model_from_manifest(m)(batch, compute_forces=True), and asserts the
      returned dict contains a finite "energy" tensor; additionally verifies
      that the call path goes through molpot.composition.build_sonata
      (e.g. via isinstance check against the type build_sonata returns or
      via monkeypatched spy).
    status: pending
  - id: ac-006
    summary: Equal seeds yield bit-identical Allegro baseline parameters across two factory calls
    type: code
    pass_when: |
      tests/test_molix/test_bench/test_factory.py invokes model_from_manifest
      twice with two BenchManifest instances sharing model="allegro", seed=0
      and identical hyperparams, then asserts torch.equal holds for every
      paired tensor in model_a.state_dict() and model_b.state_dict() (same
      keys, same values).
    status: pending
  - id: ac-007
    summary: Full check + test suite is green
    type: runtime
    pass_when: |
      `python -m pytest tests/test_molix/test_bench/ -v` exits 0 and the
      full `python -m pytest tests/ -v` run is no worse than before this
      sub-spec landed (no new failures).
    status: pending
---

# Acceptance criteria

`ac-001` … `ac-003` lock the Pydantic schema and the on-disk artifact contract that every downstream sub-spec (03 / 04 / 05) relies on — they are the read-only API of `BenchManifest`. `ac-004` and `ac-005` lock the factory's two branches against the canonical `_build_sonata_and_baseline` shape in `benchmarks/bm_molpot/bm_sonata.py`. `ac-006` is the bit-identical-init invariant that lets train-time and inference-time models share weights without surprise. `ac-007` is the standard runtime gate.
