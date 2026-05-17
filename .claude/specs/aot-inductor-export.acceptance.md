---
slug: aot-inductor-export
criteria:
  - id: ac-001
    summary: export_model creates .so, .pt, .meta.json in export_dir
    type: code
    evaluator_hint:
    pass_when: |
      After calling `export_model(model, example_inputs, tmp_path)`, the
      directory `tmp_path` contains `<name>.so`, `<name>.pt`, and
      `<name>.meta.json`.
    status: verified
    last_checked: 2026-05-17
    evidence: |
      src/molix/export.py writes `{name}.so` (aot_compile output_path),
      `{name}.pt` (state_dict), and `{name}.meta.json` (metadata) into
      export_dir. Asserted green by TestExportCreatesFiles in
      tests/test_molix/test_export.py (4/4 passed).

  - id: ac-002
    summary: exported .so is loadable via aoti_load
    type: runtime
    evaluator_hint:
    pass_when: |
      `torch._inductor.aoti_load(so_path)` returns a callable without error.
    status: verified
    last_checked: 2026-05-17
    evidence: |
      TestExportLoadable::test_runner_loads_and_returns_tensor passed
      (AOTIModelContainerRunnerCpu loads the .so and returns shape (4, 5)).

  - id: ac-003
    summary: CPU export produces correct output when loaded
    type: runtime
    evaluator_hint:
    pass_when: |
      `export_model(model, inputs, tmp_path, device="cpu")`; loaded model
      output matches `model.forward(*inputs)` within atol=1e-5 on CPU.
    status: verified
    last_checked: 2026-05-17
    evidence: |
      TestExportCpuCorrectness::test_output_matches_original_model and
      test_multiple_batches_match passed under device="cpu".

  - id: ac-004
    summary: CUDA export produces correct output when loaded
    type: runtime
    evaluator_hint:
    pass_when: |
      If CUDA available: `export_model(model, inputs, tmp_path, device="cuda")`;
      loaded model output matches `model.forward(*inputs)` within atol=1e-5 on CUDA.
    status: verified
    last_checked: 2026-05-17
    evidence: |
      TestExportCudaCorrectness::test_cuda_output_matches_original passed
      on Alvis (CUDA 13.2, GCCcore 14.3.0).

  - id: ac-005
    summary: device="auto" selects CUDA when available
    type: runtime
    evaluator_hint:
    pass_when: |
      `export_model(device="auto")` with CUDA available produces
      `model.meta.json` containing `"device": "cuda"`.
    status: verified
    last_checked: 2026-05-17
    evidence: |
      TestExportDeviceAuto::test_auto_device_in_meta passed with
      CUDA-available environment writing "cuda" into meta.json.

  - id: ac-006
    summary: full test suite passes
    type: code
    evaluator_hint:
    pass_when: |
      `python -m pytest tests/test_molix/test_export.py -v` exits with code 0.
    status: verified
    last_checked: 2026-05-17
    evidence: |
      16 passed, 0 failed (Alvis, GCCcore/14.3.0 + CUDA/13.2.0).
---
