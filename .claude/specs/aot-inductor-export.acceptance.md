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
    status: pending

  - id: ac-002
    summary: exported .so is loadable via aoti_load
    type: runtime
    evaluator_hint:
    pass_when: |
      `torch._inductor.aoti_load(so_path)` returns a callable without error.
    status: pending

  - id: ac-003
    summary: CPU export produces correct output when loaded
    type: runtime
    evaluator_hint:
    pass_when: |
      `export_model(model, inputs, tmp_path, device="cpu")`; loaded model
      output matches `model.forward(*inputs)` within atol=1e-5 on CPU.
    status: pending

  - id: ac-004
    summary: CUDA export produces correct output when loaded
    type: runtime
    evaluator_hint:
    pass_when: |
      If CUDA available: `export_model(model, inputs, tmp_path, device="cuda")`;
      loaded model output matches `model.forward(*inputs)` within atol=1e-5 on CUDA.
    status: pending

  - id: ac-005
    summary: device="auto" selects CUDA when available
    type: runtime
    evaluator_hint:
    pass_when: |
      `export_model(device="auto")` with CUDA available produces
      `model.meta.json` containing `"device": "cuda"`.
    status: pending

  - id: ac-006
    summary: full test suite passes
    type: code
    evaluator_hint:
    pass_when: |
      `python -m pytest tests/test_molix/test_export.py -v` exits with code 0.
    status: pending
---
