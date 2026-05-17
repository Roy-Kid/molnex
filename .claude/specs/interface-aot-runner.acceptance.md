---
slug: interface-aot-runner
criteria:
  - id: ac-001
    summary: libmolnex_interface builds successfully
    type: code
    evaluator_hint:
    pass_when: |
      CMake build produces `libmolnex_interface.a` under `interface/build/`.
    status: verified
    last_checked: 2026-05-17

  - id: ac-002
    summary: ModelRunner.run() matches reference output
    type: runtime
    evaluator_hint:
    pass_when: |
      C++ test loads an exported model, calls `runner.run(inputs)`, output
      matches PyTorch reference within atol=1e-5.
    status: verified
    last_checked: 2026-05-17

  - id: ac-003
    summary: update_weights via file path applies new weights correctly
    type: runtime
    evaluator_hint:
    pass_when: |
      `runner.update_weights("new_weights.pt")` then `runner.run(inputs)`
      matches forward of model with those weights within atol=1e-5.
    status: verified
    last_checked: 2026-05-17

  - id: ac-004
    summary: update_weights via tensor map applies new weights correctly
    type: runtime
    evaluator_hint:
    pass_when: |
      `runner.update_weights(param_map)` then `runner.run(inputs)` matches
      forward of model with those weights within atol=1e-5.
    status: verified
    last_checked: 2026-05-17

  - id: ac-005
    summary: update_weights concurrent with run does not error
    type: runtime
    evaluator_hint:
    pass_when: |
      10× `update_weights()` interleaved with 100× `run()` complete without
      exception; all outputs are valid tensors.
    status: verified
    last_checked: 2026-05-17

  - id: ac-006
    summary: CPU runner works correctly
    type: runtime
    evaluator_hint:
    pass_when: |
      CPU-exported model loads with correct device, `run()` output matches
      reference.
    status: verified
    last_checked: 2026-05-17

  - id: ac-007
    summary: CUDA runner works correctly when GPU available
    type: runtime
    evaluator_hint:
    pass_when: |
      CUDA-exported model loads with correct device, `run()` output matches
      reference. Skipped if no GPU.
    status: verified
    last_checked: 2026-05-17

  - id: ac-008
    summary: parameter_info returns correct names and dtypes
    type: runtime
    evaluator_hint:
    pass_when: |
      `runner.parameter_info()` returns a list matching the model's
      `state_dict` keys and scalar types.
    status: verified
    last_checked: 2026-05-17

  - id: ac-009
    summary: full C++ test suite passes
    type: code
    evaluator_hint:
    pass_when: |
      `cd interface/build && ctest --output-on-failure` exits with code 0.
    status: verified
    last_checked: 2026-05-17
---
