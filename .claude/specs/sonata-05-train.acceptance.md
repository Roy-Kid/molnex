---
slug: sonata-05-train
criteria:
  - id: ac-001
    summary: MultipoleDiagnosticHook publishes ten state["train"] scalar keys
    type: code
    pass_when: |
      `pytest tests/test_molix/test_hooks/test_multipole_diagnostic.py::test_scalar_keys -v`
      passes; after one synthetic `on_train_batch_end` call with a
      `GraphBatch` carrying `("atoms","q_perm")`, `("atoms","mu_perm")`,
      `("atoms","theta_perm")`, `("graphs","E_short")`,
      `("graphs","E_elec")`, the hook writes exactly these ten keys
      into `state["train"]`: `q_perm_mean`, `q_perm_max_abs`,
      `q_net_abs_per_atom`, `mu_perm_mean`, `mu_perm_max_abs`,
      `theta_perm_mean`, `theta_perm_max_abs`, `E_short`, `E_elec`,
      `E_elec_over_short`.
    status: pending
  - id: ac-002
    summary: MultipoleDiagnosticHook gracefully skips when keys missing
    type: code
    pass_when: |
      `pytest tests/test_molix/test_hooks/test_multipole_diagnostic.py::test_missing_keys_skip -v`
      passes; calling `on_train_batch_end` on a batch without
      `("atoms","q_perm")` does not raise and does not write multipole
      keys to `state["train"]`.
    status: pending
  - id: ac-003
    summary: run_train allegro smoke mode produces metrics.json on 4-water fixture
    type: runtime
    pass_when: |
      `pytest tests/test_molix/test_bench/test_drivers_train.py::test_run_train_allegro_smoke -v`
      passes; `run_train(manifest)` with `model="allegro"`,
      `max_epochs=1`, `max_steps=100`, the 4-water micro fixture and
      no network access returns a `Path` whose `metrics.json` exists
      and parses to a dict containing keys `energy_mae_meV_per_atom`,
      `force_rmse_meV_per_A`, `force_rmse_long_range_meV_per_A`; the
      third key is `None` for the allegro branch.
    status: pending
  - id: ac-004
    summary: run_train sonata smoke mode wires MultipoleDiagnosticHook
    type: runtime
    pass_when: |
      `pytest tests/test_molix/test_bench/test_drivers_train.py::test_run_train_sonata_smoke -v`
      passes; `run_train(manifest)` with `model="sonata"` on the
      4-water micro fixture returns a `Path`; `metrics.json` is
      present; the final `state["train"]` (captured by a test-only
      tap hook) contains all ten MultipoleDiagnosticHook keys; and
      `force_rmse_long_range_meV_per_A` is a finite float.
    status: pending
  - id: ac-005
    summary: run_train uses MmapDataset via DataModule(dataset_cls=...)
    type: code
    pass_when: |
      `pytest tests/test_molix/test_bench/test_drivers_train.py::test_datamodule_uses_mmap_dataset -v`
      passes; static assertion confirms `run_train` passes
      `dataset_cls=MmapDataset` to `DataModule.__init__` (verified by
      monkeypatching `DataModule` and inspecting the captured kwarg)
      per the `feedback_dataset_base_class.md` memory rule.
    status: pending
  - id: ac-006
    summary: run_train walks wrap_for_ddp single-process fast path with WORLD_SIZE unset
    type: code
    pass_when: |
      `pytest tests/test_molix/test_bench/test_drivers_train.py::test_no_ddp_init_when_world_size_unset -v`
      passes; with `WORLD_SIZE` absent from the environment,
      `run_train` invokes `wrap_for_ddp` exactly once and
      `torch.distributed.init_process_group` is **not** invoked
      (verified by monkeypatch counter).
    status: pending
  - id: ac-007
    summary: MetricsHook train and eval instances are independent
    type: code
    pass_when: |
      `pytest tests/test_molix/test_bench/test_drivers_train.py::test_metrics_hook_deep_copied -v`
      passes; the `MetricsHook` instance registered for train and the
      one registered for eval are distinct objects (`id(train_h) !=
      id(eval_h)`) and their internal accumulator buffers are
      distinct (mutating one does not affect the other), satisfying
      the CLAUDE.md State namespace contract rule 4.
    status: pending
  - id: ac-008
    summary: Sonata force-RMSE on bulk-water test split within +20 % of Allegro baseline
    type: performance
    evaluator_hint: scientist
    pass_when: |
      On a full-mode (non-CI) `run_train` against the real
      RPBE-D3 bulk-water dataset, the test-split metrics.json
      satisfies `force_rmse_meV_per_A <= 32.1 * 1.20` meV·Å⁻¹.
      Baseline 32.1 meV·Å⁻¹ per *J. Chem. Phys.* 163:104102 (2025);
      the 20 % tolerance is engineering judgment, not literature.
      Reported via `metrics.json` and verified by the implementer or
      compute-scientist agent reviewing the artifact directory.
    status: pending
  - id: ac-009
    summary: MultipoleDiagnosticHook bounds hold over final 10 % of training
    type: scientific
    evaluator_hint: scientist
    pass_when: |
      On the same full-mode `run_train` (sonata branch), the
      window-averaged scalars over the final 10 % of training
      steps satisfy: (a) `q_net_abs_per_atom < 1e-4 e`,
      (b) `q_perm_max_abs < 5.0 e`, (c) `E_elec_over_short < 1.0`.
      Captured from `state["train"]` at training end and written
      into `metrics.json` under
      `{"window_q_net_abs_per_atom","window_q_perm_max_abs",
      "window_E_elec_over_short"}`; verified by the test
      `tests/test_molix/test_bench/test_drivers_train.py::test_window_bounds_smoke`
      using the 4-water fixture (smoke bounds slightly loosened
      per fixture scale, full bounds asserted off-CI).
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

ac-001 / ac-002 — `MultipoleDiagnosticHook` 的 scalar 契约与缺键韧性。
ac-003 / ac-004 — `run_train` 在 allegro / sonata 两分支 smoke 模式下落盘 `metrics.json` 与 hook 串联正确。
ac-005 — DataModule 接 `MmapDataset` 而非 hard-code `CachedDataset`（CLAUDE.md memory 规则）。
ac-006 — 无 `WORLD_SIZE` 时走 single-process fast path（与 `sonata-05-hpc` 一致）。
ac-007 — train/eval 两个 `MetricsHook` 实例物理上互相独立（CLAUDE.md TrainState invariant 第 4 条）。
ac-008 — Sonata 测试集 force-RMSE 物理基线（performance）。
ac-009 — 多极/电中性/E_elec 比值物理健康范围（scientific），由 hook 报出。
ac-010 — 仓库 lint + 全量测试通过。

provenance：ac-001/ac-002/ac-005/ac-007 来自 CLAUDE.md（State namespace contract、dataset_cls memory）；ac-006 来自 `.claude/specs/sonata-05-hpc.md` § Design；ac-008 来自 *J. Chem. Phys.* 163:104102 (2025)（基线值）+ scientist 工程容差（×1.20）；ac-009 工程容差（净电荷漂移 1e-4 e / 多极坍缩 5 e / E_elec 比值 1）来自 scientist 输出，非文献锚定具体数值；ac-010 来自 CLAUDE.md § Build & Development。
