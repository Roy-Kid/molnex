---
slug: sonata-05-hpc
criteria:
  - id: ac-001
    summary: wrap_for_ddp single-process fast path returns unwrapped model
    type: code
    pass_when: |
      pytest tests/test_molix/test_core/test_ddp.py::test_wrap_single_process_fast_path
      passes; the returned model is `model` itself (identity check), the returned
      DistEnv has world_size=1 and initialized=False, and dist.is_initialized()
      remains False after the call.
    status: pending
  - id: ac-002
    summary: teardown_ddp is no-op when env.initialized=False
    type: code
    pass_when: |
      pytest tests/test_molix/test_core/test_ddp.py::test_teardown_noop_when_uninitialized
      passes; calling teardown_ddp(DistEnv(initialized=False)) does not raise and
      does not call dist.destroy_process_group.
    status: pending
  - id: ac-003
    summary: rank/world helpers safe when dist is uninitialized
    type: code
    pass_when: |
      pytest tests/test_molix/test_core/test_ddp.py::test_helpers_uninitialized
      passes; is_rank_zero() returns True, get_rank() returns 0, get_world_size()
      returns 1, barrier() returns None without raising — all without calling
      dist.* internals that require initialization.
    status: pending
  - id: ac-004
    summary: CheckpointHook does not write files when not rank 0
    type: code
    pass_when: |
      pytest tests/test_molix/test_hooks/test_rank_zero_guards.py::test_checkpoint_hook_guarded
      passes; with `is_rank_zero` monkeypatched to False, CheckpointHook.on_train_start
      does not call os.makedirs and CheckpointHook._save_checkpoint does not call
      torch.save (verified via mock assertions).
    status: pending
  - id: ac-005
    summary: JournalHook and TensorBoardHook do not emit IO when not rank 0
    type: code
    pass_when: |
      pytest tests/test_molix/test_hooks/test_rank_zero_guards.py::test_journal_and_tensorboard_guarded
      passes; with `is_rank_zero` monkeypatched to False, no JournalWriter.append
      call is made by JournalHook lifecycle methods, and no SummaryWriter.add_scalar
      / add_histogram call is made by TensorBoardHook lifecycle methods.
    status: pending
  - id: ac-006
    summary: 2-rank MLP forward+backward matches single-rank reference within atol=1e-5
    type: runtime
    evaluator_hint: multi-gpu-host
    pass_when: |
      pytest -m multi_gpu tests/test_molix/test_core/test_ddp_multi_rank.py::test_mlp_two_rank_equivalence
      passes on a host with >=2 CUDA devices; final model parameters after one
      training step on 2 ranks (with DistributedSampler) match the single-rank
      reference run on the broadcast batch to atol=1e-5 / rtol=1e-5.
    status: pending
  - id: ac-007
    summary: Sonata(compute_forces=True) is DDP-safe under 2 ranks (autograd-in-autograd)
    type: runtime
    evaluator_hint: multi-gpu-host
    pass_when: |
      pytest -m multi_gpu tests/test_molix/test_core/test_ddp_multi_rank.py::test_sonata_compute_forces_two_rank
      passes on a host with >=2 CUDA devices; energy and forces output by
      Sonata(compute_forces=True) on 2 ranks match the single-rank reference
      to atol=1e-5; one full backward pass completes without DDP reducer
      assertion errors and the optimizer step succeeds.
    status: pending
  - id: ac-008
    summary: 2-rank checkpoint run produces exactly one last.pt
    type: runtime
    evaluator_hint: multi-gpu-host
    pass_when: |
      pytest -m multi_gpu tests/test_molix/test_core/test_ddp_multi_rank.py::test_checkpoint_single_writer
      passes on a host with >=2 CUDA devices; after a 2-epoch 2-rank training
      with CheckpointHook(checkpoint_dir=tmp_dir, save_last=True), exactly one
      file named "last.pt" exists in tmp_dir, it is loadable via torch.load,
      and its global_step matches the rank-0 trainer state.
    status: pending
  - id: ac-009
    summary: full check + test suite passes (CPU paths)
    type: runtime
    pass_when: |
      `python -m pytest tests/ -v -m "not multi_gpu"` exits 0; no new lint or
      type errors are introduced relative to the dev branch baseline.
    status: pending
  - id: ac-010
    summary: docstrings present on new public symbols and updated hooks
    type: docs
    pass_when: |
      Every public symbol in src/molix/core/ddp.py (DistEnv, wrap_for_ddp,
      teardown_ddp, is_rank_zero, get_rank, get_world_size, barrier) carries a
      Google-style docstring with Args / Returns; Trainer class + Trainer.train
      docstrings mention wrap_for_ddp; JournalWriter docstring carries the
      "callers must rank-0-guard" note.
    status: pending
---

# Acceptance criteria

ac-001 — ac-003 lock down the single-process / non-distributed fallback so
the new helpers degrade safely on a laptop CPU run with no `WORLD_SIZE` env var.

ac-004 — ac-005 lock down the rank-0 guard audit on the four IO hooks; a
regression here is the *exact* failure mode (corrupted checkpoint, racing
event-file writes) that motivated this sub-spec.

ac-006 — ac-008 are the multi-rank smokes, gated on a host with ≥ 2 CUDA
devices. ac-007 is load-bearing: it is the *only* place where Sonata's
autograd-inside-autograd path is exercised under DDP; without it,
`sonata-05-driver` would discover the regression at the much more expensive
end-to-end bench layer.

ac-009 enforces baseline regression on CPU CI; ac-010 enforces the
Google-style docstring discipline (CLAUDE.md "Docstring Convention").
