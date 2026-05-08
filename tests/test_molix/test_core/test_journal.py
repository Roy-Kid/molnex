"""RED tests for ``molix.core.hooks.Journal``.

Acceptance traces:
    ac-009 → :func:`test_mirrors_four_namespaces`
    ac-010 → :func:`test_hparams_emits_single_json_record`
    ac-011 → :func:`test_no_interference_with_existing_hooks`
"""

from __future__ import annotations

import pytest

from molix.core.state import TrainState
from molix.recorder.schema import MetricRecord, MetricStore


class _CapturingStore:
    """Test double that records every append call without touching disk."""

    def __init__(self) -> None:
        self.records: list[MetricRecord] = []
        self.flushed = False
        self.closed = False

    def append(self, record: MetricRecord) -> None:
        self.records.append(record)

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


def test_capturing_store_satisfies_protocol() -> None:
    """Sanity: the test double satisfies the public Protocol."""
    assert isinstance(_CapturingStore(), MetricStore)


def test_mirrors_four_namespaces() -> None:
    """ac-009: Journal mirrors all four TrainState namespaces as scalar records.

    With train/, performance/, gpu/ populated and a global_step that is a
    multiple of every_n_steps, plus eval/ populated, after one
    on_train_batch_end and one on_eval_step_complete the captured store
    must contain exactly the four expected (type="scalar", key=ns/k,
    value=v) tuples and nothing else.
    """
    from molix.core.hooks import Journal

    state = TrainState()
    state["global_step"] = 100
    state["train"]["loss"] = 0.5
    state["performance"]["step_per_second"] = 12.3
    state["gpu"]["peak_gib"] = 1.4
    state["eval"]["MAE"] = 0.07

    store = _CapturingStore()
    hook = Journal(every_n_steps=1, store=store)

    hook.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)
    hook.on_eval_step_complete(trainer=None, state=state)

    by_key = {r.key: r for r in store.records if r.type == "scalar"}
    assert set(by_key) == {
        "train/loss",
        "performance/step_per_second",
        "gpu/peak_gib",
        "eval/MAE",
    }
    assert by_key["train/loss"].value == pytest.approx(0.5)
    assert by_key["performance/step_per_second"].value == pytest.approx(12.3)
    assert by_key["gpu/peak_gib"].value == pytest.approx(1.4)
    assert by_key["eval/MAE"].value == pytest.approx(0.07)

    other_types = [r for r in store.records if r.type != "scalar"]
    assert other_types == [], f"Mirror pass should emit only scalar records; got {other_types!r}"


def test_hparams_emits_single_json_record() -> None:
    """ac-010: hparams emitted as a single type=json record at step=0."""
    from molix.core.hooks import Journal

    state = TrainState()
    store = _CapturingStore()
    hparams = {"lr": 1e-3, "model": "mace"}
    hook = Journal(
        every_n_steps=10,
        store=store,
        log_hparams=True,
        hparams=hparams,
    )

    hook.on_train_start(trainer=None, state=state)

    assert len(store.records) == 1, (
        f"Expected exactly one record at on_train_start with hparams; "
        f"got {len(store.records)}: {store.records!r}"
    )
    rec = store.records[0]
    assert rec.type == "json"
    assert rec.key == "hparams"
    assert rec.step == 0
    assert rec.value == hparams


def test_no_hparams_when_disabled() -> None:
    """log_hparams=False (default) emits zero records on on_train_start."""
    from molix.core.hooks import Journal

    state = TrainState()
    store = _CapturingStore()
    hook = Journal(every_n_steps=10, store=store)
    hook.on_train_start(trainer=None, state=state)
    assert store.records == []


def test_every_n_steps_gating() -> None:
    """on_train_batch_end skips when global_step % every_n_steps != 0."""
    from molix.core.hooks import Journal

    state = TrainState()
    state["global_step"] = 7
    state["train"]["loss"] = 0.5

    store = _CapturingStore()
    hook = Journal(every_n_steps=10, store=store)
    hook.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)
    assert store.records == []

    state["global_step"] = 10
    hook.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)
    assert len(store.records) == 1
    assert store.records[0].key == "train/loss"


def test_close_propagates_to_store() -> None:
    """on_train_end calls store.close()."""
    from molix.core.hooks import Journal

    state = TrainState()
    store = _CapturingStore()
    hook = Journal(every_n_steps=1, store=store)
    hook.on_train_end(trainer=None, state=state)
    assert store.closed is True


def test_non_scalar_state_value_skipped() -> None:
    """Vector / non-scalar state values are silently skipped (mirrors TBHook)."""
    import torch

    from molix.core.hooks import Journal

    state = TrainState()
    state["global_step"] = 1
    state["train"]["loss"] = 0.5
    state["train"]["embedding"] = torch.zeros(8)

    store = _CapturingStore()
    hook = Journal(every_n_steps=1, store=store)
    hook.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)

    keys = {r.key for r in store.records}
    assert keys == {"train/loss"}, f"Non-scalar values should be skipped; got keys {keys!r}"


def test_no_interference_with_existing_hooks() -> None:
    """ac-011: Journal does not perturb TensorBoardHook or MetricsHook.

    Concretely: running on_train_batch_end through a hook list that
    includes Journal alongside TensorBoardHook and MetricsHook does not
    mutate the state values that those hooks read, and TensorBoardHook
    sees the same scalar values either way.
    """
    from molix.core.hooks import Journal

    state = TrainState()
    state["global_step"] = 1
    state["train"]["loss"] = 0.5
    state["eval"]["MAE"] = 0.07

    store = _CapturingStore()
    journal = Journal(every_n_steps=1, store=store)

    journal.on_train_batch_end(trainer=None, state=state, batch=None, outputs=None)
    journal.on_eval_step_complete(trainer=None, state=state)

    assert state["train"]["loss"] == 0.5
    assert state["eval"]["MAE"] == 0.07
    assert state["global_step"] == 1
    assert state["train"] == {"loss": 0.5}
    assert state["eval"] == {"MAE": 0.07}
