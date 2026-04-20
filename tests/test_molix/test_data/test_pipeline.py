"""Tests for the PipelineSpec / Pipeline builder.

PipelineSpec owns both declaration (task list, identity) and orchestration
(``.run`` / ``.transform`` / ``.cache_key`` / ``.build_cache`` / ``.cache``).
TaskEntry owns task dispatch (``.apply``).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from molix.data.cache import PackedCache
from molix.data.pipeline import Pipeline, PipelineSpec, TaskEntry
from molix.data.source import InMemorySource
from molix.data.task import BatchTask, DatasetTask, SampleTask


# ---------------------------------------------------------------------------
# Stub tasks
# ---------------------------------------------------------------------------


class CountingSample(SampleTask):
    def __init__(self, name: str = "counter") -> None:
        self._name = name
        self.calls = 0

    @property
    def task_id(self) -> str:
        return f"counting:{self._name}"

    def execute(self, data: dict) -> dict:
        self.calls += 1
        return {**data, self._name: True}


class MeanShift(DatasetTask):
    def __init__(self, key: str = "y") -> None:
        self.key = key
        self.mean = 0.0
        self.fit_calls = 0
        self.exec_calls = 0

    @property
    def task_id(self) -> str:
        return f"mean_shift:{self.key}"

    def fit(self, samples: list[dict]) -> None:
        self.fit_calls += 1
        ys = [float(s[self.key].item()) for s in samples]
        self.mean = sum(ys) / len(ys)

    def execute(self, data: dict) -> dict:
        self.exec_calls += 1
        return {**data, self.key: data[self.key] - self.mean}

    def state_dict(self) -> dict[str, Any]:
        return {"mean": torch.tensor(self.mean, dtype=torch.float64)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = float(state["mean"].item())


class NoopBatch(BatchTask):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def task_id(self) -> str:
        return "noop:batch"

    def execute(self, data: dict) -> dict:
        self.calls += 1
        return data


def _samples(n: int = 8) -> list[dict]:
    return [
        {
            "Z": torch.tensor([1, 6]),
            "pos": torch.zeros(2, 3),
            "y": torch.tensor([float(i)]),
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Builder DSL
# ---------------------------------------------------------------------------


class TestBuilder:
    def test_add_returns_self_for_chaining(self):
        p = Pipeline("x")
        assert p.add(CountingSample()) is p

    def test_add_task_instance(self):
        spec = Pipeline("p").add(CountingSample("foo")).build()
        assert len(spec.tasks) == 1
        assert spec.tasks[0].task.task_id == "counting:foo"

    def test_add_bare_callable(self):
        spec = Pipeline("p").add(lambda s: s, name="noop").build()
        assert spec.tasks[0].name == "noop"

    def test_decorator_task(self):
        p = Pipeline("p")

        @p.task
        def add_tag(s: dict) -> dict:
            return {**s, "tag": True}

        spec = p.build()
        assert spec.tasks[0].name == "add_tag"

    def test_decorator_with_explicit_name(self):
        p = Pipeline("p")

        @p.task(name="custom")
        def _f(s: dict) -> dict:
            return s

        spec = p.build()
        assert spec.tasks[0].name == "custom"

    def test_rejects_non_task_non_callable(self):
        with pytest.raises(TypeError, match="Task"):
            Pipeline("p").add(42)    # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_prepare_tasks_include_sample_and_dataset(self):
        spec = (
            Pipeline("p")
            .add(CountingSample())
            .add(MeanShift())
            .add(NoopBatch())
            .build()
        )
        assert len(spec.prepare_tasks) == 2
        assert len(spec.batch_tasks) == 1

    def test_plain_callable_counts_as_prepare(self):
        spec = Pipeline("p").add(lambda s: s, name="noop").build()
        assert len(spec.prepare_tasks) == 1
        assert len(spec.batch_tasks) == 0

    def test_prepare_tasks_order_preserved(self):
        a, b = CountingSample("a"), CountingSample("b")
        spec = Pipeline("p").add(a).add(b).build()
        assert [e.task for e in spec.prepare_tasks] == [a, b]


# ---------------------------------------------------------------------------
# pipeline_id determinism
# ---------------------------------------------------------------------------


class TestPipelineId:
    def test_stable_across_builds(self):
        a = Pipeline("p").add(CountingSample("a")).add(MeanShift("y")).build()
        b = Pipeline("p").add(CountingSample("a")).add(MeanShift("y")).build()
        assert a.pipeline_id == b.pipeline_id

    def test_task_order_changes_id(self):
        a = Pipeline("p").add(CountingSample("a")).add(MeanShift("y")).build()
        b = Pipeline("p").add(MeanShift("y")).add(CountingSample("a")).build()
        assert a.pipeline_id != b.pipeline_id

    def test_task_config_changes_id(self):
        a = Pipeline("p").add(MeanShift("y")).build()
        b = Pipeline("p").add(MeanShift("z")).build()
        assert a.pipeline_id != b.pipeline_id

    def test_pipeline_name_changes_id(self):
        a = Pipeline("a").build()
        b = Pipeline("b").build()
        assert a.pipeline_id != b.pipeline_id

    def test_id_is_hex_and_short(self):
        pid = Pipeline("p").add(CountingSample()).build().pipeline_id
        assert len(pid) == 16
        int(pid, 16)


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    def test_contains_task_names_and_types(self):
        spec = (
            Pipeline("p")
            .add(CountingSample("a"))
            .add(NoopBatch())
            .build()
        )
        d = spec.to_dict()
        assert d["name"] == "p"
        assert d["pipeline_id"] == spec.pipeline_id
        names = [t["name"] for t in d["tasks"]]
        types = [t["type"] for t in d["tasks"]]
        assert names == ["counting:a", "noop:batch"]
        assert types == ["CountingSample", "NoopBatch"]


# ---------------------------------------------------------------------------
# TaskEntry
# ---------------------------------------------------------------------------


class TestTaskEntry:
    def test_is_frozen(self):
        e = TaskEntry(name="x", task=CountingSample())
        with pytest.raises(Exception):    # FrozenInstanceError or AttributeError
            e.name = "y"                  # type: ignore[misc]

    def test_apply_dispatches_runnable(self):
        t = CountingSample()
        entry = TaskEntry(name="t", task=t)
        out = entry.apply({"y": torch.tensor([0.0])})
        assert out["counter"] is True
        assert t.calls == 1

    def test_apply_dispatches_bare_callable(self):
        entry = TaskEntry(name="f", task=lambda s: {**s, "tag": True})
        out = entry.apply({"y": 0})
        assert out["tag"] is True

    def test_apply_rejects_non_callable(self):
        entry = TaskEntry(name="bad", task=42)    # type: ignore[arg-type]
        with pytest.raises(TypeError):
            entry.apply({})


# ---------------------------------------------------------------------------
# transform — per-sample inference path
# ---------------------------------------------------------------------------


class TestTransform:
    def test_applies_prepare_tasks_in_order(self):
        counter = CountingSample("a")
        spec = Pipeline("p").add(counter).build()
        out = spec.transform({"y": torch.tensor([0.0])})
        assert out["a"] is True
        assert counter.calls == 1

    def test_skips_batch_tasks(self):
        counter = CountingSample("a")
        batch = NoopBatch()
        spec = Pipeline("p").add(counter).add(batch).build()
        spec.transform({"y": torch.tensor([0.0])})
        assert counter.calls == 1
        assert batch.calls == 0


# ---------------------------------------------------------------------------
# run — full-source iteration with fit()
# ---------------------------------------------------------------------------


class TestRun:
    def test_sample_task_once_per_sample(self):
        counter = CountingSample()
        spec = Pipeline("p").add(counter).build()
        out = list(spec.run(InMemorySource(_samples(10))))
        assert len(out) == 10
        assert counter.calls == 10

    def test_dataset_task_fit_on_full_source(self):
        shift = MeanShift()
        spec = Pipeline("p").add(shift).build()
        list(spec.run(InMemorySource(_samples(8))))
        assert shift.fit_calls == 1
        assert shift.exec_calls == 8

    def test_fit_source_scopes_fit(self):
        shift = MeanShift()
        spec = Pipeline("p").add(shift).build()
        list(
            spec.run(
                InMemorySource(_samples(10)),   # y = 0..9
                fit_source=InMemorySource(_samples(5)),  # y = 0..4 → mean = 2
            )
        )
        assert shift.fit_calls == 1
        assert abs(shift.mean - 2.0) < 1e-9

    def test_returns_iterator(self):
        import types
        spec = Pipeline("p").add(CountingSample()).build()
        it = spec.run(InMemorySource(_samples(3)))
        assert isinstance(it, types.GeneratorType)


# ---------------------------------------------------------------------------
# collect_task_states / load_task_states
# ---------------------------------------------------------------------------


class TestTaskStates:
    def test_collect_includes_only_dataset_tasks(self):
        spec = (
            Pipeline("p")
            .add(CountingSample("s"))
            .add(MeanShift("y"))
            .build()
        )
        list(spec.run(InMemorySource(_samples(4))))
        states = spec.collect_task_states()
        assert set(states) == {"mean_shift:y"}

    def test_load_restores_state(self):
        # Fit on one spec, dump, load into a fresh spec.
        a = MeanShift()
        spec_a = Pipeline("p").add(a).build()
        list(spec_a.run(InMemorySource(_samples(6))))    # y=0..5 → mean=2.5
        states = spec_a.collect_task_states()

        b = MeanShift()
        spec_b = Pipeline("p").add(b).build()
        spec_b.load_task_states(states)
        assert abs(b.mean - 2.5) < 1e-9


# ---------------------------------------------------------------------------
# cache_key — PipelineSpec-level identity
# ---------------------------------------------------------------------------


class _FakeSource:
    """Minimal object with a .source_id attribute + len + indexing."""

    def __init__(self, source_id: str, samples: list[dict] | None = None) -> None:
        self.source_id = source_id
        self._samples = samples or []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]


class TestCacheKey:
    def test_source_id_affects_key(self):
        spec = Pipeline("p").build()
        a = spec.cache_key(_FakeSource("s1"))
        b = spec.cache_key(_FakeSource("s2"))
        assert a != b

    def test_fit_source_affects_key(self):
        spec = Pipeline("p").build()
        src = _FakeSource("s")
        full_only = spec.cache_key(src)
        with_fit = spec.cache_key(src, fit_source=_FakeSource("s-subset"))
        assert full_only != with_fit

    def test_extra_affects_key(self):
        spec = Pipeline("p").build()
        a = spec.cache_key(_FakeSource("s"))
        b = spec.cache_key(_FakeSource("s"), extra={"n_train": "100"})
        assert a != b

    def test_pipeline_id_affects_key(self):
        a = Pipeline("p1").add(CountingSample()).build().cache_key(_FakeSource("s"))
        b = Pipeline("p2").add(CountingSample()).build().cache_key(_FakeSource("s"))
        assert a != b


# ---------------------------------------------------------------------------
# build_cache — unconditional run + save
# ---------------------------------------------------------------------------


class _Tag(SampleTask):
    @property
    def task_id(self) -> str:
        return "tag"

    def execute(self, data: dict) -> dict:
        return {**data, "tagged": torch.tensor(1)}


class TestBuildCache:
    def test_runs_and_saves(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).build()
        sink = tmp_path / "x.pt"
        cache = spec.build_cache(InMemorySource(_samples(4)), sink)

        assert cache.sink == sink
        loaded = cache.load()
        assert len(loaded["samples"]) == 4
        assert all("tagged" in s for s in loaded["samples"])

    def test_saves_task_states(self, tmp_path):
        shift = MeanShift("y")
        spec = Pipeline("p").add(shift).build()
        sink = tmp_path / "x.pt"
        spec.build_cache(InMemorySource(_samples(4)), sink)   # y=0..3 → mean=1.5

        loaded = PackedCache(sink).load()
        assert torch.allclose(
            loaded["task_states"]["mean_shift:y"]["mean"].double(),
            torch.tensor(1.5, dtype=torch.float64),
            atol=1e-9,
        )

    def test_fit_source_only_sees_subset(self, tmp_path):
        """Regression for the scientific-correctness bug in the old materialize()."""
        shift = MeanShift("y")
        spec = Pipeline("p").add(shift).build()

        # Full source y=0..9, fit_source y=0..4 → mean should be 2.0 not 4.5.
        spec.build_cache(
            InMemorySource(_samples(10)),
            tmp_path / "x.pt",
            fit_source=InMemorySource(_samples(5)),
        )
        payload = PackedCache(tmp_path / "x.pt").load()
        state = payload["task_states"]["mean_shift:y"]
        assert abs(float(state["mean"].item()) - 2.0) < 1e-9

    def test_is_idempotent(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).build()
        sink = tmp_path / "x.pt"
        spec.build_cache(InMemorySource(_samples(2)), sink)
        mtime = sink.stat().st_mtime_ns
        spec.build_cache(InMemorySource(_samples(99)), sink)   # no-op
        assert sink.stat().st_mtime_ns == mtime

    def test_overwrite(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).build()
        sink = tmp_path / "x.pt"
        spec.build_cache(InMemorySource(_samples(2)), sink)
        spec.build_cache(InMemorySource(_samples(5)), sink, overwrite=True)
        assert len(PackedCache(sink).load()["samples"]) == 5

    def test_accepts_packed_cache_argument(self, tmp_path):
        spec = Pipeline("p").add(_Tag()).build()
        packed = PackedCache(tmp_path / "x.pt")
        out = spec.build_cache(InMemorySource(_samples(3)), packed)
        assert out is packed
        assert len(out.load()["samples"]) == 3


# ---------------------------------------------------------------------------
# cache — DDP-aware orchestration
# ---------------------------------------------------------------------------


class TestCacheDDP:
    def test_primary_rank_builds(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        spec = Pipeline("p").add(_Tag()).build()
        src = _FakeSource("s", _samples(3))
        packed = spec.cache(src, base_dir=tmp_path)
        assert packed.is_ready()
        assert len(packed.load()["samples"]) == 3

    def test_missing_rank_treated_as_primary(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        spec = Pipeline("p").add(_Tag()).build()
        src = _FakeSource("s", _samples(2))
        packed = spec.cache(src, base_dir=tmp_path)
        assert packed.is_ready()

    def test_malformed_rank_treated_as_primary(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "not-a-number")
        spec = Pipeline("p").add(_Tag()).build()
        src = _FakeSource("s", _samples(2))
        packed = spec.cache(src, base_dir=tmp_path)
        assert packed.is_ready()

    def test_non_primary_waits_and_raises_on_timeout(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "1")
        spec = Pipeline("p").add(_Tag()).build()
        src = _FakeSource("s", _samples(1))
        # Shrink PackedCache.wait_until_ready so the test runs fast.
        import molix.data.cache as cache_mod

        orig_wait = cache_mod.PackedCache.wait_until_ready

        def fast_wait(self, *, timeout: float = 0.2, poll_interval: float = 0.05):
            return orig_wait(self, timeout=timeout, poll_interval=poll_interval)

        monkeypatch.setattr(cache_mod.PackedCache, "wait_until_ready", fast_wait)
        with pytest.raises(TimeoutError, match="Timed out"):
            spec.cache(src, base_dir=tmp_path)

    def test_sink_filename_embeds_cache_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        spec = Pipeline("qm9").add(_Tag()).build()
        src = _FakeSource("s", _samples(1))
        packed = spec.cache(src, base_dir=tmp_path)
        expected_key = spec.cache_key(src)
        assert packed.sink.name == f"qm9-{expected_key}.pt"

    def test_same_source_different_fit_source_different_sink(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        spec = Pipeline("p").add(MeanShift("y")).build()

        full = _FakeSource("full", _samples(10))
        fit_a = _FakeSource("fit-a", _samples(5))
        fit_b = _FakeSource("fit-b", _samples(3))

        packed_a = spec.cache(full, base_dir=tmp_path, fit_source=fit_a)
        packed_b = spec.cache(full, base_dir=tmp_path, fit_source=fit_b)
        assert packed_a.sink != packed_b.sink
        mean_a = packed_a.load()["task_states"]["mean_shift:y"]["mean"].item()
        mean_b = packed_b.load()["task_states"]["mean_shift:y"]["mean"].item()
        assert abs(mean_a - 2.0) < 1e-9   # y=0..4 → mean=2
        assert abs(mean_b - 1.0) < 1e-9   # y=0..2 → mean=1
