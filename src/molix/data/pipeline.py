"""Declarative pipeline container.

A :class:`PipelineSpec` is a pure description of *what tasks to run and in
what order*, plus OOP methods that orchestrate execution, caching, and
DDP-aware materialisation. There are no free functions for cache or rank
coordination — everything is encapsulated on :class:`PipelineSpec` and
:class:`~molix.data.cache.PackedCache`.

Typical usage::

    from molix.data import Pipeline, AtomicDress, NeighborList

    pipe = (
        Pipeline("qm9-u0")
        .add(AtomicDress(elements=[1, 6, 7, 8, 9], target_key="U0"))
        .add(NeighborList(cutoff=5.0))
        .build()
    )

    # In-memory execution (rarely needed directly — see `.cache` below):
    samples = list(pipe.run(source, fit_source=train_subset))

    # DDP-aware cache materialisation (normal path):
    packed = pipe.cache(
        source,
        base_dir=run_dir / "cache",
        fit_source=train_subset,
        extra={"n_train": str(n_train), "seed": str(seed)},
    )
    ds = MmapDataset(packed.sink)
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from molix.data.cache import PackedCache
from molix.data.task import BatchTask, DatasetTask, Runnable, SampleTask

__all__ = ["TaskEntry", "PipelineSpec", "Pipeline"]


@dataclass(frozen=True)
class TaskEntry:
    """A single registered task inside a pipeline."""

    name: str
    task: Any

    def apply(self, data: Any) -> Any:
        """Dispatch *data* through the wrapped task.

        Accepts either a :class:`~molix.data.task.Runnable` (uses
        ``execute``) or a plain callable.
        """
        task = self.task
        if isinstance(task, Runnable):
            return task.execute(data)
        if callable(task):
            return task(data)
        raise TypeError(f"Task {self.name!r} is neither Runnable nor callable: {type(task)}")


class PipelineSpec:
    """Compiled, immutable pipeline description.

    Holds both *what* the pipeline does (task list, grouping, identity)
    and the methods that execute or materialise it. There is no separate
    ``execute`` / ``cache`` / ``ddp`` module — orchestration lives here.
    """

    __slots__ = ("name", "pipeline_id", "tasks")

    def __init__(
        self,
        name: str,
        pipeline_id: str,
        tasks: tuple[TaskEntry, ...],
    ) -> None:
        self.name = name
        self.pipeline_id = pipeline_id
        self.tasks = tasks

    # -- grouping ----------------------------------------------------------

    @property
    def prepare_tasks(self) -> tuple[TaskEntry, ...]:
        """Tasks executed *before* the DataLoader (sample- and dataset-level)."""
        return tuple(
            e
            for e in self.tasks
            if isinstance(e.task, (SampleTask, DatasetTask))
            or (callable(e.task) and not isinstance(e.task, BatchTask))
        )

    @property
    def batch_tasks(self) -> tuple[TaskEntry, ...]:
        """Tasks executed *post-collate* inside the DataLoader."""
        return tuple(e for e in self.tasks if isinstance(e.task, BatchTask))

    # -- introspection -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pipeline_id": self.pipeline_id,
            "tasks": [
                {
                    "name": e.name,
                    "type": type(e.task).__name__,
                    "task_id": getattr(e.task, "task_id", type(e.task).__name__),
                }
                for e in self.tasks
            ],
        }

    def __repr__(self) -> str:
        names = ", ".join(e.name for e in self.tasks)
        return f"PipelineSpec(name={self.name!r}, tasks=[{names}], id={self.pipeline_id})"

    # -- execution ---------------------------------------------------------

    def transform(self, sample: dict) -> dict:
        """Apply every prepare task to *sample* in order.

        Per-sample inference transform. Any
        :class:`~molix.data.task.DatasetTask` in the pipeline must have
        been fitted already (e.g. via a prior :meth:`run`); the task will
        raise if called unfit.
        """
        for entry in self.prepare_tasks:
            sample = entry.apply(sample)
        return sample

    def run(
        self,
        source: Any,
        *,
        fit_source: Any | None = None,
    ) -> Iterator[dict]:
        """Iterate *source*, fit every :class:`DatasetTask`, yield processed samples.

        Pure in-memory; no disk IO. For persistence, use :meth:`cache` or
        :meth:`build_cache`.

        Args:
            source: Raw :class:`~molix.data.source.DataSource` — every
                sample is processed.
            fit_source: Source used for :meth:`DatasetTask.fit`. Defaults
                to *source*. Pass a distinct (e.g. train-only) source so
                that fit-dependent tasks never peek at validation / test
                data.

        Yields:
            Processed sample dicts in *source* order.
        """
        prepare = self.prepare_tasks
        has_dataset_task = any(isinstance(e.task, DatasetTask) for e in prepare)

        # Case A: explicit fit_source (subset). Fit on the subset, then apply
        # every prepare task to the full source. Two passes, but the fit pass
        # is small.
        if has_dataset_task and fit_source is not None:
            fit_data = [fit_source[i] for i in range(len(fit_source))]
            for entry in prepare:
                if isinstance(entry.task, DatasetTask):
                    entry.task.fit(fit_data)
                fit_data = [entry.apply(s) for s in fit_data]

            for i in range(len(source)):
                s = source[i]
                for entry in prepare:
                    s = entry.apply(s)
                yield s
            return

        # Case B: fit on the full source. Interleave fit() with execute() so
        # each task is applied to every sample exactly once.
        buffered = [source[i] for i in range(len(source))]
        for entry in prepare:
            if isinstance(entry.task, DatasetTask):
                entry.task.fit(buffered)
            buffered = [entry.apply(s) for s in buffered]
        yield from buffered

    def collect_task_states(self) -> dict[str, dict[str, Any]]:
        """Return ``{entry.name: task.state_dict()}`` for every :class:`DatasetTask`.

        Call right after :meth:`run` / :meth:`build_cache` so the fitted
        state is captured before the task instance goes out of scope.
        """
        states: dict[str, dict[str, Any]] = {}
        for entry in self.tasks:
            if isinstance(entry.task, DatasetTask):
                states[entry.name] = entry.task.state_dict()
        return states

    def load_task_states(self, states: dict[str, dict[str, Any]]) -> None:
        """Restore fitted state into each :class:`DatasetTask` by entry name."""
        names = [e.name for e in self.tasks]
        if len(set(names)) != len(names):
            raise RuntimeError(
                f"Duplicate task names in pipeline {self.name!r}: {names!r}. "
                f"This should be caught at Pipeline.add; state routing is "
                f"ambiguous."
            )
        by_name = {e.name: e.task for e in self.tasks}
        for name, state in states.items():
            task = by_name.get(name)
            if task is None or not isinstance(task, DatasetTask):
                continue
            task.load_state_dict(state)

    # -- cache identity & materialisation ----------------------------------

    def cache_key(
        self,
        source: Any,
        *,
        fit_source: Any | None = None,
        extra: dict[str, str] | None = None,
    ) -> str:
        """Stable 12-hex identity for this (pipeline, source, fit_source, extra) tuple.

        Forwards to :meth:`PackedCache.make_key` with the pipeline's
        ``pipeline_id`` and the source's ``source_id``. Changing *source*,
        *fit_source*, or anything in *extra* invalidates the cache.
        """
        fit_source_id = fit_source.source_id if fit_source is not None else None
        return PackedCache.make_key(
            pipeline_id=self.pipeline_id,
            source_id=source.source_id,
            fit_source_id=fit_source_id,
            extra=extra,
        )

    def build_cache(
        self,
        source: Any,
        sink: str | Path | PackedCache,
        *,
        fit_source: Any | None = None,
        overwrite: bool = False,
    ) -> PackedCache:
        """Run the pipeline on *source* and write the result to *sink*.

        Unconditional — does not consult rank or readiness. Prefer
        :meth:`cache` for DDP-aware workflows.

        Args:
            source: Full source to materialise.
            sink: Target cache file (path-like or existing
                :class:`PackedCache`).
            fit_source: See :meth:`run`.
            overwrite: If the sink already exists, replace it. Otherwise
                keep the existing file (no-op).

        Returns:
            The :class:`PackedCache` bound to *sink*.
        """
        packed = sink if isinstance(sink, PackedCache) else PackedCache(sink)
        if packed.sink.exists() and not overwrite:
            return packed
        samples = list(self.run(source, fit_source=fit_source))
        packed.save(samples, task_states=self.collect_task_states(), overwrite=overwrite)
        return packed

    def cache(
        self,
        source: Any,
        *,
        base_dir: str | Path,
        fit_source: Any | None = None,
        extra: dict[str, str] | None = None,
        overwrite: bool = False,
    ) -> PackedCache:
        """DDP-aware cache materialisation.

        On rank 0 (``$RANK == "0"`` or unset), build the cache if missing
        (or ``overwrite`` requested). On other ranks, poll until the file
        appears via :meth:`PackedCache.wait_until_ready`.

        The sink file is placed at
        ``base_dir / f"{self.name}-{self.cache_key(...)}.pt"`` — callers
        do not choose filenames so that cache identity and location stay
        aligned.

        Args:
            source: Full source to materialise.
            base_dir: Directory holding this run's caches. Created if
                missing (rank 0 only).
            fit_source: See :meth:`run`.
            extra: Extra identity strings folded into
                :meth:`cache_key` (split sizes, seed, dtype, ...).
            overwrite: Force rebuild on rank 0 even if a cache exists.

        Returns:
            The :class:`PackedCache` instance for the materialised file.
        """
        base_dir = Path(base_dir)
        key = self.cache_key(source, fit_source=fit_source, extra=extra)
        sink = base_dir / f"{self.name}-{key}.pt"
        packed = PackedCache(sink)

        if _is_primary_rank():
            sink.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not packed.is_ready():
                self.build_cache(source, packed, fit_source=fit_source, overwrite=overwrite)
        else:
            packed.wait_until_ready()
        return packed


# ---------------------------------------------------------------------------
# Builder DSL
# ---------------------------------------------------------------------------


class Pipeline:
    """Fluent builder for a :class:`PipelineSpec`.

    Three equivalent task-registration styles::

        Pipeline("p").add(NeighborList(cutoff=5.0))        # Task instance
        Pipeline("p").add(my_callable, name="normalize")   # bare callable
        Pipeline("p").task(lambda s: s, name="noop")       # @decorator
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: list[TaskEntry] = []
        self._names: set[str] = set()

    def _register(self, entry_name: str, task: Any) -> None:
        if entry_name in self._names:
            raise ValueError(
                f"Task name {entry_name!r} already registered in pipeline "
                f"{self.name!r}. Task names must be unique so fitted state "
                f"can be unambiguously routed on cache load."
            )
        self._names.add(entry_name)
        self._entries.append(TaskEntry(entry_name, task))

    def task(self, fn: Any = None, *, name: str | None = None) -> Any:
        """Register a bare function as a sample-level task."""

        def decorator(f: Any) -> Any:
            entry_name = name or getattr(f, "__name__", "task")
            self._register(entry_name, f)
            return f

        if fn is not None:
            return decorator(fn)
        return decorator

    def add(self, task: Any, *, name: str | None = None) -> "Pipeline":
        """Add a Task instance or plain callable. Returns self for chaining."""
        _validate_task(task)
        entry_name = name or getattr(task, "task_id", type(task).__name__)
        self._register(entry_name, task)
        return self

    def build(self) -> PipelineSpec:
        tasks = tuple(self._entries)
        return PipelineSpec(self.name, _stable_pipeline_id(self.name, tasks), tasks)


# ---------------------------------------------------------------------------
# Validation & identity
# ---------------------------------------------------------------------------


def _validate_task(task: Any) -> None:
    """Enforce: every registered task is a Task subclass or a plain callable."""
    if isinstance(task, (SampleTask, DatasetTask, BatchTask)):
        return
    if callable(task):
        return
    raise TypeError(
        f"Task must be a SampleTask/DatasetTask/BatchTask or callable, got {type(task).__name__}"
    )


def _stable_pipeline_id(name: str, tasks: tuple[TaskEntry, ...]) -> str:
    """Deterministic 16-hex id derived from name + task composition."""
    parts = [name]
    for e in tasks:
        tid = getattr(e.task, "task_id", type(e.task).__qualname__)
        parts.append(f"{e.name}:{tid}:{type(e.task).__name__}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _is_primary_rank() -> bool:
    """Return ``True`` when ``$RANK`` is ``"0"``, unset, or malformed.

    Drives rank-0-builds-others-wait in :meth:`PipelineSpec.cache`. This
    mirrors the launcher convention (``torchrun`` / ``torch.distributed``
    set ``RANK`` on every worker); we deliberately do not call
    ``torch.distributed.get_rank`` because the cache stage typically runs
    *before* the process group is initialised.
    """
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True
