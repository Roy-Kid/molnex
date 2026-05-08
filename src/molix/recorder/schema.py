"""Record schema and writer Protocol for the metric recorder.

This module implements the wire schema described in molrec spec §Metric
records and §Metric types. The on-disk encoding (Zarr v3 in
:mod:`molix.recorder.zarr_store`, JSONL or anything else in future
backends) is decoupled from the in-memory record shape defined here.

References:
    molrec spec §Structure / §Metric records / §Metric types / §Key namespace
    /Users/roykid/work/molcrafts/molrec/docs/spec/metrics.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

RESERVED_METRIC_TYPES: Final[tuple[str, ...]] = (
    "scalar",
    "histogram",
    "text",
    "image_ref",
    "json",
)
"""The five reserved metric types from molrec spec §Metric types.

Each entry's ``value`` contract:

* ``scalar`` — a finite Python ``int`` / ``float`` (NaN written by the
  caller signals model divergence and is preserved verbatim).
* ``histogram`` — ``{"bins": Sequence[float], "counts": Sequence[float]}``
  with ``len(bins) == len(counts) + 1`` (numpy histogram convention).
* ``text`` — a plain ``str``.
* ``image_ref`` — ``{"path": str, "caption": str | None}``.
* ``json`` — any JSON-serialisable Python value.
"""


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """One immutable metric record matching molrec §Metric records.

    The field order is fixed by the spec: ``(type, key, step,
    wall_time_ns, value, tags)``. Frozen because molrec mandates that
    historical records are not mutated; the writer's append-only
    discipline starts here in memory.

    Args:
        type: One of :data:`RESERVED_METRIC_TYPES`.
        key: Stable slash-separated namespaced name, e.g. ``"train/loss"``.
            Must be non-empty.
        step: Optional global step (``None`` permitted by the spec).
            When present, must be a finite integer.
        wall_time_ns: Wall-clock timestamp in **epoch nanoseconds**
            (``int64``). This is a deliberate physical-encoding deviation
            from the spec's ISO-8601 string for numerical sort/diff
            efficiency; readers expose both views.
        value: Payload whose Python type must match ``type``'s contract;
            see :data:`RESERVED_METRIC_TYPES`.
        tags: Optional JSON-compatible mapping of free-form annotations.
    """

    type: str
    key: str
    step: int | None
    wall_time_ns: int
    value: Any
    tags: dict[str, Any] | None


@runtime_checkable
class MetricStore(Protocol):
    """Append-only writer Protocol for metric records.

    The Protocol exposes exactly three methods — ``append``, ``flush``,
    ``close`` — and **no** mutator (``update`` / ``delete`` / ``set``
    / ``overwrite`` / ``truncate``). This enforces molrec spec §Metric
    records' "writers should not mutate historical records" rule at the
    type level: a backend that wanted to expose a destructive operation
    would fail :func:`isinstance` against this Protocol.
    """

    def append(self, record: MetricRecord) -> None:
        """Append one record to the underlying store.

        Implementations may buffer in memory and flush opportunistically;
        the contract is that after :meth:`close` returns, every record
        ever appended is persisted.
        """
        ...

    def flush(self) -> None:
        """Force any buffered records out to the backing store.

        Useful before reading from the same store in another process,
        or before snapshotting the on-disk state for a checkpoint.
        """
        ...

    def close(self) -> None:
        """Finalise the store: flush, write the optional index, release handles.

        After ``close`` no further :meth:`append` calls are permitted;
        attempting one is implementation-defined behaviour.
        """
        ...
