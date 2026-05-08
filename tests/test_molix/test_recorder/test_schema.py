"""RED tests for ``molix.recorder.schema`` — MetricRecord + MetricStore.

Acceptance traces:
    ac-001 → :func:`test_metric_record_fields`
    ac-002 → :func:`test_metric_store_protocol_is_append_only`
"""

from __future__ import annotations

import dataclasses
import re
from typing import get_type_hints

import pytest


def test_metric_record_fields() -> None:
    """ac-001: MetricRecord is a frozen dataclass with the molrec-required fields.

    The dataclass must expose exactly six fields in the documented order:
    (type, key, step, wall_time_ns, value, tags) — and must be frozen so
    historical records cannot be mutated in memory before being written.
    """
    from molix.recorder.schema import MetricRecord

    field_names = tuple(f.name for f in dataclasses.fields(MetricRecord))
    assert field_names == ("type", "key", "step", "wall_time_ns", "value", "tags"), (
        f"MetricRecord must declare fields exactly as "
        f"(type, key, step, wall_time_ns, value, tags); got {field_names!r}"
    )

    assert MetricRecord.__dataclass_params__.frozen is True, (
        "MetricRecord must be frozen so historical records are immutable in "
        "memory; molrec spec §Metric records — append-oriented invariant."
    )

    record = MetricRecord(
        type="scalar",
        key="train/loss",
        step=42,
        wall_time_ns=1_700_000_000_000_000_000,
        value=0.123,
        tags=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.value = 0.456  # type: ignore[misc]


def test_metric_store_protocol_is_append_only() -> None:
    """ac-002: MetricStore Protocol exposes only append/flush/close.

    The Protocol must NOT advertise any mutator/destructor method
    (update/delete/set/overwrite/truncate). This enforces the molrec
    spec §Metric records "append-oriented; writers should not mutate
    historical records" rule at the type level.
    """
    from molix.recorder.schema import MetricStore

    annotations = get_type_hints(MetricStore)
    method_names = {
        name for name in dir(MetricStore) if not name.startswith("_") and name not in annotations
    }

    assert method_names == {"append", "flush", "close"}, (
        f"MetricStore Protocol must expose exactly "
        f"{{'append', 'flush', 'close'}}; got {method_names!r}. "
        f"No update/delete/set/overwrite/truncate is permitted "
        f"(molrec spec §Metric records — append-oriented)."
    )

    forbidden = re.compile(r"^(update|delete|set|overwrite|truncate)$")
    forbidden_present = [n for n in method_names if forbidden.match(n)]
    assert not forbidden_present, (
        f"MetricStore Protocol must not expose any name matching "
        f"^(update|delete|set|overwrite|truncate)$; got {forbidden_present!r}"
    )


def test_metric_record_accepts_step_none() -> None:
    """molrec spec: ``step`` is optional. None must be accepted."""
    from molix.recorder.schema import MetricRecord

    record = MetricRecord(
        type="scalar",
        key="performance/step_per_second",
        step=None,
        wall_time_ns=1,
        value=10.0,
        tags=None,
    )
    assert record.step is None


def test_metric_record_reserved_types_are_documented() -> None:
    """The five reserved type strings must be exposed as a public Final.

    molrec spec §Metric types reserves {scalar, histogram, text,
    image_ref, json}. The schema module must surface this vocabulary
    so callers and tests can refer to it without hard-coding strings.
    """
    from molix.recorder import schema

    reserved = getattr(schema, "RESERVED_METRIC_TYPES", None)
    assert reserved is not None, (
        "schema.RESERVED_METRIC_TYPES must be defined so the reserved "
        "vocabulary lives in one place."
    )
    assert set(reserved) == {"scalar", "histogram", "text", "image_ref", "json"}, (
        f"RESERVED_METRIC_TYPES must equal molrec's five reserved types; got {set(reserved)!r}"
    )
