"""Append-only metric persistence for MolNex training runs.

This package implements the metric stream side of the molrec spec
(see ``/Users/roykid/work/molcrafts/molrec/docs/spec/metrics.md``).
The **Structure**, **Metric records**, **Metric types**, and **Key
namespace** sections are adopted verbatim:

* §Structure — long-table layout under ``metrics/records/<run_id>/``
  with parallel 1-D arrays for ``type``, ``key``, ``step``,
  ``wall_time``, ``value``, and optional ``tags``.
* §Metric records — every record carries ``type / key / wall_time /
  value`` (required) and optionally ``step / tags``; writers are
  append-oriented and never mutate historical records.
* §Metric types — five reserved types: ``scalar``, ``histogram``,
  ``text``, ``image_ref``, ``json``.
* §Key namespace — slash-separated stable names aligned with MolNex's
  :class:`~molix.core.state.TrainState` namespaces (``train/*``,
  ``eval/*``, ``performance/*``, ``gpu/*``).

Public surface:

* :class:`MetricRecord` — frozen dataclass for one record.
* :class:`MetricStore` — Protocol for append-only backends.
* :class:`ZarrMetricStore` — Zarr v3 backend.
* :class:`MetricsReader` — read-side counterpart.

The hook that drives a :class:`MetricStore` from a running
:class:`~molix.core.trainer.Trainer` is
:class:`molix.core.hooks.Journal`.
"""

from __future__ import annotations

from molix.recorder.reader import MetricsReader
from molix.recorder.schema import RESERVED_METRIC_TYPES, MetricRecord, MetricStore
from molix.recorder.zarr_store import ZarrMetricStore

__all__ = [
    "RESERVED_METRIC_TYPES",
    "MetricRecord",
    "MetricStore",
    "MetricsReader",
    "ZarrMetricStore",
]
