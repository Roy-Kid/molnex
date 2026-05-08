"""Read-side counterpart to :mod:`molix.recorder.zarr_store`.

The reader treats the on-disk Zarr group as the authoritative record
stream, per molrec spec §Index ("readers must not treat an index as
authoritative if the underlying metric records are available"). The
optional ``metrics/index`` subgroup is consulted for fast key
enumeration when present, but the reader transparently rebuilds the
key index from the parallel ``key`` array if the subgroup is absent.

References:
    molrec spec §Structure / §Index
    /Users/roykid/work/molcrafts/molrec/docs/spec/metrics.md
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from molix.recorder.schema import MetricRecord


class MetricsReader:
    """Read-side facade over a Zarr v3 metric store.

    Args:
        path: Root directory of the Zarr store (matches the writer's
            ``path`` argument).
        run_id: Per-run identifier (matches the writer's ``run_id``).
    """

    def __init__(self, path: str | os.PathLike[str], run_id: str) -> None:
        import zarr

        self._zarr = zarr
        self._path = Path(path)
        self._run_id = run_id
        self._root = zarr.open(str(self._path), mode="r")
        self._grp = self._root[f"metrics/records/{run_id}"]

    def __len__(self) -> int:
        """Number of records in the store."""
        return int(self._grp["type"].shape[0])

    def keys(self) -> list[str]:
        """Distinct stable keys in append order of first appearance.

        Prefers the derived ``metrics/index`` subgroup when present,
        otherwise reconstitutes from the ``key`` array directly.
        """
        try:
            index_grp = self._grp["index"]
            cached = index_grp.attrs.get("series_keys")
            if cached is not None:
                return list(cached)
        except KeyError:
            pass
        return self._rebuild_keys()

    def _rebuild_keys(self) -> list[str]:
        keys = list(self._grp["key"][:])
        seen: dict[str, None] = {}
        for k in keys:
            if k not in seen:
                seen[k] = None
        return list(seen.keys())

    def scalars(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """Return parallel ``(steps, values)`` arrays for one scalar key.

        Args:
            key: Stable namespaced key (e.g. ``"train/loss"``).

        Returns:
            ``(steps, values)`` — both 1-D arrays of equal length, in
            append order. ``steps`` is ``int64`` (``-1`` for any record
            whose original ``step`` was ``None``); ``values`` is
            ``float64``.
        """
        type_arr = np.asarray(self._grp["type"][:])
        key_arr = np.asarray(self._grp["key"][:])
        mask = (type_arr == "scalar") & (key_arr == key)
        steps = np.asarray(self._grp["step"][:])[mask]
        values = np.asarray(self._grp["value_scalar"][:])[mask]
        return steps, values

    def records(self) -> Iterator[MetricRecord]:
        """Iterate every record in append order.

        The five molrec metric types are reconstituted by routing
        ``value`` through the per-type sparse arrays. Histograms,
        image_refs, and JSON values are JSON-decoded back to their
        original Python shape.
        """
        type_arr = self._grp["type"][:]
        key_arr = self._grp["key"][:]
        step_arr = self._grp["step"][:]
        wall_arr = self._grp["wall_time_ns"][:]
        tags_arr = self._grp["tags"][:]
        scalar_arr = self._grp["value_scalar"][:]
        text_arr = self._grp["value_text"][:]
        hist_arr = self._grp["value_histogram"][:]
        image_arr = self._grp["value_image_ref"][:]
        json_arr = self._grp["value_json"][:]

        n = len(type_arr)
        for i in range(n):
            t = str(type_arr[i])
            step_i = int(step_arr[i])
            tags_str = str(tags_arr[i])
            tags: dict[str, Any] | None = json.loads(tags_str) if tags_str else None

            value: Any
            match t:
                case "scalar":
                    value = float(scalar_arr[i])
                case "histogram":
                    value = json.loads(str(hist_arr[i]))
                case "text":
                    value = str(text_arr[i])
                case "image_ref":
                    value = json.loads(str(image_arr[i]))
                case "json":
                    value = json.loads(str(json_arr[i]))
                case other:
                    raise ValueError(f"Unknown metric type {other!r} at row {i}")

            yield MetricRecord(
                type=t,
                key=str(key_arr[i]),
                step=None if step_i == -1 else step_i,
                wall_time_ns=int(wall_arr[i]),
                value=value,
                tags=tags,
            )
