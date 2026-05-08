"""Zarr v3 backend implementing the molrec metric record schema.

This module is the **write side** of the metric recorder. The on-disk
layout follows molrec spec §Structure
(``/Users/roykid/work/molcrafts/molrec/docs/spec/metrics.md``):

* ``metrics/records/<run_id>/`` is a Zarr v3 group with parallel 1-D
  arrays for ``type``, ``key``, ``step``, ``wall_time_ns``, ``tags``,
  and one ``value_<kind>`` array per molrec metric type.
* ``step`` carries ``-1`` as the missing-value sentinel (the spec
  permits ``step`` to be optional; we don't admit float-NaN on an
  ``int64`` array).
* ``wall_time_ns`` is ``int64`` epoch nanoseconds; the group
  attribute ``wall_time_unit = "epoch_ns"`` declares this deliberate
  deviation from the spec's ISO-8601 string encoding.
* Sharding (``chunks=(1024,)``, ``shards=(1_048_576,)``) collapses up
  to a million rows into a single physical file per parallel array,
  defending against the inode-explosion failure mode that motivated
  :class:`~molix.data.cache.PackedCache` over
  ``TensorDict.memmap_()``.

Records are append-only; the :class:`ZarrMetricStore` API exposes
``append`` / ``flush`` / ``close`` exclusively, never ``update`` or
``delete`` (molrec spec §Metric records: "writers should not mutate
historical records").

References:
    molrec spec §Structure / §Metric records / §Metric types
    /Users/roykid/work/molcrafts/molrec/docs/spec/metrics.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final

from molix.recorder.schema import MetricRecord

_SPEC_VERSION: Final[str] = "molrec-metrics/0.1"

_VALUE_ARRAY_NAMES: Final[tuple[str, ...]] = (
    "value_scalar",
    "value_text",
    "value_histogram",
    "value_image_ref",
    "value_json",
)


class ZarrMetricStore:
    """Append-only Zarr v3 backend for molrec metric records.

    The store materialises one Zarr group per ``run_id`` under
    ``<path>/metrics/records/<run_id>/``. Multiple runs may share a
    parent ``<path>``; each run is independent.

    Args:
        path: Root directory of the Zarr store. Created if absent.
        run_id: Per-run identifier; used as the leaf group name and
            written into the group ``attrs`` for self-description.
        chunk_rows: Rows per Zarr chunk (default 1024). Smaller =
            finer-grained random reads; larger = lower metadata
            overhead.
        shard_rows: Rows per shard file (default 1,048,576). The
            shard rolls multiple chunks into one physical file —
            this is what keeps the file count O(num_runs * num_arrays)
            rather than O(num_records / chunk_rows).
        spec_version: Recorded into ``attrs["spec_version"]`` so a
            future reader can detect format drift.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        run_id: str,
        *,
        chunk_rows: int = 1024,
        shard_rows: int = 1_048_576,
        spec_version: str = _SPEC_VERSION,
    ) -> None:
        if chunk_rows <= 0:
            raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
        if shard_rows <= 0 or shard_rows % chunk_rows != 0:
            raise ValueError(
                f"shard_rows ({shard_rows}) must be a positive multiple of "
                f"chunk_rows ({chunk_rows})"
            )

        import zarr

        self._zarr = zarr
        self._path = Path(path)
        self._run_id = run_id
        self._chunk_rows = chunk_rows
        self._shard_rows = shard_rows
        self._closed = False

        self._root = zarr.open(str(self._path), mode="a")
        self._grp = self._root.require_group(f"metrics/records/{run_id}")

        self._grp.attrs["wall_time_unit"] = "epoch_ns"
        self._grp.attrs["spec_version"] = spec_version
        self._grp.attrs["run_id"] = run_id

        self._arrays = self._init_arrays()
        self._n_rows = int(self._arrays["type"].shape[0])
        self._buffer: dict[str, list[Any]] = {name: [] for name in self._arrays}

    def _init_arrays(self) -> dict[str, Any]:
        """Open or create the parallel 1-D arrays under the run group."""
        common: dict[str, Any] = {
            "shape": (0,),
            "chunks": (self._chunk_rows,),
            "shards": (self._shard_rows,),
        }
        specs: list[tuple[str, str, Any]] = [
            ("type", "str", ""),
            ("key", "str", ""),
            ("step", "int64", -1),
            ("wall_time_ns", "int64", 0),
            ("tags", "str", ""),
            ("value_scalar", "float64", float("nan")),
            ("value_text", "str", ""),
            ("value_histogram", "str", ""),
            ("value_image_ref", "str", ""),
            ("value_json", "str", ""),
        ]
        out: dict[str, Any] = {}
        existing = set(self._grp.array_keys())
        for name, dtype, fill_value in specs:
            if name in existing:
                out[name] = self._grp[name]
            else:
                out[name] = self._grp.create_array(
                    name,
                    dtype=dtype,
                    fill_value=fill_value,
                    **common,
                )
        return out

    def append(self, record: MetricRecord) -> None:
        """Buffer one record; values written on the next :meth:`flush` / :meth:`close`.

        Buffering is per-store (not per-array) so the 10 parallel arrays
        always grow in lock-step and never desynchronise on a partial
        write. A SIGKILL between :meth:`append` and :meth:`flush` loses
        the buffered tail but never corrupts the on-disk record stream.
        """
        if self._closed:
            raise RuntimeError("ZarrMetricStore.append called after close()")
        if not record.key:
            raise ValueError("MetricRecord.key must be a non-empty string")

        self._buffer["type"].append(record.type)
        self._buffer["key"].append(record.key)
        self._buffer["step"].append(-1 if record.step is None else int(record.step))
        self._buffer["wall_time_ns"].append(int(record.wall_time_ns))
        self._buffer["tags"].append(
            "" if record.tags is None else json.dumps(record.tags, sort_keys=True)
        )

        for name in _VALUE_ARRAY_NAMES:
            self._buffer[name].append(self._arrays[name].fill_value)

        match record.type:
            case "scalar":
                self._buffer["value_scalar"][-1] = float(record.value)
            case "histogram":
                self._buffer["value_histogram"][-1] = json.dumps(
                    {
                        "bins": list(record.value["bins"]),
                        "counts": list(record.value["counts"]),
                    }
                )
            case "text":
                self._buffer["value_text"][-1] = str(record.value)
            case "image_ref":
                self._buffer["value_image_ref"][-1] = json.dumps(record.value)
            case "json":
                self._buffer["value_json"][-1] = json.dumps(record.value)
            case other:
                raise ValueError(
                    f"Unknown metric type {other!r}; expected one of "
                    f"{('scalar', 'histogram', 'text', 'image_ref', 'json')!r}"
                )

    def flush(self) -> None:
        """Drain the in-memory buffer into the Zarr arrays."""
        if self._closed:
            raise RuntimeError("ZarrMetricStore.flush called after close()")
        n = len(self._buffer["type"])
        if n == 0:
            return

        new_total = self._n_rows + n
        for name, arr in self._arrays.items():
            arr.resize((new_total,))
            arr[self._n_rows : new_total] = self._buffer[name]
            self._buffer[name] = []

        self._n_rows = new_total

    def close(self) -> None:
        """Flush, write the derived index, and mark the store closed.

        The index is advisory per molrec spec §Index — readers should
        prefer reconstituting from records when both are available, so
        the writer treats it as a convenience side-output.
        """
        if self._closed:
            return
        self.flush()

        index_grp = self._grp.require_group("index")
        index_grp.attrs["line_count"] = self._n_rows
        keys_arr = self._arrays["key"]
        types_arr = self._arrays["type"]
        if self._n_rows > 0:
            keys = list(keys_arr[:])
            types = list(types_arr[:])
            seen = {}
            for k, t in zip(keys, types, strict=True):
                seen.setdefault(k, t)
            index_grp.attrs["series_count"] = len(seen)
            index_grp.attrs["series_keys"] = list(seen.keys())
            index_grp.attrs["series_types"] = list(seen.values())
        else:
            index_grp.attrs["series_count"] = 0
            index_grp.attrs["series_keys"] = []
            index_grp.attrs["series_types"] = []

        self._closed = True

    def __enter__(self) -> "ZarrMetricStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
