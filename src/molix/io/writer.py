"""Sharded Zarr v3 writer for the MolNex training-event journal.

The on-disk layout matches the original ``recorder/zarr_store.py``
verbatim — only the Python API has been simplified:

* The writer accepts kwargs (``type=``, ``key=``, ``step=``,
  ``wall_time_ns=``, ``value=``, ``tags=``) instead of a
  ``MetricRecord`` dataclass — there is one producer
  (:class:`molix.hooks.journal.JournalHook`) and we YAGNI'd the
  schema layer.
* Five reserved ``type`` strings (preserved verbatim from the
  legacy ``RESERVED_METRIC_TYPES`` tuple) dispatch to per-type
  parallel arrays inside :meth:`JournalWriter.append` via a
  ``match`` block; unknown types raise :class:`ValueError`.

The chunk + shard sizes (``chunks=(1024,)``, ``shards=(1_048_576,)``)
are deliberate: they collapse up to a million rows into a single
physical file per parallel array, matching the inode-discipline
posture of :class:`molix.data.cache.PackedCache` and defending
against the per-record-subdirectory failure mode of
``TensorDict.memmap_``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final

_RESERVED_TYPES: Final[tuple[str, ...]] = (
    "scalar",
    "histogram",
    "text",
    "image_ref",
    "json",
)

_SPEC_VERSION: Final[str] = "molrec-metrics/0.1"

_VALUE_ARRAY_NAMES: Final[tuple[str, ...]] = (
    "value_scalar",
    "value_text",
    "value_histogram",
    "value_image_ref",
    "value_json",
)


class JournalWriter:
    """Append-only Zarr v3 writer for training-event records.

    The writer materialises one Zarr group per ``run_id`` under
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
            this is what keeps the file count
            ``O(num_runs * num_arrays)`` rather than
            ``O(num_records / chunk_rows)``.
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

    def append(
        self,
        *,
        type: str,
        key: str,
        step: int | None,
        wall_time_ns: int,
        value: Any,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """Buffer one record; values written on the next :meth:`flush` / :meth:`close`.

        Args:
            type: One of ``"scalar"``, ``"histogram"``, ``"text"``,
                ``"image_ref"``, ``"json"``.
            key: Stable slash-separated namespaced name, e.g.
                ``"train/loss"``. Must be non-empty.
            step: Optional global step. ``None`` is encoded as ``-1``
                in the on-disk array (the ``step`` array is ``int64``
                so float-NaN is unavailable).
            wall_time_ns: Wall-clock timestamp in epoch nanoseconds.
            value: Payload whose Python type must match ``type``'s
                contract — ``float`` for scalar, ``{"bins":..., "counts":...}``
                for histogram, ``str`` for text, ``{"path":..., "caption":...}``
                for image_ref, JSON-serialisable value for json.
            tags: Optional JSON-compatible mapping of free-form
                annotations.

        Raises:
            ValueError: If ``key`` is empty or ``type`` is not one of
                the five reserved strings.
            RuntimeError: If called after :meth:`close`.
        """
        if self._closed:
            raise RuntimeError("JournalWriter.append called after close()")
        if not key:
            raise ValueError("key must be a non-empty string")

        self._buffer["type"].append(type)
        self._buffer["key"].append(key)
        self._buffer["step"].append(-1 if step is None else int(step))
        self._buffer["wall_time_ns"].append(int(wall_time_ns))
        self._buffer["tags"].append("" if tags is None else json.dumps(tags, sort_keys=True))

        for name in _VALUE_ARRAY_NAMES:
            self._buffer[name].append(self._arrays[name].fill_value)

        match type:
            case "scalar":
                self._buffer["value_scalar"][-1] = float(value)
            case "histogram":
                self._buffer["value_histogram"][-1] = json.dumps(
                    {
                        "bins": list(value["bins"]),
                        "counts": list(value["counts"]),
                    }
                )
            case "text":
                self._buffer["value_text"][-1] = str(value)
            case "image_ref":
                self._buffer["value_image_ref"][-1] = json.dumps(value)
            case "json":
                self._buffer["value_json"][-1] = json.dumps(value)
            case other:
                raise ValueError(
                    f"Unknown record type {other!r}; expected one of {_RESERVED_TYPES!r}"
                )

    def flush(self) -> None:
        """Drain the in-memory buffer into the Zarr arrays."""
        if self._closed:
            raise RuntimeError("JournalWriter.flush called after close()")
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
        """Flush, write the derived index, and mark the writer closed."""
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
            seen: dict[str, str] = {}
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

    def __enter__(self) -> "JournalWriter":
        """Enter the writer context, returning ``self``."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the journal on context exit (also flushes on exceptions)."""
        self.close()
