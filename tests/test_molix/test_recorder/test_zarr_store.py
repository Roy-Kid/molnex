"""RED tests for ``molix.recorder.zarr_store.ZarrMetricStore``.

Acceptance traces:
    ac-003 → :func:`test_five_types_roundtrip`
    ac-004 → :func:`test_append_does_not_mutate_history`
    ac-005 → :func:`test_layout_matches_molrec_spec`
    ac-006 → :func:`test_sharding_keeps_file_count_small`
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from molix.recorder.schema import MetricRecord


def _scalar_record(step: int, key: str = "train/loss", value: float | None = None) -> MetricRecord:
    return MetricRecord(
        type="scalar",
        key=key,
        step=step,
        wall_time_ns=1_700_000_000_000_000_000 + step,
        value=float(step) if value is None else value,
        tags=None,
    )


def test_layout_matches_molrec_spec(tmp_path: Path) -> None:
    """ac-005: Zarr group layout matches molrec spec §Structure.

    After appending records of multiple types and closing, the on-disk
    Zarr v3 group at ``<store>/metrics/records/<run_id>/`` must expose
    parallel 1-D arrays named per the spec, plus our deliberate
    ``epoch_ns`` deviation declared in group attrs.
    """
    from molix.recorder.zarr_store import ZarrMetricStore

    store = ZarrMetricStore(tmp_path, run_id="run0")
    store.append(_scalar_record(step=0))
    store.append(
        MetricRecord(
            type="json",
            key="hparams",
            step=0,
            wall_time_ns=1,
            value={"lr": 1e-3},
            tags=None,
        )
    )
    store.close()

    import zarr

    root = zarr.open(str(tmp_path), mode="r")
    grp = root["metrics/records/run0"]

    expected_arrays = {"type", "key", "step", "wall_time_ns", "value_scalar", "value_json"}
    actual_arrays = set(grp.array_keys())
    assert expected_arrays.issubset(actual_arrays), (
        f"Zarr group must expose at least {expected_arrays!r}; got {actual_arrays!r}"
    )

    n = grp["type"].shape[0]
    for name in ("key", "step", "wall_time_ns"):
        assert grp[name].shape[0] == n, (
            f"Parallel array {name!r} length {grp[name].shape[0]} != type length {n}"
        )

    assert grp.attrs["wall_time_unit"] == "epoch_ns"
    assert isinstance(grp.attrs.get("spec_version"), str) and grp.attrs["spec_version"], (
        "spec_version attr must be a non-empty string"
    )


def test_five_types_roundtrip(tmp_path: Path) -> None:
    """ac-003: All five molrec metric types round-trip through the store."""
    from molix.recorder.reader import MetricsReader
    from molix.recorder.zarr_store import ZarrMetricStore

    records = [
        MetricRecord("scalar", "train/loss", 0, 1, 0.42, None),
        MetricRecord(
            "histogram",
            "weights/embed",
            0,
            2,
            {"bins": [0.0, 0.5, 1.0], "counts": [3.0, 5.0]},
            None,
        ),
        MetricRecord("text", "note/run", 0, 3, "first epoch ok", None),
        MetricRecord(
            "image_ref",
            "viz/sample",
            0,
            4,
            {"path": "viz/0.png", "caption": "first"},
            None,
        ),
        MetricRecord("json", "hparams", 0, 5, {"lr": 1e-3, "model": "mace"}, None),
    ]

    store = ZarrMetricStore(tmp_path, run_id="run0")
    for r in records:
        store.append(r)
    store.close()

    read = list(MetricsReader(tmp_path, run_id="run0").records())
    assert len(read) == len(records)

    for orig, got in zip(records, read, strict=True):
        assert got.type == orig.type
        assert got.key == orig.key
        assert got.step == orig.step
        assert got.wall_time_ns == orig.wall_time_ns

        if orig.type == "scalar":
            assert got.value == pytest.approx(orig.value)
        elif orig.type == "histogram":
            np.testing.assert_array_equal(
                np.asarray(got.value["bins"]), np.asarray(orig.value["bins"])
            )
            np.testing.assert_array_equal(
                np.asarray(got.value["counts"]), np.asarray(orig.value["counts"])
            )
        else:
            assert got.value == orig.value


def test_append_does_not_mutate_history(tmp_path: Path) -> None:
    """ac-004: Calling append after a previous batch leaves history bit-identical."""
    from molix.recorder.zarr_store import ZarrMetricStore

    store = ZarrMetricStore(tmp_path, run_id="run0")
    for i in range(1000):
        store.append(_scalar_record(step=i, value=float(i)))
    store.flush()

    import zarr

    grp = zarr.open(str(tmp_path), mode="r")["metrics/records/run0"]
    snap_step = np.asarray(grp["step"][:1000]).copy()
    snap_value = np.asarray(grp["value_scalar"][:1000]).copy()

    for i in range(1000, 2000):
        store.append(_scalar_record(step=i, value=float(i)))
    store.close()

    grp2 = zarr.open(str(tmp_path), mode="r")["metrics/records/run0"]
    np.testing.assert_array_equal(snap_step, np.asarray(grp2["step"][:1000]))
    np.testing.assert_array_equal(snap_value, np.asarray(grp2["value_scalar"][:1000]))
    assert grp2["step"].shape[0] == 2000


def test_sharding_keeps_file_count_small(tmp_path: Path) -> None:
    """ac-006: 100,000 records × 5 keys produces O(50) physical files.

    Sharding sanity: the run-group directory must hold ≤ 50 regular
    data files (excluding ``zarr.json`` metadata files), with ≤ 5
    shard files per parallel array. This is the direct defence against
    the ``TensorDict.memmap_`` per-record-subdirectory failure mode
    that motivated ``PackedCache``.
    """
    from molix.recorder.zarr_store import ZarrMetricStore

    keys = [f"train/m{i}" for i in range(5)]
    store = ZarrMetricStore(tmp_path, run_id="run0")
    for step in range(20_000):
        for k in keys:
            store.append(_scalar_record(step=step, key=k))
    store.close()

    run_dir = tmp_path / "metrics" / "records" / "run0"
    files = []
    for root, _, fnames in os.walk(run_dir):
        for f in fnames:
            if f.endswith(".json"):
                continue
            files.append(os.path.join(root, f))

    assert len(files) <= 50, (
        f"Sharding regression: 100k records produced {len(files)} files "
        f"(expected ≤ 50). PackedCache-style inode discipline broken."
    )


def test_protocol_compliance(tmp_path: Path) -> None:
    """ZarrMetricStore satisfies the MetricStore Protocol."""
    from molix.recorder.schema import MetricStore
    from molix.recorder.zarr_store import ZarrMetricStore

    store = ZarrMetricStore(tmp_path, run_id="run0")
    try:
        assert isinstance(store, MetricStore)
    finally:
        store.close()
