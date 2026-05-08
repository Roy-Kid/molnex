"""RED tests for ``molix.recorder.reader.MetricsReader``.

Acceptance traces:
    ac-007 → :func:`test_scalars_returns_parallel_arrays`
    ac-008 → :func:`test_reader_works_without_index`
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from molix.recorder.schema import MetricRecord


def _scalar(step: int, key: str = "train/loss", value: float | None = None) -> MetricRecord:
    return MetricRecord(
        type="scalar",
        key=key,
        step=step,
        wall_time_ns=1_700_000_000_000_000_000 + step,
        value=float(step) if value is None else value,
        tags=None,
    )


def test_scalars_returns_parallel_arrays(tmp_path: Path) -> None:
    """ac-007: ``MetricsReader.scalars(key)`` returns parallel ``(steps, values)``.

    Appending 50 scalar records under ``"train/loss"`` with steps 0..49
    must round-trip through ``scalars`` as two equal-length numpy arrays
    where ``steps == arange(50)`` and ``values`` matches the appended
    sequence.
    """
    from molix.recorder.reader import MetricsReader
    from molix.recorder.zarr_store import ZarrMetricStore

    with ZarrMetricStore(tmp_path, run_id="run0") as store:
        for i in range(50):
            store.append(_scalar(step=i, value=float(i) * 0.1))

    reader = MetricsReader(tmp_path, run_id="run0")
    steps, values = reader.scalars("train/loss")

    assert steps.shape == (50,)
    assert values.shape == (50,)
    np.testing.assert_array_equal(steps, np.arange(50, dtype=steps.dtype))
    np.testing.assert_allclose(values, np.arange(50) * 0.1)


def test_reader_works_without_index(tmp_path: Path) -> None:
    """ac-008: ``MetricsReader`` rebuilds the index when ``metrics/index`` is absent.

    The index is advisory per molrec spec §Index — readers must
    reconstitute from records when the derived index is missing.
    """
    from molix.recorder.reader import MetricsReader
    from molix.recorder.zarr_store import ZarrMetricStore

    with ZarrMetricStore(tmp_path, run_id="run0") as store:
        for i in range(10):
            store.append(_scalar(step=i, key="train/loss"))
            store.append(_scalar(step=i, key="eval/MAE", value=0.1 * i))

    reader_full = MetricsReader(tmp_path, run_id="run0")
    keys_full = sorted(reader_full.keys())
    len_full = len(reader_full)

    index_dir = tmp_path / "metrics" / "records" / "run0" / "index"
    if index_dir.exists():
        shutil.rmtree(index_dir)

    reader_no_index = MetricsReader(tmp_path, run_id="run0")
    assert sorted(reader_no_index.keys()) == keys_full
    assert len(reader_no_index) == len_full


def test_records_iterates_in_append_order(tmp_path: Path) -> None:
    """``records()`` must yield records in append order."""
    from molix.recorder.reader import MetricsReader
    from molix.recorder.zarr_store import ZarrMetricStore

    appended = [
        _scalar(step=0, key="train/loss", value=1.0),
        _scalar(step=0, key="eval/MAE", value=0.5),
        _scalar(step=1, key="train/loss", value=0.9),
    ]
    with ZarrMetricStore(tmp_path, run_id="run0") as store:
        for r in appended:
            store.append(r)

    reader = MetricsReader(tmp_path, run_id="run0")
    got = list(reader.records())
    assert len(got) == 3
    for orig, observed in zip(appended, got, strict=True):
        assert observed.type == orig.type
        assert observed.key == orig.key
        assert observed.step == orig.step
        assert observed.value == orig.value
