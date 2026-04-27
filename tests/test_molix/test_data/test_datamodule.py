"""Tests for DataModule."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from molix.data.cache import PackedCache
from molix.data.collate import DEFAULT_TARGET_SCHEMA
from molix.data.datamodule import DataModule
from molix.data.dataset import CachedDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_samples(n: int = 10) -> list[dict]:
    return [
        {
            "Z": torch.tensor([1, 6], dtype=torch.long),
            "pos": torch.randn(2, 3),
            "edge_index": torch.tensor([[0, 1]], dtype=torch.long),
            "bond_diff": torch.randn(1, 3),
            "bond_dist": torch.tensor([1.5]),
            "targets": {"U0": torch.tensor([float(i)])},
        }
        for i in range(n)
    ]


def _write_and_load_split(tmp_path: Path, train_n: int, val_n: int,
                          dataset_cls=CachedDataset):
    samples = _make_samples(train_n + val_n)
    train_sink = tmp_path / "train.pt"
    val_sink = tmp_path / "val.pt"
    PackedCache(train_sink).save(samples[:train_n])
    PackedCache(val_sink).save(samples[train_n:])
    return dataset_cls(train_sink), dataset_cls(val_sink)


@pytest.fixture
def _dm_factory(tmp_path):
    def _make(**kwargs) -> DataModule:
        train, val = _write_and_load_split(tmp_path, 8, 2)
        # Tiny test datasets — pickling/spawning overhead outweighs parallelism.
        # The user-facing DataModule default is num_workers=4; tests pin 0.
        kwargs.setdefault("pin_memory", False)
        kwargs.setdefault("num_workers", 0)
        return DataModule(train, val, **kwargs)
    return _make


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestDataModuleConstruction:
    def test_stores_datasets(self, tmp_path):
        train, val = _write_and_load_split(tmp_path, 4, 2)
        dm = DataModule(train, val, batch_size=2, pin_memory=False)
        assert dm.train_dataset is train
        assert dm.val_dataset is val
        assert dm.batch_size == 2

    def test_setup_is_noop(self, _dm_factory):
        dm = _dm_factory()
        dm.setup("fit")
        dm.setup("test")

    def test_persistent_workers_requires_num_workers(self, tmp_path):
        train, val = _write_and_load_split(tmp_path, 4, 2)
        dm = DataModule(
            train, val,
            num_workers=0,
            persistent_workers=True,
            pin_memory=False,
        )
        assert dm.persistent_workers is False


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------


class TestDataLoaders:
    def test_train_dataloader_returns_dataloader(self, _dm_factory):
        dm = _dm_factory(batch_size=4)
        assert isinstance(dm.train_dataloader(), DataLoader)

    def test_val_dataloader_returns_dataloader(self, _dm_factory):
        dm = _dm_factory(batch_size=4)
        assert isinstance(dm.val_dataloader(), DataLoader)

    def test_train_dataloader_batch_size(self, _dm_factory):
        dm = _dm_factory(batch_size=4)
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert batch["atoms", "Z"].shape[0] == 8   # 4 mols × 2 atoms

    def test_val_dataloader_no_shuffle(self, _dm_factory):
        dm = _dm_factory(batch_size=2)
        dl = dm.val_dataloader()
        assert dl.sampler is None or not getattr(dl.sampler, "shuffle", False)

    def test_multiple_epochs_consistent(self, _dm_factory):
        dm = _dm_factory(batch_size=4)
        dl1 = dm.train_dataloader()
        dl2 = dm.train_dataloader()
        assert isinstance(dl1, DataLoader)
        assert isinstance(dl2, DataLoader)


# ---------------------------------------------------------------------------
# Collation integration
# ---------------------------------------------------------------------------


class TestCollation:
    def test_batch_has_graph_batch_structure(self, _dm_factory):
        from molix.data.types import GraphBatch
        dm = _dm_factory(batch_size=4)
        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, GraphBatch)

    def test_targets_in_graphs(self, _dm_factory):
        dm = _dm_factory(batch_size=4)
        batch = next(iter(dm.train_dataloader()))
        assert "U0" in batch["graphs"].keys()
        assert batch["graphs", "U0"].shape == (4,)

    def test_custom_target_schema(self, tmp_path):
        from molix.data.collate import TargetSchema
        train, val = _write_and_load_split(tmp_path, 4, 2)
        schema = TargetSchema(graph_level={"U0"}, atom_level=set())
        dm = DataModule(train, val, target_schema=schema, batch_size=2,
                        num_workers=0, pin_memory=False)
        batch = next(iter(dm.train_dataloader()))
        assert "U0" in batch["graphs"].keys()

    def test_target_schema_auto_discovered_from_dataset(self, tmp_path):
        from molix.data.collate import TargetSchema

        class SchemaCarrier(CachedDataset):
            target_schema = TargetSchema(graph_level={"U0"}, atom_level=frozenset())

        samples = _make_samples(6)
        tsink, vsink = tmp_path / "t.pt", tmp_path / "v.pt"
        PackedCache(tsink).save(samples[:4])
        PackedCache(vsink).save(samples[4:])
        dm = DataModule(SchemaCarrier(tsink), SchemaCarrier(vsink),
                        batch_size=2, pin_memory=False)
        assert dm.target_schema.graph_level == frozenset({"U0"})

    def test_explicit_target_schema_overrides_dataset(self, tmp_path):
        from molix.data.collate import TargetSchema

        class SchemaCarrier(CachedDataset):
            target_schema = TargetSchema(graph_level={"foo"}, atom_level=frozenset())

        samples = _make_samples(6)
        tsink, vsink = tmp_path / "t.pt", tmp_path / "v.pt"
        PackedCache(tsink).save(samples[:4])
        PackedCache(vsink).save(samples[4:])
        explicit = TargetSchema(graph_level={"U0"}, atom_level=frozenset())
        dm = DataModule(SchemaCarrier(tsink), SchemaCarrier(vsink),
                        target_schema=explicit, batch_size=2, pin_memory=False)
        assert dm.target_schema is explicit


# ---------------------------------------------------------------------------
# Pickling / forkserver compatibility (Python 3.14 default)
# ---------------------------------------------------------------------------


class TestPickling:
    def test_collate_fn_is_picklable(self, _dm_factory):
        import pickle

        dm = _dm_factory(batch_size=2)
        fn = dm._make_collate_fn()
        fn2 = pickle.loads(pickle.dumps(fn))
        samples = _make_samples(2)
        out1 = fn(samples)
        out2 = fn2(samples)
        assert torch.equal(out1["atoms", "Z"], out2["atoms", "Z"])


# ---------------------------------------------------------------------------
# Worker start method (Python 3.14 multi-threaded-fork deprecation)
# ---------------------------------------------------------------------------


class TestWorkerContext:
    def test_default_is_spawn(self, _dm_factory):
        """Default sidesteps Python 3.14's multi-threaded forkserver warning."""
        dm = _dm_factory()
        assert dm.multiprocessing_context == "spawn"

    def test_context_forwarded_when_workers_enabled(self, _dm_factory):
        dm = _dm_factory(num_workers=2, persistent_workers=False)
        dl = dm.train_dataloader()
        # torch.DataLoader resolves the string to a BaseContext object.
        ctx = dl.multiprocessing_context
        assert ctx is not None
        # Accept either the raw string (some torch versions keep it) or a
        # resolved BaseContext (others resolve eagerly).
        name = ctx if isinstance(ctx, str) else ctx.get_start_method()
        assert name == "spawn"

    def test_context_ignored_when_num_workers_zero(self, _dm_factory):
        dm = _dm_factory(num_workers=0)
        dl = dm.train_dataloader()
        # DataLoader normalises an unused context to ``None`` in sync mode.
        assert dl.multiprocessing_context is None

    def test_explicit_override(self, tmp_path):
        train, val = _write_and_load_split(tmp_path, 4, 2)
        dm = DataModule(
            train, val,
            batch_size=2,
            num_workers=2,
            persistent_workers=False,
            pin_memory=False,
            multiprocessing_context="forkserver",
        )
        assert dm.multiprocessing_context == "forkserver"
        dl = dm.train_dataloader()
        ctx = dl.multiprocessing_context
        name = ctx if isinstance(ctx, str) else ctx.get_start_method()
        assert name == "forkserver"


# ---------------------------------------------------------------------------
# Epoch hook
# ---------------------------------------------------------------------------


class TestEpochHook:
    def test_on_epoch_start_no_crash_without_ddp(self, _dm_factory):
        dm = _dm_factory()
        dm.train_dataloader()
        dm.on_epoch_start(0)
        dm.on_epoch_start(1)


# ---------------------------------------------------------------------------
# from_cached_pipeline factory
# ---------------------------------------------------------------------------


class TestFromCachedPipeline:
    def test_end_to_end_factory(self, tmp_path):
        """Factory caches, splits, exposes stats, yields GraphBatch."""
        from molix.data import InMemorySource, NeighborList, Pipeline
        from molix.data.types import GraphBatch

        samples = _make_samples(10)
        source = InMemorySource(samples)
        pipe = (
            Pipeline("factory-smoke")
            .add(NeighborList(cutoff=3.0, max_num_pairs=32, pbc=False))
            .build()
        )
        dm = DataModule.from_cached_pipeline(
            pipe, source,
            base_dir=tmp_path,
            split_sizes=(6, 2, 2),
            seed=7,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        # Splits wired through
        assert len(dm.train_dataset) == 6
        assert len(dm.val_dataset) == 2
        assert dm.test_dataset is not None and len(dm.test_dataset) == 2
        assert dm.split_indices is not None and len(dm.split_indices) == 3
        assert dm.full_dataset is not None and len(dm.full_dataset) == 10

        # Connectivity stats derived from packed-cache pointers — no task needed.
        assert dm.full_dataset.avg_num_neighbors > 0.0
        assert dm.full_dataset.max_atoms > 0
        assert dm.full_dataset.max_edges > 0
        # Subsets report split-local stats (no peek at val/test).
        assert dm.train_dataset.avg_num_neighbors > 0.0
        assert dm.train_dataset.max_atoms <= dm.full_dataset.max_atoms

        # DataLoader produces a GraphBatch
        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, GraphBatch)
        assert batch["atoms", "Z"].shape[0] == 4   # 2 mols × 2 atoms

    def test_two_way_split(self, tmp_path):
        from molix.data import (
            InMemorySource, NeighborList, Pipeline,
        )

        samples = _make_samples(8)
        pipe = (
            Pipeline("two-way")
            .add(NeighborList(cutoff=3.0, max_num_pairs=32, pbc=False))
            .build()
        )
        dm = DataModule.from_cached_pipeline(
            pipe, InMemorySource(samples),
            base_dir=tmp_path,
            split_sizes=(5, 3),
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        assert len(dm.train_dataset) == 5
        assert len(dm.val_dataset) == 3
        assert dm.test_dataset is None

    def test_rejects_oversized_split(self, tmp_path):
        from molix.data import InMemorySource, NeighborList, Pipeline

        pipe = (
            Pipeline("over")
            .add(NeighborList(cutoff=3.0, max_num_pairs=32, pbc=False))
            .build()
        )
        with pytest.raises(ValueError, match="exceeds"):
            DataModule.from_cached_pipeline(
                pipe, InMemorySource(_make_samples(5)),
                base_dir=tmp_path,
                split_sizes=(4, 3),   # sum=7 > 5
                pin_memory=False,
            )

    def test_rejects_single_split(self, tmp_path):
        from molix.data import InMemorySource, NeighborList, Pipeline

        pipe = Pipeline("one").add(NeighborList(cutoff=3.0, pbc=False)).build()
        with pytest.raises(ValueError, match=">= 2"):
            DataModule.from_cached_pipeline(
                pipe, InMemorySource(_make_samples(5)),
                base_dir=tmp_path,
                split_sizes=(5,),
                pin_memory=False,
            )
