"""DDP-aware DataModule."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from molix.data.collate import DEFAULT_TARGET_SCHEMA, TargetSchema, collate_molecules
from molix.data.dataset import BaseDataset, MmapDataset, SubsetDataset, split_indices
from molix.data.pipeline import TaskEntry
from molix.data.source import SubsetSource

if TYPE_CHECKING:
    from molix.data.pipeline import PipelineSpec


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DataModuleProtocol(Protocol):
    """Protocol consumed by the Trainer."""

    def setup(self, stage: str = "fit") -> None: ...
    def train_dataloader(self) -> Iterable: ...
    def val_dataloader(self) -> Iterable: ...
    def on_epoch_start(self, epoch: int) -> None: ...


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _get_world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class DataModule:
    """DDP-aware DataLoader wrapper.

    Takes pre-built train/val datasets and wraps them in DataLoaders.
    Dataset construction (downloading, pipeline transforms, caching,
    storage strategy) is the dataset class's responsibility — not ours.

    Usage::

        packed = pipe.cache(source, base_dir=run_dir / "cache",
                            fit_source=train_source)
        ds = MmapDataset(packed.sink)
        train_ds, val_ds = ds.split(ratio=0.8)
        dm = DataModule(train_ds, val_ds,
                        target_schema=QM9Source.TARGET_SCHEMA,
                        batch_tasks=pipe.batch_tasks,
                        batch_size=32, num_workers=4)
        trainer.train(datamodule=dm, max_epochs=100)

    Args:
        train_dataset: Pre-built training dataset.
        val_dataset: Pre-built validation dataset.
        target_schema: Which target keys are graph-level vs atom-level.
        batch_tasks: Post-collate transforms (from pipeline.batch_tasks).
        batch_size: Samples per batch (per rank in DDP).
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors for faster GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Batches prefetched per worker.
        seed: RNG seed for DDP sampler shuffling.
        multiprocessing_context: Start method for DataLoader worker
            processes. Defaults to ``"spawn"`` — this is the future-proof
            choice for three reasons:

            1. Python 3.14 switched the POSIX default start method to
               ``forkserver``. The forkserver daemon itself is launched
               via ``os.fork()``, and PyTorch (plus most scientific
               libs) spawns background threads at import, so by the
               time the first DataLoader is built the main process is
               already multi-threaded. That triggers
               ``DeprecationWarning: This process ... is multi-threaded,
               use of fork() may lead to deadlocks in the child``.
               ``spawn`` never forks.
            2. ``spawn`` is the only safe start method when CUDA
               tensors live in the parent (``fork`` silently duplicates
               CUDA contexts, which is undefined behaviour).
            3. It's the default on macOS and Windows already, so
               ``spawn``-clean code is what survives portability.

            The cost is a slower worker boot (~1–5 s each), paid once
            per run when ``persistent_workers=True`` — negligible for
            any real training job. Pass ``"fork"`` / ``"forkserver"``
            explicitly if you need to opt out (e.g. pure-CPU Linux
            jobs with cold-start cost sensitivity). Ignored when
            ``num_workers == 0``.
    """

    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: BaseDataset,
        *,
        target_schema: TargetSchema | None = None,
        batch_tasks: Sequence[TaskEntry] | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = None,
        seed: int = 42,
        multiprocessing_context: str | None = "spawn",
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # Auto-discover schema via getattr (some dataset subclasses declare
        # a .target_schema attribute). Explicit user override always wins.
        if target_schema is None:
            target_schema = getattr(train_dataset, "target_schema", DEFAULT_TARGET_SCHEMA)
        self.target_schema = target_schema
        self.batch_tasks: tuple[TaskEntry, ...] = tuple(batch_tasks) if batch_tasks else ()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self.multiprocessing_context = multiprocessing_context

        self._train_sampler: DistributedSampler | None = None
        self._val_sampler: DistributedSampler | None = None
        # Set by :meth:`from_cached_pipeline`; ``None`` under direct construction.
        self.full_dataset: BaseDataset | None = None
        self.test_dataset: BaseDataset | None = None
        self.split_indices: tuple[list[int], ...] | None = None

    # -- Factory ------------------------------------------------------------

    @classmethod
    def from_cached_pipeline(
        cls,
        pipe: "PipelineSpec",
        source: Any,
        *,
        base_dir: str | Path,
        split_sizes: Sequence[int],
        seed: int = 42,
        extra: dict[str, str] | None = None,
        target_schema: TargetSchema | None = None,
        **dataloader_kwargs: Any,
    ) -> "DataModule":
        """One-shot: fit pipeline on train split, cache, and build a DataModule.

        Absorbs the boilerplate
        ``pipe.cache(source, ...) → MmapDataset → SubsetDataset → DataModule``
        found in every training script, and wires the fit-source to the
        training split so the fitted state matches the split the model
        trains on.

        Args:
            pipe: A built :class:`~molix.data.pipeline.PipelineSpec`.
            source: Full :class:`~molix.data.source.DataSource` (all
                samples; this factory handles splitting).
            base_dir: Directory for the cache file.
            split_sizes: Per-split sample counts. Length 2 for
                ``(train, val)`` or length 3 for ``(train, val, test)``.
                Sum must be ``<= len(source)``.
            seed: RNG seed for the random permutation used to produce
                split indices.
            extra: Extra identity fields for the cache key (same meaning
                as in :meth:`PipelineSpec.cache`).
            target_schema: Override graph-level vs atom-level target
                classification. Defaults to
                ``getattr(source, "TARGET_SCHEMA", DEFAULT_TARGET_SCHEMA)``.
            **dataloader_kwargs: Forwarded to ``cls(...)`` (batch_size,
                num_workers, pin_memory, persistent_workers, etc.).

        Returns:
            A ready-to-use :class:`DataModule`. ``dm.full_dataset`` gives
            access to ``.stats()``; ``dm.test_dataset`` holds the test
            split when ``split_sizes`` has three entries;
            ``dm.split_indices`` holds the per-split index lists (useful
            for reproducibility logging).
        """
        if len(split_sizes) < 2:
            raise ValueError(
                f"split_sizes must have >= 2 entries (train, val[, test]), "
                f"got {split_sizes!r}"
            )
        # split_indices validates sum(sizes) <= n and raises ValueError otherwise.
        indices = split_indices(len(source), split_sizes, seed=seed)
        fit_source = SubsetSource(source, indices[0])
        packed = pipe.cache(
            source,
            base_dir=Path(base_dir),
            fit_source=fit_source,
            extra=extra,
        )
        full = MmapDataset(packed.sink)
        subsets = [SubsetDataset(full, idx) for idx in indices]

        schema = target_schema
        if schema is None:
            schema = getattr(source, "TARGET_SCHEMA", None) or DEFAULT_TARGET_SCHEMA

        dm = cls(
            subsets[0],
            subsets[1],
            target_schema=schema,
            batch_tasks=pipe.batch_tasks,
            **dataloader_kwargs,
        )
        dm.full_dataset = full
        dm.test_dataset = subsets[2] if len(subsets) >= 3 else None
        dm.split_indices = tuple(indices)
        return dm

    def _worker_context(self) -> str | None:
        """Start method passed to :class:`DataLoader`, or ``None`` for sync.

        ``DataLoader`` rejects ``multiprocessing_context`` when
        ``num_workers == 0``, so we only forward it for the async path.
        """
        return self.multiprocessing_context if self.num_workers > 0 else None

    # -- Lifecycle (Trainer calls these) ------------------------------------

    def setup(self, stage: str = "fit") -> None:
        pass  # datasets are ready at construction time

    def train_dataloader(self) -> DataLoader:
        if _is_distributed():
            self._train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=_get_world_size(),
                rank=_get_rank(),
                shuffle=True,
                seed=self.seed,
            )
            shuffle = False
        else:
            self._train_sampler = None
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=self._train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self._make_collate_fn(),
            drop_last=_is_distributed(),
            multiprocessing_context=self._worker_context(),
        )

    def val_dataloader(self) -> DataLoader:
        if _is_distributed():
            self._val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=_get_world_size(),
                rank=_get_rank(),
                shuffle=False,
            )
        else:
            self._val_sampler = None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self._val_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self._make_collate_fn(),
            multiprocessing_context=self._worker_context(),
        )

    def _make_collate_fn(self) -> "_CollateFn":
        return _CollateFn(self.target_schema, self.batch_tasks)

    # -- Epoch hook ---------------------------------------------------------

    def on_epoch_start(self, epoch: int) -> None:
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)
        if self._val_sampler is not None:
            self._val_sampler.set_epoch(epoch)


# ---------------------------------------------------------------------------
# Picklable collate wrapper (required for non-fork start methods)
# ---------------------------------------------------------------------------


class _CollateFn:
    """Picklable collate callable for DataLoader workers.

    ``spawn`` / ``forkserver`` start methods both send the collate callable
    to workers through ``pickle``, so a local closure won't survive the
    trip. A top-level class keeps it picklable on every supported Python
    and every platform (``spawn`` is already the default on macOS/Windows
    and is what we default to in :class:`DataModule` to sidestep the
    Python 3.14 multi-threaded-fork DeprecationWarning).
    """

    def __init__(
        self, schema: TargetSchema, batch_tasks: Sequence[TaskEntry]
    ) -> None:
        self.schema = schema
        self.batch_tasks = batch_tasks

    def __call__(self, samples: list[dict]) -> dict:
        batch = collate_molecules(samples, self.schema)
        for entry in self.batch_tasks:
            batch = entry.apply(batch)
        return batch
