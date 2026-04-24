"""Data pipeline for molecular ML.

Layering:
    task     — Task / SampleTask / DatasetTask / BatchTask (transform primitives).
    source   — DataSource protocol + in-memory and subset sources.
    pipeline — declarative container: what tasks to run, in what order, with
               what identity; plus ``.run`` / ``.cache`` / ``.cache_key``
               methods for execution and DDP-aware materialisation.
    cache    — :class:`PackedCache`: short-lived, atomically-written scratch
               file that stores the output of ``PipelineSpec.cache``.
    dataset  — MmapDataset, CachedDataset, SubsetDataset: readers over
               :class:`PackedCache` files.

There is no free-function cache / execute / ddp surface — all orchestration
lives on :class:`PipelineSpec` (and on :class:`PackedCache` for file IO).
"""

# Task hierarchy
from molix.data.task import (
    BatchTask,
    DatasetTask,
    Runnable,
    SampleTask,
    Task,
)

# Built-in tasks
from molix.data.tasks import AtomicDress, NeighborList, UnitConvert

# Data sources
from molix.data.source import DataSource, InMemorySource, SubsetSource

# Pipeline DSL
from molix.data.pipeline import Pipeline, PipelineSpec, TaskEntry

# Cache
from molix.data.cache import PackedCache

# Dataset classes
from molix.data.dataset import (
    BaseDataset,
    CachedDataset,
    MmapDataset,
    SubsetDataset,
)

# DataModule
from molix.data.datamodule import DataModule, DataModuleProtocol

# Collation
from molix.data.collate import (
    DEFAULT_TARGET_SCHEMA,
    TargetSchema,
    collate_molecules,
)

# Types
from molix.data.types import (
    AtomData,
    EdgeData,
    EdgeRepEdges,
    GraphBatch,
    GraphData,
    NodeRepAtoms,
)

__all__ = [
    # Task hierarchy
    "Task",
    "SampleTask",
    "DatasetTask",
    "BatchTask",
    "Runnable",
    # Built-in tasks
    "NeighborList",
    "AtomicDress",
    "UnitConvert",
    # Sources
    "DataSource",
    "InMemorySource",
    "SubsetSource",
    # Pipeline
    "Pipeline",
    "PipelineSpec",
    "TaskEntry",
    # Cache
    "PackedCache",
    # Dataset classes
    "BaseDataset",
    "CachedDataset",
    "MmapDataset",
    "SubsetDataset",
    # DataModule
    "DataModule",
    "DataModuleProtocol",
    # Collation
    "collate_molecules",
    "TargetSchema",
    "DEFAULT_TARGET_SCHEMA",
    # Types
    "AtomData",
    "EdgeData",
    "GraphData",
    "GraphBatch",
    "NodeRepAtoms",
    "EdgeRepEdges",
]
