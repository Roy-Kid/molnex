# molix.data

Molecular data pipeline with nested TensorDict batch types.

- `types`: TensorDict subclasses — `AtomData`, `EdgeData`, `GraphData`, `GraphBatch`
- `task`: Task hierarchy — `SampleTask`, `DatasetTask`, `BatchTask`
- `source`: Data source abstraction (`DataSource` protocol, `InMemorySource`, `SubsetSource`)
- `pipeline`: Declarative container **and** orchestrator — `Pipeline` builder, `PipelineSpec` with `.run` / `.cache_key` / `.cache` / `.build_cache` methods. Owns DDP-aware cache materialisation.
- `cache`: `PackedCache` class — one atomic `.pt` file per cache; owns readiness, mmap load, and rank-aware polling.
- `tasks/`: Built-in tasks (`NeighborList`, `AtomicDress`).
- `collate`: `collate_molecules` converts sample dicts → nested `GraphBatch`.
- `dataset`: `MmapDataset` / `CachedDataset` read `PackedCache` files; `SubsetDataset` is an index view.
- `datamodule`: DDP-aware `DataModule` integrating datasets + collation + post-collate `BatchTask`s.
