---
title: Sonata 05 — In-tree bench manifest schema and model factory
status: approved
created: 2026-05-11
---

# Sonata 05 — In-tree bench manifest schema and model factory

## Summary

本子规范是 Sonata 端到端工作流（bulk water RPBE-D3）五子链中的第 2 个，位于
`sonata-05-data` 之后、`sonata-05-train` / `sonata-05-md` / `sonata-05-obs`
之前。它在新的 `src/molix/bench/` 侧位子包中落地两件基础设施：
（1）`BenchManifest` —— 描述「system × model × stage × split × seed × hyperparams」
组合的 Pydantic 清单；（2）`model_from_manifest` —— 从清单一次性构造与
`benchmarks/bm_molpot/bm_sonata.py:_build_sonata_and_baseline` 等价的
Allegro baseline 或 Sonata 模型。同时确立 `runs/<system>/<model>/<stage>/`
作为整链的产物布局。所有下游 stage 驱动器只读取 `BenchManifest`、调用
`model_from_manifest` 与 `manifest.to_artifact_dir()`，从而保证 train、eval、
md、obs 看到的模型字位一致、产物落点一致。

## Design

`molix.bench` 与 `molix.profiler` 同为「侧位基础设施」—— 训练代码绝不
import 它，但端到端 driver 与外部 CLI 通过它统一描述实验意图。

新增三个文件（均位于 `src/molix/bench/`）：

- `__init__.py` —— 仅做 re-export：`BenchManifest`、`system_artifact_dir`、
  `model_from_manifest`。
- `manifest.py` —— `BenchManifest(BaseModel)`，使用
  `ConfigDict(arbitrary_types_allowed=True, frozen=True)`：

  ```
  system:          Literal["water_rpbe_d3"]
  model:           Literal["allegro", "sonata"]
  stage:           Literal["train", "eval", "md", "obs"]
  split:           Literal["train", "val", "test"] | None = None
  runs_root:       Path = Path("./runs")
  seed:            int = 0
  hyperparams:     dict[str, Any] = {}
  dataset_root:    Path | None = None
  checkpoint_path: Path | None = None
  ```

  辅助：

  - `classmethod from_dict(cls, d: Mapping[str, Any]) -> BenchManifest`
    —— 透传 Pydantic 验证。
  - `classmethod from_yaml(cls, path: str | Path) -> BenchManifest`
    —— 内部 `yaml.safe_load` 后委托 `from_dict`。
  - `def to_artifact_dir(self, *, overwrite: bool = False) -> Path`
    —— 解析为 `runs_root / system / model / stage`；当目录已存在且
    非空且 `overwrite=False` 时抛 `FileExistsError`；否则
    `mkdir(parents=True, exist_ok=True)` 并返回该 `Path`。`stage` 自身
    所有 split 共享同一目录（split 用于 dataset 读取，不参与产物路径）。

  顶层 helper：

  - `def system_artifact_dir(system: str, model: str, stage: str, *,
    runs_root: Path = Path("./runs")) -> Path` —— 纯路径计算，不创建
    目录；测试和外部脚本用它做断言。

- `factory.py` —— `model_from_manifest(manifest: BenchManifest) -> nn.Module`：

  - 解析 hyperparams（与 `bm_sonata.py:51–95` 对齐的默认值）：
    `r_max=5.0`、`l_max=2`、`num_features=64`、`num_layers=2`、
    `num_elements=20`、`avg_num_neighbors=12.0`、`sigma=1.0`、`dl=2.0`、
    `type_embed_dim=32`、`latent_mlp_width=64`、`latent_mlp_depth=1`、
    `num_bessel=8`。`manifest.hyperparams` 中显式给出的键覆盖默认值，
    未识别的键忽略（escape hatch；后续子规范会引入强类型 sub-model
    收紧此处）。
  - `manifest.model == "allegro"`：以 `manifest.seed` 调用
    `torch.manual_seed`，构造一个 `expose_tensor_track=False` 的
    `Allegro`，再以 `seed+1` 构造同形 `EdgeEnergyHead(out_key="energy")`，
    用从 `bm_sonata.py` 抽离的薄 `nn.Module` 包装器（命名为
    `_ShortRangeBaseline`，与 `bm_sonata.py:459` 的实现一致）返回。
  - `manifest.model == "sonata"`：以同样的种子方案构造
    `expose_tensor_track=True` 的 `Allegro` 与 short-range
    `EdgeEnergyHead(out_key="energy_short")`，然后调用
    `molpot.composition.build_sonata(encoder, sigma=sigma, dl=dl,
    charge=True, dipole=True, quadrupole=True,
    constrain_total_charge=True, avg_num_neighbors=avg_num_neighbors,
    short_range_head=short_head)` 并返回。
  - `factory.py` 仅承担「manifest → nn.Module」职责；下游 train / md /
    obs driver 由后续子规范实现，不在此处出现。

生命周期：`BenchManifest` 不可变（`frozen=True`），因而可以被 train
driver、md driver、observable driver 同步引用而不发生意外改写；
`to_artifact_dir` 是唯一具有副作用（mkdir / FileExistsError）的方法，
默认拒绝写入非空目录避免覆盖既有 run。

## Files to create or modify

- `src/molix/bench/__init__.py` (new)
- `src/molix/bench/manifest.py` (new)
- `src/molix/bench/factory.py` (new)
- `tests/test_molix/test_bench/__init__.py` (new)
- `tests/test_molix/test_bench/test_manifest.py` (new)
- `tests/test_molix/test_bench/test_factory.py` (new)

## Tasks

- [ ] Write failing tests for BenchManifest schema validation and artifact-dir resolution (tests/test_molix/test_bench/test_manifest.py)
- [ ] Implement BenchManifest with from_dict/from_yaml/to_artifact_dir in src/molix/bench/manifest.py and system_artifact_dir helper
- [ ] Write failing tests for model_from_manifest parity with bm_sonata._build_sonata_and_baseline on both allegro and sonata branches (tests/test_molix/test_bench/test_factory.py)
- [ ] Implement model_from_manifest in src/molix/bench/factory.py reusing the _ShortRangeBaseline wrapper and build_sonata call shape from benchmarks/bm_molpot/bm_sonata.py
- [ ] Wire src/molix/bench/__init__.py to re-export BenchManifest, system_artifact_dir, model_from_manifest and add tests/test_molix/test_bench/__init__.py
- [ ] Add Google-style docstrings with units on every public symbol in src/molix/bench/*.py
- [ ] Run full check + test suite

## Testing strategy

Happy path:

- `BenchManifest.from_dict({...minimal fields...})` validates and returns a
  frozen instance; `to_artifact_dir()` creates
  `runs/water_rpbe_d3/allegro/train/` under a `tmp_path` `runs_root`.
- `BenchManifest.from_yaml(tmp_path / "m.yaml")` round-trips a written
  YAML file and matches `from_dict` on the same payload.
- `model_from_manifest(manifest_allegro)` returns an `nn.Module` whose
  `forward(batch, compute_forces=True)` produces `{"energy", "forces"}` on
  a 4-water synthetic batch (reuse `bm_sonata._build_synthetic_water_box`
  helper via direct call from the test module).
- `model_from_manifest(manifest_sonata)` returns a Sonata composer whose
  `forward(batch, compute_forces=True)` produces an `energy` of the same
  dtype / shape as the Allegro branch.
- Two `model_from_manifest` calls with the same `seed` and
  `manifest.model == "allegro"` produce **bit-identical** parameter
  tensors (key invariant — downstream md / obs reload via this path).

Edge cases:

- `from_dict({...invalid system...})` raises `pydantic.ValidationError`.
- Assigning to a manifest field after construction raises `ValidationError`
  (frozen check).
- `to_artifact_dir()` raises `FileExistsError` when target exists and is
  non-empty and `overwrite=False`; succeeds with `overwrite=True` or on
  an empty pre-existing dir.
- `stage="md"` accepts `split=None`; `stage="train"` with `split=None` is
  allowed by the schema (the consuming sub-spec decides the requirement).
- Unknown keys in `hyperparams` are ignored by `model_from_manifest` and
  do not raise.

Out of scope from testing here: training-loop correctness, MD
trajectory shape, observable values — those live in 03 / 04 / 05.

## Out of scope

- Strongly-typed `TrainHyperparams` / `MDHyperparams` / `ObsHyperparams`
  Pydantic sub-models. They will be introduced by `sonata-05-train`,
  `sonata-05-md`, `sonata-05-obs` respectively and will replace the
  current `hyperparams: dict[str, Any]` escape hatch incrementally.
- Hydra / OmegaConf integration. `from_yaml` is intentionally a thin
  `yaml.safe_load` wrapper; multi-file composition / interpolation is
  deferred.
- Multi-system support beyond `water_rpbe_d3`. The `Literal` annotation
  is designed to grow with a one-line append, but the first round is
  single-system. Cross-system schema migration is out of scope.
- A CLI entrypoint (`python -m molix.bench …`) — none of train / eval /
  md / obs lives in this sub-spec, so an entrypoint here would be
  orphaned.
- Wiring `WaterLESSource` inside the manifest. The manifest carries
  `dataset_root` as a `Path` only; instantiation happens in the
  consuming `sonata-05-train` driver.
