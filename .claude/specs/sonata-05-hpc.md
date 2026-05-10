---
title: Sonata HPC — first-class DDP wrap shim for molix.core.Trainer
status: approved
created: 2026-05-10
---

# Sonata HPC — first-class DDP wrap shim for molix.core.Trainer

## Summary

为 Sonata HPC 科学正确性验证链（`sonata-05-validation`）的训练驱动落地多 GPU 基础设施。本规范在 `molix.core` 层加入第一个 first-class 的 DDP 包装层 `wrap_for_ddp` / `teardown_ddp`，使 `bm_sonata_hpc.py` 以及未来任何需要 `--world-size > 1` 的基准驱动都可以通过 `torchrun --standalone --nproc_per_node=N` 在单节点多 GPU 上启动。同时审计四个会写文件 / 写事件的 hook（`CheckpointHook` / `JournalHook` / `TensorBoardHook` / `ScalarHook`）以及 `JournalWriter`，确保所有磁盘写入都被 rank-0 守卫，避免多 rank 同时写入产生竞争或文件损坏。`Trainer` 本身保持单一职责（编排），不增加 `ddp=True` 形式的耦合 kwarg —— DDP 是用户调用方的责任，由 `wrap_for_ddp` 在交给 `Trainer` 之前完成。

## Domain basis

本子规范是工程集成性质（PyTorch DDP idiom），不引入新的物理。引用：

- PyTorch DDP 文档：<https://pytorch.org/docs/stable/notes/ddp.html>
- torchrun（弹性启动）：<https://pytorch.org/docs/stable/elastic/run.html>
- 现有 DDP-aware 数据层：`src/molix/data/datamodule.py:41-50`（`_is_distributed` / `_get_rank` / `_get_world_size` 环境变量查找模式 — 本规范的 helper 复用同一个约定）。

依赖关系（chain `sonata-05-validation`）：

- **depends on:** `sonata-05-data`（真实 bulk-water + charged-dimer Source 类必须先落盘；本规范的多 rank smoke test 使用 toy fixture，真实数据集由 out-of-tree 训练驱动消费）。
- **consumed by:** out-of-tree HPC 训练驱动（`bm_sonata_hpc.py` 等用户脚本调用 `wrap_for_ddp` 完成端到端 HPC 基准；驱动本身**不进本仓库**）。

二阶 autograd 风险：Sonata 在 `compute_forces=True` 时通过 `torch.autograd.grad` 在 forward 内部计算 `F = -dE/dx`，这与 DDP 的 reducer 钩子之间存在已知的微妙交互（DDP 以为 backward 只走一次）。规范要求对 `Sonata(compute_forces=True)` 在 N≥2 ranks 上单独 smoke-test，与单 rank 数值一致到 `atol=1e-5`，否则将作为运行时 time-bomb 暴露给 out-of-tree 训练驱动。

## Design

新增模块 `src/molix/core/ddp.py`，暴露两个无状态函数和一个轻量 dataclass：

```
@dataclass(frozen=True)
class DistEnv:
    rank: int
    local_rank: int
    world_size: int
    backend: str          # "nccl" | "gloo"
    initialized: bool     # True iff this call (or a prior one) ran init_process_group

def wrap_for_ddp(
    model: nn.Module,
    *,
    device: torch.device | None = None,
    backend: str = "nccl",
    find_unused_parameters: bool = False,
) -> tuple[nn.Module, DistEnv]: ...

def teardown_ddp(env: DistEnv) -> None: ...

def is_rank_zero() -> bool: ...
def get_rank() -> int: ...
def get_world_size() -> int: ...
def barrier() -> None: ...
```

行为：

- `wrap_for_ddp` 从环境变量 `LOCAL_RANK` / `RANK` / `WORLD_SIZE` 读取拓扑（与 `datamodule.py` 同一约定）。当三者均未设置或 `WORLD_SIZE=1`：直接返回 `(model, DistEnv(rank=0, local_rank=0, world_size=1, backend="gloo", initialized=False))`，**不**调用 `init_process_group`，**不**包 DDP（保持单进程 fast path）。
- 当 `WORLD_SIZE>1`：先 `torch.cuda.set_device(local_rank)`（如果 `device` 是 CUDA），再 `init_process_group(backend, ...)`（用 `dist.is_initialized()` 守卫，幂等），把 `model` 移动到 `device`，再用 `DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=find_unused_parameters)` 包起来，返回 wrapped model 和填好的 `DistEnv(initialized=True)`。
- `teardown_ddp(env)` 在 `env.initialized=True` 时调用 `dist.destroy_process_group()`；否则 no-op。设计为对称的 setup/teardown，方便 `bm_sonata_hpc.py` 在 `try/finally` 块里清理。
- `is_rank_zero` / `get_rank` / `get_world_size` / `barrier` 是 `dist.*` 的安全 wrapper：当 `not dist.is_available() or not dist.is_initialized()` 时分别返回 `True` / `0` / `1` / no-op。它们成为 hooks 的统一入口，避免每个 hook 自己粘 `if dist.is_initialized() and dist.get_rank() != 0`。

`Trainer` 不变（除了文档字符串补充一段说明）：用户在调用 `Trainer(model=...)` 之前自己先 `model, dist_env = wrap_for_ddp(model, device=...)`，把 wrapped model 喂给 `Trainer`。`Trainer` 继续把 `model` 当成普通 `nn.Module` 用；DDP 的 reducer 钩子由 PyTorch 在 `loss.backward()` 自动触发。

Hook 端 rank-0 守卫策略：选择 hook 自我守卫（`if not is_rank_zero(): return`）而不是给 `BaseHook` 增加 `ranks_to_run_on` 字段 —— 后者更具侵入性，会污染所有非 IO hook。审计并修改这些 hook 的所有写盘 / 发事件路径：

| hook / 文件 | 守卫位置 |
|---|---|
| `CheckpointHook._save_checkpoint`（`src/molix/hooks/checkpoint.py:130`） | 在 `torch.save` 之前 + 在 `os.makedirs` 之前（`on_train_start`） |
| `JournalHook._mirror_namespaces` / `_emit_histograms` / `on_train_start` / `on_train_end`（`src/molix/hooks/journal.py`） | 入口处单一守卫 |
| `TensorBoardHook` 全部 IO 方法（`src/molix/hooks/tensorboard.py`） | `on_train_start` / `on_train_batch_end` / `on_eval_step_complete` / `on_epoch_end` / `on_train_end` 入口处守卫 |
| `JournalWriter`（`src/molix/io/writer.py`） | 不在 writer 内守卫；调用方（`JournalHook`）守卫；writer 文档补一行 "callers must rank-0-guard"。 |

Eval reduction 策略：本规范选择**不**修改 `MetricsHook` 做 all-reduce，理由有二：(1) 修改会扩散到 metric protocol；(2) `state["eval"][...]` 名空间契约（CLAUDE.md TrainState invariant）只允许单 rank 写入。多 rank 下每个 rank 的 eval 子集不同，但 `DistributedSampler` 在 val 集上保证近似均匀划分；规范允许各 rank 的 eval 标量在 rank 0 上略有偏差（仅作为信号），HPC 准确度评估靠 `sonata-05-driver` 的诊断核 + 显式 all-reduce 完成。在新增的 hook 文档段补一段 "Multi-rank eval semantics" 警告。

## Files to create or modify

- `src/molix/core/ddp.py` (new) — `DistEnv` dataclass + `wrap_for_ddp` / `teardown_ddp` / `is_rank_zero` / `get_rank` / `get_world_size` / `barrier`.
- `src/molix/core/__init__.py` — re-export 上述六个符号。
- `src/molix/core/trainer.py` — 仅在 `Trainer` 类的 docstring（class-level 与 `train` 方法）追加一段说明：DDP 是用户责任，参考 `wrap_for_ddp`；不改任何执行逻辑。
- `src/molix/hooks/checkpoint.py` — `_save_checkpoint` + `on_train_start` 中 `os.makedirs` 之前加 rank-0 守卫。
- `src/molix/hooks/journal.py` — `on_train_start` / `_mirror_namespaces` / `_emit_histograms` / `on_train_end` 入口处守卫。
- `src/molix/hooks/tensorboard.py` — 全部五个 lifecycle 方法入口处守卫。
- `src/molix/io/writer.py` — 仅在模块 / 类 docstring 追加 "callers must rank-0-guard" 一行；不改逻辑。
- `tests/test_molix/test_core/test_ddp.py` (new) — `wrap_for_ddp` / `teardown_ddp` / 单进程 fast path 单元测试 + `is_rank_zero` / `get_rank` / `get_world_size` / `barrier` 的非分布式回退测试。
- `tests/test_molix/test_core/test_ddp_multi_rank.py` (new) — 通过 `torch.multiprocessing.spawn` 启动 N=2 进程，pytest mark `multi_gpu`：覆盖 (a) 普通 MLP forward+backward 数值与单 rank 一致到 atol=1e-5；(b) `Sonata(compute_forces=True)` 同样数值一致到 atol=1e-5（autograd-inside-autograd 的 DDP 兼容性 smoke test）。
- `tests/test_molix/test_hooks/test_rank_zero_guards.py` (new) — 通过 monkeypatch `molix.core.ddp.is_rank_zero` 返回 `False` 验证 `CheckpointHook` / `JournalHook` / `TensorBoardHook` 的所有 IO 路径都不写文件、不调用 writer。

## Tasks

- [ ] Write failing tests for wrap_for_ddp / teardown_ddp single-process fast path (tests/test_molix/test_core/test_ddp.py)
- [ ] Write failing tests for hook rank-0 guards (tests/test_molix/test_hooks/test_rank_zero_guards.py)
- [ ] Write failing multi-rank smoke test (MLP + Sonata compute_forces=True, atol=1e-5) (tests/test_molix/test_core/test_ddp_multi_rank.py)
- [ ] Implement DistEnv + wrap_for_ddp + teardown_ddp + is_rank_zero/get_rank/get_world_size/barrier in src/molix/core/ddp.py
- [ ] Re-export new symbols from src/molix/core/__init__.py
- [ ] Add rank-0 guards to CheckpointHook in src/molix/hooks/checkpoint.py
- [ ] Add rank-0 guards to JournalHook in src/molix/hooks/journal.py
- [ ] Add rank-0 guards to TensorBoardHook in src/molix/hooks/tensorboard.py
- [ ] Update docstrings: Trainer (DDP-is-user-responsibility note), JournalWriter (callers-must-guard note), each new module per Google style with units
- [ ] Run full check + test suite

## Testing strategy

Happy path:

- Single-process fast path: `wrap_for_ddp(model)` with no `WORLD_SIZE` env var returns `(model, DistEnv(world_size=1, initialized=False))` and never calls `init_process_group`.
- `is_rank_zero()` / `get_rank()` / `get_world_size()` / `barrier()` work correctly when `dist` is uninitialized (return `True` / `0` / `1` / no-op).

Edge cases:

- `WORLD_SIZE=1` env var explicitly set → still single-process fast path (don't init NCCL for a 1-rank job).
- Double init: calling `wrap_for_ddp` twice in the same process must be guarded by `dist.is_initialized()` and not raise.
- `teardown_ddp(DistEnv(initialized=False))` → no-op, no exception.
- Hook IO methods called on a fake non-rank-0 process: monkeypatch `is_rank_zero` → `False`, assert no `torch.save` / `SummaryWriter.add_scalar` / `JournalWriter.append` is invoked.

Multi-rank smoke (gated `pytest.mark.multi_gpu`, requires N≥2 GPUs; skipped on CPU CI):

- Boot 2 ranks via `torch.multiprocessing.spawn`. On rank 0 vs rank 1 + reduction, assert MLP forward output on the same input batch (broadcast) is equal to single-rank reference within `atol=1e-5`.
- Repeat with `molpot.composition.Sonata(...).forward(..., compute_forces=True)` to exercise the autograd-inside-autograd path under DDP. Assert energies and forces match single-rank reference to `atol=1e-5`. This is the load-bearing test that gates the `sonata-05-driver` chain link.
- Boot 2 ranks with `CheckpointHook(checkpoint_dir=tmp_dir)`. Run 2 epochs. Assert exactly one `last.pt` exists at the end (not two, not corrupted) — confirms rank-0 guard works in a real distributed context.

Domain validation: not applicable for this sub-spec (pure engineering integration). Sonata's domain validation lives in `sonata-05-driver`'s diagnostic kernels.

## Out of scope

- 多节点（multi-node）DDP — `torchrun` 跨节点 rendezvous、网络选择、故障转移均不在本规范覆盖。`Trainer` 的 torchrun-elastic 文档注释保持不变；多节点验证留给后续规范。
- sbatch / SLURM 模板 — 用户直接调用 `torchrun --standalone --nproc_per_node=N benchmarks/bm_molpot/bm_sonata_hpc.py`；调度器集成不归本规范管。
- DeepSpeed / FSDP / ZeRO — 仅 vanilla `DistributedDataParallel`。
- 跨 rank 的梯度累积（gradient accumulation）超过 PyTorch DDP 默认行为 —— 不实现 `gradient_accumulation_steps` × `world_size` 的耦合调度；如果 step protocol 内部用了 micro-batch 累积，DDP reducer 的 `no_sync()` 上下文留给后续规范。
- 任何对 `TrainState` 名空间契约的改动（CLAUDE.md TrainState invariant 保持不变 —— 不引入 `state["dist"]` 子名空间，不添加 rank-aware 写入路径）。
- Eval 阶段的 metric all-reduce —— 各 rank 的 `state["eval"][...]` 仍是局部值；HPC 端到端正确性由 `sonata-05-driver` 的诊断核显式 all-reduce 完成。
- 真实数据集集成 —— bulk-water / charged-dimer Source 类由 `sonata-05-data` 提供；本规范的 smoke test 使用 toy fixture。
- `bm_sonata_hpc.py` 驱动本身 —— out-of-tree 用户脚本，**不进本仓库**；本规范只负责暴露 `wrap_for_ddp` / rank-0 守卫 / `DistEnv` 让外部驱动消费。
