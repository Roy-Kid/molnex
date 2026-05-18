---
title: 移除 TensorDict 子类化，把 batch 还原为 plain nested TensorDict
status: code-complete
created: 2026-05-18
completed: 2026-05-18
---

# 移除 TensorDict 子类化，把 batch 还原为 plain nested TensorDict

## 摘要

`src/molix/data/types.py` 用 `AtomData / EdgeData / GraphData / GraphBatch`
四个 `TensorDict` 子类来给 post-collate batch 强制 schema，但实际并未
依赖任何子类特有行为——它们只是带 docstring 的 type tag。这层间接性
带来的代价：(1) 第三方依赖耦合（tensordict 子类化 API 历史上不稳定）、
(2) 新人 debug 时类型 narrow 迷惑、(3) 247 处类型名引用锁定了维护面、
(4) `torch.compile` 子类化图断裂风险。

本 spec 系统性清理：post-collate batch 改为 plain `tensordict.TensorDict`
（嵌套 `atoms / edges / graphs` 三个 namespace），删除四个子类；编码器
I/O 契约保持 `TensorDict → TensorDict`（in-place 写回 `node_features` 等）；
推荐继承 `TensorDictModuleBase` 拿 in_keys/out_keys 校验，但不强制；
atoms/edges/graphs 各自的 batch_size、必备字段、shape 契约迁移到
CLAUDE.md 文档强制。**不做向后兼容。**

## 设计

### 数据契约

```python
# Before
from molix.data.types import AtomData, EdgeData, GraphData, GraphBatch

batch = GraphBatch(
    atoms=AtomData({"Z": z, "pos": pos, "batch": b}, batch_size=[N]),
    edges=EdgeData({"edge_index": ei, ...}, batch_size=[E]),
    graphs=GraphData({"num_atoms": na, ...}, batch_size=[B]),
    batch_size=[],
)

# After
from tensordict import TensorDict

batch = TensorDict(
    {
        "atoms": TensorDict({"Z": z, "pos": pos, "batch": b}, batch_size=[N]),
        "edges": TensorDict({"edge_index": ei, ...}, batch_size=[E]),
        "graphs": TensorDict({"num_atoms": na, ...}, batch_size=[B]),
    },
    batch_size=[],
)
```

访问语义不变：`batch["atoms", "Z"]`、`batch["edges", "edge_index"]`、
`batch["graphs", "energy"]` 全部继续工作（plain TensorDict 原生支持
tuple-key）。

### 编码器契约

不变：`forward(td: TensorDict) -> TensorDict`，原位写回结果。

```python
# Recommended: TensorDictModuleBase gives in_keys/out_keys validation
class PiNet(TensorDictModuleBase):
    in_keys = [("atoms", "Z"), ("atoms", "pos"), ("edges", "edge_index"), ...]
    out_keys = [("atoms", "node_features"), ("edges", "edge_features")]

    def forward(self, td: TensorDict) -> TensorDict:
        ...
        td["atoms", "node_features"] = ...
        return td

# Allowed: plain nn.Module if you don't want in_keys validation
class MyEncoder(nn.Module):
    def forward(self, td: TensorDict) -> TensorDict:
        ...
        return td
```

### Schema 迁移到文档

`atoms / edges / graphs` 三个 namespace 各自的必备字段、shape、
`batch_size` 不再由 Python 类型强制。改为 CLAUDE.md "Two-tier data
contract" 节的 markdown 表格，作为开发者契约。

### 不做的事

- 不引入 `@tensorclass` 风格的 dataclass schema（用户明确反对额外类型层）
- 不替换 tensordict 本身（保留作为 dict-of-tensor 容器，享受 tuple-key 访问）
- 不动 `PackedCache` / 数据集 / SampleTask（这些已经是 flat dict，不受影响）
- 不动 C++ `interface/`（不依赖 tensordict）

## 需要创建或修改的文件

**删除（1）：**
- `src/molix/data/types.py` — 整文件删

**核心改造（5）：**
- `src/molix/data/__init__.py` — 去掉 4 个类的 import + `__all__`
- `src/molix/data/collate.py` — 构造换为 `TensorDict(..., batch_size=[N|E|B])`
- `src/molix/data/task.py` — 文档引用更新
- `src/molix/data/README.md` — 数据流描述更新
- `src/molix/profiler/mock.py` + `src/molix/profiler/dataloader.py` + `src/molix/profiler/__init__.py` — mock batch 改 plain TensorDict

**类型注解 / docstring 清理（~8 src 文件）：**
- `src/molzoo/{pinet, mace, allegro, sonata}.py`
- `src/molpot/composition/sonata.py`、`src/molpot/heads/{edge, electrostatics, multipole}.py`
- `src/molix/core/losses/molecular.py`、`src/molix/core/losses/__init__.py`、`src/molix/core/steps/__init__.py`
- `src/molix/profiler/module.py`

**测试 / 基准（~19 文件）：**
- `tests/symmetry_helpers.py`、`tests/test_molix/test_data/*.py`、`tests/test_molpot/test_composition/*.py`、`tests/test_molpot/test_heads/*.py`、`tests/test_molzoo/*.py`
- `benchmarks/conftest.py`、`benchmarks/molpot/sonata.py`、`benchmarks/molzoo/{bm_allegro, bm_pinet}.py`

**文档 + notes（2）：**
- `CLAUDE.md` — "Two-tier data contract" 节改写
- `.claude/notes/notes.md` — 架构规则

## 任务

- [x] **T1 — 删除 `src/molix/data/types.py`**
- [x] **T2 — 改造 `src/molix/data/collate.py`**
- [x] **T3 — Profiler 模块（mock / dataloader / __init__）**
- [x] **T4 — molzoo 编码器（pinet / mace / allegro / sonata）**
- [x] **T5 — molpot composition + heads**
- [x] **T6 — molix core (losses / steps)**
- [x] **T7 — 测试套件**
- [x] **T8 — 基准（benchmarks/）**
- [x] **T9 — CLAUDE.md "Two-tier data contract" 节**
- [x] **T10 — `src/molix/data/README.md` + `src/molix/data/task.py` 文档**
- [x] **T11 — `.claude/notes/notes.md`**
- [x] **T12 — 全量验收**：`src/` 零命中；2091 passed, 8 skipped, 17 xfailed

## 当前范围外

- 不替换 tensordict（仍作 dict-of-tensor 容器使用）
- 不引入 dataclass / TypedDict / Pydantic schema 来取代删除的子类
- 不动 `PackedCache` 或缓存层
- 不动 C++ `interface/` 或 AOT 导出
- 不优化 collate 性能或引入 `torch.compile`
- 不重命名 `node_features` / `edge_features` 等约定 key
