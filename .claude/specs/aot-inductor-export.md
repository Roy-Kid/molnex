---
title: AOT Inductor 模型导出
status: approved
created: 2026-05-14
---

# AOT Inductor 模型导出

## 摘要

在 `molix` 中增加纯 Python 的 `export_model()` 函数，将训练好的 `nn.Module` 模型通过 `torch._inductor.aot_compile()` 导出为 `.so` 文件，同时保存权重（`.pt`）和元数据（`.meta.json`）。对用户完全隐藏 AOT Inductor 的内部细节——用户只需传入模型和样例输入，得到一个可直接部署的导出目录。

## 设计

### 入口

```python
from molix import export_model

export_model(model, example_inputs, "path/to/export_dir")
# → export_dir/
#   ├── model.so
#   ├── model.pt
#   └── model.meta.json
```

### 职责

- 将模型 `.eval()` 并移动到目标设备
- 调用 `torch._inductor.aot_compile()` 生成 `.so`
- `torch.save(model.state_dict())` 保存权重
- 写入 `meta.json`：设备类型、输入 shape/dtype、模型类名
- 不暴露 `aot_compile` 的任何参数（`output_path`、`options` 等由内部处理）
- `device="auto"` 时自动检测 CUDA 可用性

### 设备自动检测

- `"auto"` → `torch.cuda.is_available()` 决定 `"cuda"` / `"cpu"`
- 显式 `"cuda"` 或 `"cpu"` → 直接使用

## 需要创建或修改的文件

- `src/molix/export.py` (new)
- `src/molix/__init__.py` (modify — add `export_model` to `__all__`)
- `tests/test_molix/test_export.py` (new)

## 任务

- [ ] Write failing tests in `tests/test_molix/test_export.py`
- [ ] Implement `export_model()` in `src/molix/export.py`
- [ ] Add `export_model` to `src/molix/__init__.py` exports
- [ ] Run test suite

## 测试策略

- 正向路径：导出小型全连接网络 → `.so`、`.pt`、`.meta.json` 存在
- `.so` 可被 `torch._inductor.aoti_load()` 加载
- CPU / CUDA 分别测试
- `device="auto"` 在 CUDA 环境下写入 `"device": "cuda"`
- 错误路径：不存在的导出目录、非 nn.Module 输入

## 当前范围外

- 动态形状
- 模型量化/剪枝
- C++ 侧的加载和推理（独立需求，见 `interface-aot-runner`）
