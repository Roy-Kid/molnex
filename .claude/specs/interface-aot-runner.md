---
title: C++ AOT 模型运行时库
status: approved
created: 2026-05-14
---

# C++ AOT 模型运行时库

## 摘要

在顶层新建 `interface/` 目录（与 `src/` 并列），提供纯 C++ 的 `ModelRunner` 库供外部 C++ 项目（如 LAMMPS 插件）链接使用。该库封装 `AOTIModelContainerRunnerCuda` / `AOTIModelContainerRunnerCpu`，支持加载已导出的 `.so` 模型进行推理和双 buffer 零停机权重热更新。不依赖 Python，不在 Python wheel 中。

## 设计

### 目录结构

```
interface/                              # 顶层，与 src/ 并列
├── CMakeLists.txt
├── include/molnex/interface/
│   └── model_runner.h                  # 公开头文件
├── src/
│   └── model_runner.cpp                # 实现
└── tests/
    ├── CMakeLists.txt
    └── test_model_runner.cpp           # C++ 单元测试
```

### 公开 API

```cpp
namespace molnex::interface {

class ModelRunner {
public:
    // 从导出目录加载模型。从 meta.json 读取设备类型，自动选择 CudaRunner 或 CpuRunner。
    // num_models >= 2 时启用双 buffer 热更新。
    explicit ModelRunner(const std::string& model_dir, int num_models = 2);

    // 同步推理
    std::vector<torch::Tensor> run(const std::vector<torch::Tensor>& inputs);

    // 异步推理
    std::vector<torch::Tensor> run_async(
        const std::vector<torch::Tensor>& inputs,
        c10::cuda::CUDAStream stream
    );

    // 从 .pt 文件加载新权重，双 buffer 零停机切换
    void update_weights(const std::string& weight_path);

    // 直接用 tensor map 更新权重
    void update_weights(
        const std::unordered_map<std::string, torch::Tensor>& params
    );

    // 查询模型参数名和类型
    std::vector<std::pair<std::string, std::string>> parameter_info() const;
};

}  // namespace molnex::interface
```

### 双 Buffer 热更新

1. `num_models=2` 启用双 buffer（active + inactive）
2. `update_weights()` → `update_inactive_constant_buffer(params)` → `swap_constant_buffer()`
3. 推理线程全程无锁、不中断

### Build

- `libmolnex_interface` 静态库，依赖 `find_package(Torch REQUIRED)`
- 顶层 `CMakeLists.txt` 增加 `add_subdirectory(interface)`
- 不在 `pyproject.toml` 的 `wheel.packages` 中
- C++ 标准：C++17

### 元数据协议

加载时从 `<model_dir>/<name>.meta.json` 读取设备类型，匹配 runner 类型。字段：
- `device`: `"cuda"` | `"cpu"`
- `inputs`: `[{"shape": [...], "dtype": "..."}]`
- `model_class`: 模型类名（信息用）

## 需要创建或修改的文件

- `interface/CMakeLists.txt` (new)
- `interface/include/molnex/interface/model_runner.h` (new)
- `interface/src/model_runner.cpp` (new)
- `interface/tests/CMakeLists.txt` (new)
- `interface/tests/test_model_runner.cpp` (new)
- `CMakeLists.txt` (modify — add `add_subdirectory(interface)`)

## 任务

- [ ] Write C++ `ModelRunner` header `interface/include/molnex/interface/model_runner.h`
- [ ] Write C++ `ModelRunner` implementation `interface/src/model_runner.cpp`
- [ ] Write C++ tests `interface/tests/test_model_runner.cpp` + CMakeLists.txt
- [ ] Set up build: `interface/CMakeLists.txt` + update top-level `CMakeLists.txt`
- [ ] Build and run C++ test suite

## 测试策略

### C++ 测试 (`interface/tests/test_model_runner.cpp`)

- 加载 Python `export_model()` 导出的 `.so` → `run()` 输出与参考值一致（atol=1e-5）
- CPU runner 正确加载 CPU 导出的模型
- CUDA runner 正确加载 CUDA 导出的模型（GPU 可用时）
- `update_weights(path)` 后推理结果反映新权重
- `update_weights(map)` 直接传 tensor map 等价于从文件加载
- 并发：10× `update_weights()` 与 100× `run()` 交叉执行无崩溃
- 错误路径：损坏的 `.so`、不匹配的设备类型

## 当前范围外

- 动态形状
- 多 GPU 推理
- pybind11 Python 绑定（纯 C++ 库）
- Python 侧导出工具（独立需求，见 `aot-inductor-export`）
