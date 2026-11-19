# TensorRT-RTX 与 TensorRT-LLM：职责、执行路线与开源边界

调研日期：2026-11-19。本文是 NVIDIA 厂商补充研究，不新增 105 篇权威 Manifest 文档。

## 1. 证据基线与范围

| 项目 | 本次基线 | 取证方式 |
| --- | --- | --- |
| TensorRT-RTX | `cbffacf00c7d681bb31793e5413463e836af10d3`，1.7 Release | 本地只读源码与官方 1.7 在线文档 |
| TensorRT-LLM | `2672c80c97c4a8ef6274bac50f096850bb4f3356`，`version.py` 为 `1.4.0rc5` | GitHub API 按 SHA 读取官方仓库 README、架构、迁移及硬件文档；未克隆或运行 |

在线 TensorRT-LLM 文档页面标注生成自 `ab2dfeb`，与本次 main SHA 不同；已用固定 SHA 的仓库文档复核本文核心结论。这里的“当前”仅指上述基线，不能回溯套用到所有旧版或稳定版。

已执行 `python3 tools/materials.py search 'TensorRT'`，本次没有命中；以下结论直接采用官方文档与源码。

权威主题对应：[54：执行模式分类](../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-54-Eager、JIT、AoT、Trace与Replay.md)、[52：图捕获与 Shape](../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-52-CUDAGraph、PiecewiseCapture与Shape.md)。这些是概念归属，不表示既有 vLLM Claim 已覆盖两个 NVIDIA 产品。

## 2. 核心区别

TensorRT-RTX 的核心职责是将提交的网络编译为 Engine，并在终端 RTX GPU 上执行。TensorRT-LLM 的核心职责覆盖 LLM 请求的持续处理：调度、KV Cache、模型前向、采样及输出更新。两者的主要区别是职责范围和目标部署场景，而非同一个引擎简单分为 PC 版和服务器版。

| 维度 | TensorRT-RTX | TensorRT-LLM（上述当前基线） |
| --- | --- | --- |
| 官方定位 | 面向桌面、笔记本、工作站 RTX GPU 的模型优化器与运行时 | 面向大模型推理的框架、执行器与服务接口 |
| 模型入口 | ONNX、原生网络 API；另有 PyTorch 编译和 ORT/Windows ML 集成 | Hugging Face 模型 ID、本地模型及量化 checkpoint |
| 主执行路线 | AOT 生成可移植、可 JIT 的 Engine，终端 JIT 做硬件特化 | PyTorch 原生后端；CUDA Graph 等执行优化；已移除 TensorRT Engine 后端 |
| 主要管理对象 | 网络、Engine、ExecutionContext、输入输出 buffer、stream | 请求队列、每步 batch、KV Cache、ModelEngine、Sampler |
| 生成式 LLM | 原生 SDK 不直接提供完整 LLM 部署；通过 ORT GenAI/Windows ML 等集成 | 直接提供 `LLM.generate()` 等生成接口及在线服务工具 |
| 官方硬件示例 | RTX 产品线，具体模型/功能有各自支持约束 | A100、H100/H200、L20/L40、B200/GB200 等，亦列 DGX Spark |
| 公开源码边界 | GitHub 为配套样例/Demo，核心编译器和运行时通过 SDK 交付 | 公开推理框架与内核源码，仓库 Apache-2.0；依赖的 CUDA 等组件不因此成为全开源 |

不能据此把“LLM”解释成完全不支持终端设备，或把“RTX”解释成只能跑视觉模型。官方 TRT-RTX 文档明确提供经 ORT GenAI/Windows ML 执行 LLM 的路径；当前 TRT-LLM 硬件页列有 DGX Spark。具体 GeForce 型号是否受支持，应查对应版本与功能矩阵，不能由 CUDA 架构相同直接推定。

## 3. 两条端到端链路

```text
TensorRT-RTX 原生 SDK：
网络定义/ONNX → Builder AOT → 可分发 Engine
  → 目标 GPU JIT → ExecutionContext → GPU 前向 → 输出 tensor

TensorRT-LLM 的 LLM API：
模型 checkpoint → LLM → 每个 rank 的 PyExecutor
  → 取请求 → Scheduler → KVCacheManager → ModelEngine.forward
  → Sampler/输出处理 → 下一轮或完成
```

后者含逐步生成和多请求资源管理；前者的 Engine 前向本身不等于完整服务循环。RTX 应用也可在其外部实现生成循环，或将这部分交给 ORT GenAI 等上层组件。

与本工作区研究对象的职责对齐而言，TensorRT-LLM 更接近 vLLM 所覆盖的请求执行系统；TensorRT-RTX 更接近模型编译与执行 SDK。这是基于官方对象职责的比较，不表示它们互为替换或共享同一种内部实现。

## 4. 版本变化：不要沿用早期 TensorRT-LLM 架构

固定 SHA 的 `docs/source/legacy/tensorrt-backend-removal.md:L3-L18` 明确说明：

- PyTorch 已是唯一执行后端，`LLM(backend="tensorrt")` 会报错。
- `trtllm-build`、`trtllm-refit`、`trtllm-prune` 已移除，无单独 Engine 构建步骤。
- 模型 checkpoint 直接加载，`tensorrt` pip 依赖已移除。

因此，“TensorRT-RTX 用 JIT，而 TensorRT-LLM 总是先将模型离线编译成 TensorRT Engine”不适用于本次当前基线。旧教程中的 TensorRT Engine 路线应按其版本阅读；不能推导为当前 LLM 不使用任何编译或 CUDA Graph 优化。

**路线侧重与研究成熟度**：RTX 主交付路线为 AOT + 终端 JIT；当前 TRT-LLM 的 TensorRT Engine 路线已移除，主交付路线为 PyTorch 原生执行器及相应优化。两者本工作区证据成熟度均为 **★★★☆☆（官方契约/静态取证）**：未做 GPU 测试，不据此评价生产性能或官方产品成熟度。

## 5. 精准证据入口与工程边界

1. RTX `samples/helloWorld/python/hello_world.py:L129-L180`：Builder 序列化、Runtime 反序列化及创建执行上下文；`L184-L227`：I/O、stream 和执行调用。开源范围见根 `README.md:L8-L10`。本地基线见[整体架构文档](./tensorrt-rtx-architecture.md)。
2. TRT-LLM [`docs/source/developer-guide/overview.md:L15-L43`](https://github.com/NVIDIA/TensorRT-LLM/blob/2672c80c97c4a8ef6274bac50f096850bb4f3356/docs/source/developer-guide/overview.md#L15-L43)：`LLM`、`PyExecutor`、四个组件及逐轮处理；`L53-L82`：CUDA Graph、Overlap Scheduler。
3. TRT-LLM [`docs/source/legacy/tensorrt-backend-removal.md:L3-L36`](https://github.com/NVIDIA/TensorRT-LLM/blob/2672c80c97c4a8ef6274bac50f096850bb4f3356/docs/source/legacy/tensorrt-backend-removal.md#L3-L36)：当前后端和迁移边界。
4. TRT-LLM [`docs/source/supported-hardware.md:L3-L7`](https://github.com/NVIDIA/TensorRT-LLM/blob/2672c80c97c4a8ef6274bac50f096850bb4f3356/docs/source/supported-hardware.md#L3-L7)：官方硬件列表；[`docs/source/llm-api/index.md:L3-L8`](https://github.com/NVIDIA/TensorRT-LLM/blob/2672c80c97c4a8ef6274bac50f096850bb4f3356/docs/source/llm-api/index.md#L3-L8)：PyTorch 原生后端与多卡、多节点范围。
5. [RTX 官方架构](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/architecture/architecture-overview.html)：Two-Phase Compilation、Complementary Software、Relation to NVIDIA TensorRT；明确当前未集成 TensorRT-LLM，不应把 TRT-LLM 画成 RTX 的上层既有产品链路。
6. TRT-LLM [固定 SHA README](https://github.com/NVIDIA/TensorRT-LLM/blob/2672c80c97c4a8ef6274bac50f096850bb4f3356/README.md)、[LICENSE](https://github.com/NVIDIA/TensorRT-LLM/blob/2672c80c97c4a8ef6274bac50f096850bb4f3356/LICENSE)：公开项目与许可证范围。当前 README/架构还包含独立 Visual Generation 路径，故不将产品限定为“只支持文本 LLM”。

RTX 闭源优化器内部 IR、pass 和 JIT 代码生成实现为 **UNKNOWN**。本文未对两个项目作性能测评，也未验证相同模型的兼容性、跨 GPU 运行或多卡扩展收益。
