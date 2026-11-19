# TensorRT-RTX：定位、两阶段编译与开源边界

- 调研日期：2026-11-19（Asia/Shanghai）。
- 开源仓库基线：`NVIDIA/TensorRT-RTX`，commit `cbffacf00c7d681bb31793e5413463e836af10d3`。
- SDK 文档基线：当日读取的官方 `latest` 架构、安装与许可页面；这些是可变网页，不等同于固定源码。
- 研究范围：RTX PC 推理优化器/运行时的公开契约与源码交付范围；未安装 SDK、未做设备测试。
- 权威主题对应：[54：Eager、JIT、AoT、Trace 与 Replay](../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-54-Eager、JIT、AoT、Trace与Replay.md)。本文是补充资料，不新增权威 Manifest 文档或改写固定 vLLM Claim。

## 1. 定位与官方路线

TensorRT for RTX 是面向桌面、笔记本与工作站 RTX GPU 的推理优化器和运行时。官方主交付路径是 **AOT 优化 + 终端 GPU JIT 专门化**：预先生成可分发引擎，到用户设备后再编译为适合该 GPU 的执行计划。它解决消费端不同 RTX GPU 的模型部署问题。

路线成熟度研究评级 **★★★☆☆（已发布 SDK 与公开 API 契约；本次未设备验证）**，依据是官方安装包、运行时 API 与示例均已交付；这不是生产性能评级。Eager/Capture 是其他层面的执行机制，不能用是否支持 CUDA Graph 来替代对两阶段编译路线的描述。

官方 [Architecture Overview](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/architecture/architecture-overview.html) 的 `Two-Phase Compilation`、`ONNX` 与 `Relation to NVIDIA TensorRT` 是本节证据入口。可从 ONNX、原生网络定义 API、ONNX Runtime/Windows ML 接入；特定集成与硬件支持应另查当前支持矩阵。

## 2. 最小端到端链路

```text
训练好的模型 -> ONNX / 网络定义 API
            -> AOT builder -> 可分发的 JIT-able engine
            -> 终端 RTX GPU 上的 runtime/JIT -> GPU 专用执行计划
            -> 推理；可用 runtime cache 保留编译结果
```

AOT 可以在 CPU 上构建；终端 JIT 依实际 GPU 专门化。可移植性受支持架构、构建配置和 TensorRT-RTX 版本约束，不表示引擎可以无条件跨任意平台或版本使用。完整编译器 IR、pass、kernel 选择内部实现未从公开仓库核查到，记为 `UNKNOWN`，不据官方功能描述推演内部算法。

TensorRT-RTX 与 TensorRT-LLM 是不同产品。官方架构文档说明，LLM 可通过 Windows ML/ONNX Runtime GenAI 的 execution provider 路径集成；不能将它视为包含完整 LLM 调度与服务栈的 TensorRT-LLM 替代品。

## 3. 是否开源：按组件区分

| 部分 | 当日核查结果 | 证据 |
| --- | --- | --- |
| GitHub 中的 samples、demo 等配套组件 | 公开源码；仓库 LICENSE 为 Apache-2.0 | 固定 commit 的 README、LICENSE、samples/README.md、demo/README.md |
| SDK 核心库、ONNX parser 库、运行时/JIT 实现 | 安装文档以 DLL/共享库方式交付；该公开仓库不提供完整核心实现源码 | 官方安装文档；samples 构建仍依赖外部 SDK 的 include/lib |
| SDK 使用与分发条款 | 使用 TensorRT-RTX 自身的软件许可，不应将仓库 Apache-2.0 泛化至整个 SDK | 官方 SLA 第 1、2、6 节 |

结论是 **部分开源，核心 SDK 未公开完整源码**。可以阅读、修改配套示例，调用已交付 API；仅克隆这个仓库不能从源码构建完整 TensorRT-RTX 编译器与运行时。这里没有声称所有 SDK 依赖都闭源，也不对独立的 ONNX Runtime EP 或其他集成仓库作全量许可审计。

## 4. 固定源码行号与官方证据

- `NVIDIA/TensorRT-RTX/README.md:L8-L10`：SDK 另行下载；仓库包含配套开源组件。[固定入口](https://github.com/NVIDIA/TensorRT-RTX/blob/cbffacf00c7d681bb31793e5413463e836af10d3/README.md#L8-L10)。
- `NVIDIA/TensorRT-RTX/samples/CMakeLists.txt:L19-L27`：`TRTRTX_INSTALL_DIR` 必填，并从 SDK 目录读取 include/lib。[固定入口](https://github.com/NVIDIA/TensorRT-RTX/blob/cbffacf00c7d681bb31793e5413463e836af10d3/samples/CMakeLists.txt#L19-L27)。
- [固定仓库 LICENSE](https://github.com/NVIDIA/TensorRT-RTX/blob/cbffacf00c7d681bb31793e5413463e836af10d3/LICENSE)：Apache License 2.0；适用对象是对应组件。
- [Installing TensorRT-RTX](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/installing.html)：Windows SDK 内容列出核心 DLL、parser DLL、headers、samples 与 Python bindings；PyPI 安装提供 bindings 与共享库。
- [SDK 软件许可](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html)：二进制软件与另附许可组件分别适用对应条款。本记录仅核对交付和开源边界，不提供个案法律意见。

源码配套组件的目录与构建依赖核查为 `SOURCE_IMPLEMENTED × STATIC_REVIEWED`；编译路线与支持边界为官方文档契约，未升级为设备实测证明。

## 5. 本地克隆后的进一步阅读

2026-11-19 已将同一 SHA 克隆至 `nvidia/third_party/TensorRT-RTX/`。端到端 API 调用、动态形状、缓存、内存及真实模型管线，见 [NVIDIA 导航](../../nvidia/docs/README.md) 与 [TensorRT-RTX 整体架构解读](../../nvidia/docs/tensorrt-rtx-architecture.md)。
