# 大模型推理软硬件栈与大厂方案调研

> **资料角色声明**：
> 本目录归档早期通用架构设计、运行时接口分析（Modular AsyncRT、TT-Runtime 等）与历史对比。
> **权威文档关系**：本项目唯一权威结论集合为 [`../20260903-vllm-ai-infra-research/`](../20260903-vllm-ai-infra-research/) 目录下的 105 篇文档。本目录文档保留用于底层运行时技术参考与历史追溯；若与 105 篇权威集或固定源码产生冲突，必须以固定源码 SHA 及权威文档结论为准。

## 核心调研主线

1. **AOT 静态编译路线**：
   - 静态图优化、分桶机制（Shape Bucketing）、StableHLO/MLIR 编译、离线常量与静态内存排布、Runtime 产物加载与执行。
   - 重点参考：Qualcomm QAIC (QEfficient/QAIC Compiler)、Google TPU (XLA/PJRT)、Tenstorrent (TT-MLIR)。权威对应参见 `20260903-vllm-ai-infra-research/` 对应章节（11-qaic、12-tenstorrent、13-tpu）。
2. **解析执行与动态 Capture 路线**：
   - PyTorch Eager 派发、PrivateUse1 设备抽象、Graph Capture/Replay（CUDA Graph / CANN Graph）、PagedAttention 自定义算子定制与异步任务队列。
   - 重点参考：Huawei Ascend (Torch_NPU + CANN + TorchAir)、PyTorch 原生调度。权威对应参见 `20260903-vllm-ai-infra-research/` 对应章节（04-vllm-execution、10-ascend）。

## 目录结构与阅读建议

1. `02-vendor-research`：各厂商端到端全栈与专项技术调研（Ascend、TPU/XLA、高通 Cloud AI 等）。
2. `03-runtime-interfaces`：底层运行时接口抽象研究（Modular AsyncRT、Tenstorrent Runtime、CANN Runtime 等）。

## 专题研究文档索引与权威对应

| 本地文档 | 标题 | 核心方向 | 对应权威 Owner (105 篇) |
| --- | --- | --- | --- |
| [02-vendor-research/20-vllm-torchtpu-xla.md](./02-vendor-research/20-vllm-torchtpu-xla.md) | TPU/XLA 路线与 vLLM 适配 | AOT 静态编译与 SPMD 分布式 | `13-tpu/` (20260903-130 ~ 135) |
| [02-vendor-research/21-ascend-stack.md](./02-vendor-research/21-ascend-stack.md) | 华为昇腾全栈架构 | Eager/Capture 混合与 CANN 驱动 | `10-ascend/20260903-100, 101` |
| [02-vendor-research/22-pytorch-eager-graph-inductor-memory.md](./02-vendor-research/22-pytorch-eager-graph-inductor-memory.md) | PyTorch 内存模型与执行对比 | Eager vs Graph Capture vs Inductor | `04-vllm-execution/`, `05-vllm-compile/` |
| [02-vendor-research/23-vllm-ascend-features-and-code-walkthrough.md](./02-vendor-research/23-vllm-ascend-features-and-code-walkthrough.md) | vLLM-Ascend 插件源码剖析 | ModelRunner、Attention 算子与 Capture | `10-ascend/20260903-102, 103` |
| [02-vendor-research/24-torch-npu-architecture-and-runtime-deep-dive.md](./02-vendor-research/24-torch-npu-architecture-and-runtime-deep-dive.md) | torch_npu 架构与运行时深入解析 | PrivateUse1 设备接入与 C++ 调度 | `10-ascend/20260903-101` |
| [02-vendor-research/25-ascend-capture-and-replay-deep-dive.md](./02-vendor-research/25-ascend-capture-and-replay-deep-dive.md) | 昇腾图捕获与重放技术详解 | Graph Capture、内存锁与零拷贝 | `10-ascend/20260903-104` |
| [02-vendor-research/26-torch-npu-stream-event-taskqueue.md](./02-vendor-research/26-torch-npu-stream-event-taskqueue.md) | torch_npu 双重异步与任务队列 | 异步流、Event 同步与流水线设计 | `10-ascend/20260903-101, 103` |
| [02-vendor-research/27-qualcomm-cloud-ai-stack.md](./02-vendor-research/27-qualcomm-cloud-ai-stack.md) | 高通 Cloud AI 全栈技术架构 | AOT 编译器与 Runtime 架构 | `11-qaic/20260903-110, 111` |
| [02-vendor-research/28-vllm-qaic-backend.md](./02-vendor-research/28-vllm-qaic-backend.md) | 高通 vLLM 后端适配深度分析 | 插件注册、AoT 与 PYT 双模式对比 | `11-qaic/20260903-112, 114` |
| [02-vendor-research/29-torch-qaic-backend.md](./02-vendor-research/29-torch-qaic-backend.md) | 高通 PyTorch 设备后端解析 | 设备扩展、内存/同步与算子接入 | `11-qaic/20260903-111, 113` |
| [02-vendor-research/31-cross-vendor-collective-communication-deep-research.md](./02-vendor-research/31-cross-vendor-collective-communication-deep-research.md) | 跨厂商集合通信算子底层实现与优化对比 | 集合通信底层原理、MC2、CCL与算法优化 | `14-decisions/20260903-140, 141` |
| [02-vendor-research/32-collective-communication-kernel-optimization-handbook.md](./02-vendor-research/32-collective-communication-kernel-optimization-handbook.md) | 集合通信算子内核自研与性能优化实战手册 | 数学推导、状态机流转与自研架构法则 | `14-decisions/20260903-140, 141` |
| [02-vendor-research/33-moe-collective-communication-and-eplb-handbook.md](./02-vendor-research/33-moe-collective-communication-and-eplb-handbook.md) | MoE 专家并行 AllToAll 与动态负载均衡优化指南 | EPLB 调度、DualPipe 流水与稀疏通信 | `14-decisions/20260903-140, 141` |
| [02-vendor-research/34-context-parallel-collective-communication-handbook.md](./02-vendor-research/34-context-parallel-collective-communication-handbook.md) | 长文本上下文并行 (CP) 集合通信与算子实战指南 | Ulysses AllToAll、RingAttention 与 PCP/DCP | `06-vllm-distributed/20260903-65` |
| [02-vendor-research/35-pd-disaggregated-kv-transfer-handbook.md](./02-vendor-research/35-pd-disaggregated-kv-transfer-handbook.md) | 大模型 P/D 分离部署与跨节点 KV 传输实战手册 | Mooncake RDMA、QAIC DMA 与逐层流水 | `08-production/20260903-82` |
| [03-runtime-interfaces/30-reference-interfaces.md](./03-runtime-interfaces/30-reference-interfaces.md) | 运行时抽象接口对比 | 跨平台运行时契约与设计 | `14-decisions/20260903-143` |
| [03-runtime-interfaces/31-modular-device-context.md](./03-runtime-interfaces/31-modular-device-context.md) | Modular device_context 研究 | 异构硬件抽象模型 | 接口参考扩展 |
| [03-runtime-interfaces/32-modular-asyncrt-c-abi.md](./03-runtime-interfaces/32-modular-asyncrt-c-abi.md) | Modular AsyncRT C ABI 分析 | 异步 C ABI 运行时接口 | 接口参考扩展 |
| [03-runtime-interfaces/33-tt-mlir-runtime.md](./03-runtime-interfaces/33-tt-mlir-runtime.md) | Tenstorrent TT-MLIR Runtime | MLIR 编译产物运行时 | `12-tenstorrent/20260903-122` |
| [03-runtime-interfaces/34-tt-runtime-backends.md](./03-runtime-interfaces/34-tt-runtime-backends.md) | TT-Runtime 对接后端分析 | Metalium 与底层驱动分发 | `12-tenstorrent/20260903-122, 125` |
| [03-runtime-interfaces/35-ttmetal-naming.md](./03-runtime-interfaces/35-ttmetal-naming.md) | TT-Metal 核心概念与命名解析 | 计算核心、Buffer 与队列 | `12-tenstorrent/20260903-120, 122` |
