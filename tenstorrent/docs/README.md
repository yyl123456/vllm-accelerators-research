# Tenstorrent 软件栈调研现状与知识库

> **资料角色声明**：
> 本目录归档 Tenstorrent（Wormhole / Blackhole 架构）在 vLLM 服务端推理软件栈中的技术调研。
> **权威文档关系**：Tenstorrent 路线的唯一权威结论集合为 [`../../20260903-vllm-ai-infra-research/12-tenstorrent/`](../../20260903-vllm-ai-infra-research/12-tenstorrent/)（文档编号 `20260903-120` ~ `126`）。
> **固定源码 Revision**：
> - `tenstorrent/third_party/vllm-tt-plugin/` 严格锁定为 `f6995475739201fd1d2883adec477ecb37f951d9`
> - `tenstorrent/third_party/tt-metal/` 严格锁定为 `c634b1ca4c10eea5037d80767ce5f9e2912a401f`

Tenstorrent 方案兼具 **基于 MLIR/XLA 的静态编译探索** 与 **基于 TT-Metal / Command Queue 的自定义调度执行**。

---

## 路线子目录说明与权威对应

- **`aot/`**：**TT-MLIR / TT-XLA 编译路线**
  - **核心链路**：JAX/PyTorch-XLA/ONNX → `tt-xla` / PJRT → `tt-mlir` 编译器（TTIR / TTMetal Dialect 优化与 Lowering）→ 生成 Flatbuffer 执行格式二进制 → Runtime 加载执行。
  - **对应权威 Owner**：`20260903-122` (TTMetal-TTNN 与 ModelGenerator)。
- **`capture-eager/`**：**vLLM TT-Plugin 与原生执行路线**
  - **核心链路**：`vllm-tt-plugin` 平台插件 → 自定义 Engine 与 Scheduler → 调用 TTNN 算子与 TT-Metal Command Queue 下发任务。
  - **关键机制**：多 Lane 调度、Device 端 Sampling、异步 Decode、基于 SRAM/DRAM 分层内存的 Buffer 管理。
  - **对应权威 Owner**：`20260903-123` (vLLM-TT-Plugin 与自定义 Engine)、`20260903-124` (TT 调度与 Prefill-Decode)、`20260903-125` (TT-Trace 与 AsyncDecode)。
- **`comm/`**：**硬件架构与通用驱动**
  - 覆盖 Tensix 处理器核（Baby RISC-V + 矩阵/向量引擎）、片上 L1 SRAM 与 2D Torus NoC 片上网络、PCIe 拓扑与 `tt-umd` (用户态驱动) / `tt-kmd` (内核态驱动) 交互。
  - **对应权威 Owner**：`20260903-120` (Tensix、Tile 与 SRAM)、`20260903-121` (Wormhole、T3K 与 Mesh 拓扑)、`20260903-126` (DeviceGroup 与验证边界)。

---

## 调研现状：已证实事实 vs 显式空缺

### 1. 已证实事实（基于真实源码与静态审阅）
- [x] **芯片与网络架构**：片上 Tensix 核通过 2D NoC 互联，显式暴露 L1 SRAM 给编程栈，通过 TT-Metal 的 Buffer/Kernel 进行直接调度（权威证据：`20260903-120`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。
- [x] **vLLM-TT 插件架构**：独立插件 `tenstorrent/third_party/vllm-tt-plugin` 实现了定制化的 Runner、多 Lane 分流与异步 Decode 特性（权威证据：`20260903-123`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。

### 2. 显式空缺与未决问题（待调研项 / UNKNOWN）
- [ ] **【空缺 1】TT-MLIR 端到端 Serving 成熟度**：目前生产级 vLLM 插件仍主要基于手工调优的 TTNN 算子库与 TT-Metal，基于 TT-MLIR 生成的 Flatbuffer 二进制在端到端 LLM Serving 中的性能表现与动态分桶集成尚未完全打通（标记为 `UNKNOWN`，证据等级：`VENDOR_CLAIM × UNVALIDATED`）。
- [ ] **【空缺 2】跨芯片大规模 Mesh 集合通信性能**：Blackhole 多卡直连网络在面对超大模型 Tensor Parallelism 时的拥塞控制与通信延迟边界尚待实测数据验证（标记为 `UNKNOWN`）。
- [ ] **【空缺 3】真实 Wormhole/Blackhole 硬件实测**：未访问物理加速卡，不得宣称具备 `DEVICE_SMOKE`、`DEVICE_NUMERIC` 或真实吞吐数据。
