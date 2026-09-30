# 高通 Cloud AI (QAIC) 调研现状与知识库

> **资料角色声明**：
> 本目录归档高通 Cloud AI 100 / Ultra 服务端 LLM 推理软件栈的技术调研。
> **权威文档关系**：高通 QAIC 路线的唯一权威结论集合为 [`../../20260903-vllm-ai-infra-research/11-qaic/`](../../20260903-vllm-ai-infra-research/11-qaic/)（文档编号 `20260903-110` ~ `116`）。
> **固定源码 Revision**：`qualcomm/third_party/vllm-qaic/` 严格锁定为 `3212cc670130b7e5290b429f781dc978d7ecf430`。

高通方案在业内非常典型：**在同一个平台入口（`vllm-qaic`）下，并存着 AOT 静态编译与 PyTorch Eager/解析执行 两条截然不同的执行路线**。

---

## 官方路线侧重与侧重程度说明

- **AOT 静态编译路线（绝对主导 / 工业成熟路径，侧重程度：★★★★★）**：
  - **定位与实际使用**：高通 Cloud AI 100/Ultra 在生产环境与官方基准测试（如 MLPerf）中的**绝对标准路径**。官方 SDK、编译链（`QEfficient` 导出 ONNX → QAIC 编译器 `qaic-compile`/`qaic-exec` 生成 QPC 二进制包）以及官方评测工具（`qaic-runner`）均围绕 AOT 路线构建。
  - **源码默认行为**：在 `vllm-qaic` 中，平台默认分发行为是只要环境中未安装闭源的 `torch_qaic` 动态包（`platform_base.py:65`：`is_aot = not _torch_qaic_installed`），系统**无缝 Fallback 并强制走 `QaicWorkerAoT` 与 `QaicModelRunnerAoT` 路径**。该模式绕过 PyTorch 调度开销，执行确定性与硬件利用率最高。
- **PyTorch / PYT Eager 路线（演进探索 / 受限受众路径，侧重程度：★★☆☆☆）**：
  - **定位与实际使用**：高通为了迎合 PyTorch 动态图与 PagedAttention 生态而新增的次要路线，目前仍处于向生态对齐的演进期。
  - **受限因素**：其底层核心依赖闭源分发的 `torch-qaic` wheel（通过 QAIC Apps SDK 的 `--install-torch-qaic` 安装到 `/opt/qti-aic/integrations/torch_qaic/`），且 C++ Dispatcher/Allocator 均闭源；在 vLLM 插件中该路线明确关闭了部分分布式特性、投机解码（SpD）与 Disaggregated Serving（`platform_base.py:282-290`），工业公开落地案例远少于标准 AoT。

---

## 路线子目录说明与权威对应

- **`aot/`**：**AOT 静态编译路线**
  - **核心链路**：PyTorch / Transformers → `QEfficient` 模型转换/优化与 ONNX 导出 → QAIC 编译器（`qaic-compile` / `qaic-exec`）生成 QPC 二进制包 → Runtime（`qaicrt`）加载并在设备执行。
  - **调度与内存**：采用多档位 Shape Bucketing 与 CCL (Compute Context Length) 动态上下文长度；静态分配固定缓冲区；绕过 PyTorch 算子下发。
  - **专题文档**：[`aot/30-qaic-aot-deep-research.md`](./aot/30-qaic-aot-deep-research.md)（AOT 静态编译全链路与槽位机制深度调研）。
  - **对应权威 Owner**：`20260903-114` (QAIC-AoT-QPC-Artifact 与 Profile)、`20260903-115` (QAIC 调度与 Batch)。
- **`capture-eager/`**：**解析执行与动态图（PYT）路线**
  - **核心链路**：PyTorch 保持模型定义 → `torch-qaic` 提供 `qaic` 设备后端（PrivateUse1 / C++ Dispatcher）与设备张量 → 逐算子派发（通过 `register_qaic_customop()` 注册定制算子，Decode 注意力目前主要通过 CPU SDPA/PagedAttention 混合回退）或图捕获/JIT 运行时执行。
  - **调度与内存**：支持标准 PagedAttention 与动态 KV 映射。
  - **对应权威 Owner**：`20260903-113` (QAIC-Eager 执行、KV 与特性边界)。
- **`comm/`**：**通用基础设施与公共契约**
  - 覆盖 Cloud AI 100/Ultra 硬件规格（NSP 算力核、片上 SRAM/DDR、PCIe 拓扑）、`vllm-qaic` 平台初始化与两路线切换机制（`platform_base.py` 根据 `torch_qaic` 包存在与否决定）、QAIC SDK 驱动与版本兼容矩阵、QCCL 分布式通信后端（`platform_base.py:67`：`dist_backend = "qccl"`）。
  - **对应权威 Owner**：`20260903-110` (硬件与通信)、`20260903-111` (SDK 与软件栈)、`20260903-112` (Platform 与 ModeSelection)、`20260903-116` (版本 Tuple 与验证)。

---

## 调研现状：已证实事实 vs 显式空缺

为确保技术分析不建立在错误或臆测的基础上，特此明确当前调研的证据边界：

### 1. 已证实事实（基于开源源码与 SDK 公开文档）
- [x] **平台路线判定机制**：`vllm-qaic` 的 `platform_base.py:64-68` 并不单纯依据 `--enforce-eager`，而是优先根据环境中是否导入了 `torch_qaic` 模块（`is_aot = not _torch_qaic_installed`）来选择 `QaicWorkerAoT` 还是 `QaicWorkerPyt`，且在 `platform_base.py:201-215` 中会自动覆写 `enforce_eager` 标记（权威证据：`20260903-112`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。
- [x] **AoT 运行生命周期**：已完整追踪 `vllm-qaic/vllm_qaic/worker/worker.py:632`（`QaicWorkerAoT`）与 `model_runner.py:389`（`QaicModelRunnerAoT`）；证实其输入张量经由 CPU/NumPy 组织并直接调用 `qaicrt.Context` / `Program` / `Queue`（`qaic_session_np.py:109-155`）下发执行（权威证据：`20260903-114`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。
- [x] **QPC 产物契约**：证实 QPC 是编译后的静态二进制包，内含静态 shape 限制及绑定的固定 buffer 描述符，通过 `qaicrt.Qpc.getIoDescriptor()` 反序列化校验输入输出绑定位点（权威证据：`20260903-114`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。
- [x] **PYT 模式特性门禁限制**：在 `platform_base.py:282-297` 中，PYT 模式被显式禁止使用 Disaggregated Serving（`assert kv_transfer_config is None`）、禁止投机解码 SpD（`raise ValueError`）并强制关闭 `async_scheduling`。
- [x] **分布式后端通信**：`platform_base.py:67` 明确声明分布式通信后端为 `dist_backend: str = "qccl"`，通信封装层 `QAicCommunicator` 继承自 `DeviceCommunicatorBase`，在 `worker.py:299-305` 中调用 `init_worker_distributed_environment` 初始化。

### 2. 显式空缺与未决问题（待调研项 / UNKNOWN）
- [ ] **【空缺 1】`torch-qaic` 内部 C++ / Dispatcher 实现缺失**：目前 `torch-qaic` 仅有二进制 wheel 包与外部 Python 接口调用证据，其 C++ 层的算子注册表、私有 Allocator 算法与 Stream/Event 实现源码闭源，底层如何将 PyTorch 算子映射至设备尚缺源码级证据（标记为 `UNKNOWN`，证据等级：`VENDOR_CLAIM × UNVALIDATED`）。
- [ ] **【空缺 2】PYT 路线底层是否存在 Graph Capture / JIT 融合**：官方文档提及 JIT runtime，但缺乏底层是将每个 ATen 算子独立启动（Eager），还是在底层默默执行类似 CUDA Graph / 子图编译重放的直接证据（标记为 `UNKNOWN`）。
- [ ] **【空缺 3】NSP 之间的通信原语实现**：多卡/多芯片间的分布式并行（如 Tensor Parallelism）在底层走 PCIe 还是专有 CCL 库，具体的 Collective 通信实现细节待核实（标记为 `UNKNOWN`）。
- [ ] **【空缺 4】真实物理卡实测指标**：在未实际挂载 Cloud AI 100/Ultra 硬件卡的环境下，不得宣称具备 `DEVICE_SMOKE`、`DEVICE_NUMERIC` 或压测指标。
