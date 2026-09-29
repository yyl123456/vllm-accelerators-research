# Huawei Ascend 软件栈调研现状与知识库

> **资料角色声明**：
> 本目录归档华为昇腾（Ascend NPU）在 vLLM 服务端推理软件栈中的技术调研与专项分析。
> **权威文档关系**：华为昇腾路线的唯一权威结论集合为 [`../../20260903-vllm-ai-infra-research/10-ascend/`](../../20260903-vllm-ai-infra-research/10-ascend/)（文档编号 `20260903-100` ~ `105`）。
> **固定源码 Revision**：`huawei/third_party/vllm-ascend/` 严格锁定为 `3546357838389aa7201d9b44e234f831e98b2fd4`。

昇腾方案是业内**“解析执行与动态 Capture 路线（Eager + Graph Replay）”**的代表范式，同时通过 TorchAir 探索了 `torch.compile` / GE 图编译路线。

---

## 官方路线侧重与侧重程度说明

- **解析执行与动态 Capture 路线（绝对主导 / 工业交付主线，侧重程度：★★★★★）**：
  - **定位与实际使用**：华为昇腾在 vLLM 服务端推理中的**唯一主力生产路径**。基于 `torch_npu` (PrivateUse1)、ATen 算子映射（`op-plugin`）对接 CANN ACLNN，并通过 `vllm-ascend` 的 ModelRunner 进行 **ACL Graph 捕获与重放**。该路线兼顾了 PyTorch 动态图生态、PagedAttention 定制算子与零 Python 开销的极速执行。
- **AOT / GE 图编译路线（前沿探索 / 实验验证，侧重程度：★★☆☆☆）**：
  - **定位与实际使用**：华为通过 `torchair` 对接 PyTorch 2.x `torch.compile` / Dynamo，将 FX Graph 转换为 GE (Graph Engine) 静态图。
  - **受限因素**：在面对大模型长序列、连续批处理（Continuous Batching）以及动态 Shape 频繁变化的 Serving 场景下，GE 图的重编译开销和动态算子覆盖度仍存在工程挑战，目前主要作为前沿编译探索，非 vLLM 线上服务默认推荐。

---

## 路线子目录说明与权威对应

- **`capture-eager/`**：**解析执行与动态 Capture / ACL Graph 路线（主线）**
  - **核心链路**：`vllm-ascend` (ModelRunner / ACL Graph) → `torch_npu` (PrivateUse1 / NPUStream) → `op-plugin` (ATen 映射至 ACLNN/ACL) → CANN Runtime (Task 队列下发) → Ascend Driver → NPU。
  - **关键机制**：ACL Graph 捕获与重放（消除 Python 开销）、双重异步任务下发（PT Copy / NPU Task Queue）、专用 FlashAttention/PagedAttention NPU 算子定制。
  - **对应权威 Owner**：`20260903-102` (Plugin 激活与 Platform)、`20260903-103` (AscendRunner 与 KV)、`20260903-104` (ACLGraph 与 Shape 策略)。
- **`aot/`**：**AOT 图编译路线（TorchAir / GE）**
  - **核心链路**：通过 `torchair` 对接 PyTorch 2.x `torch.compile` / Dynamo 前端，将 FX Graph 转换为 GE (Graph Engine) 静态图后进行整图优化并下发。
  - **对应权威 Owner**：`20260903-104`。
- **`comm/`**：**通用底层硬件与运行时基础设施**
  - 覆盖 Ascend 芯片架构（DaVinci 架构、Cube/Vector 计算单元、Unified Buffer / L1 / L2 / HBM 存储层次）、CANN 驱动与 HAL 接口、HCCL 集合通信拓扑与版本兼容矩阵。
  - **对应权威 Owner**：`20260903-100` (硬件结构与搬运)、`20260903-101` (CANN 与 HCCL)、`20260903-105` (Feature Matrix 与 Release Tuple)。

---

## 调研现状：已证实事实 vs 显式空缺

### 1. 已证实事实（基于真实源码与静态审阅）
- [x] **PrivateUse1 设备注册**：`torch_npu` 完整实现了 PyTorch C++ 原生 `c10::DeviceType::PrivateUse1` 接口，支持设备张量与 Dispatcher 分派（权威证据：`20260903-101`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。
- [x] **双重异步架构（Double-Asynchronous Pipeline）**：`torch_npu` 包含 Host 端的 TaskQueue 线程与 Device 端的 NPUStream 硬件队列，并通过事件依赖维护交叉因果保序（权威证据：`20260903-101, 103`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。
- [x] **ACL Graph 捕获机制**：`vllm-ascend` 在预热阶段对固定 Batch 尺寸进行图捕获，执行期通过更新固定输入内存并调用 `aclmdlExecute` 实现零 Python 开销重放（权威证据：`20260903-104`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。

### 2. 显式空缺与未决问题（待调研项 / UNKNOWN）
- [ ] **【空缺 1】CANN 闭源层内部调度细节**：CANN Runtime 下层（如 AscendCL 底层 `libascendcl.so` 与驱动沟通细节、内部任务调度仲裁器）部分闭源，部分内部锁机制缺乏源码证据（标记为 `UNKNOWN`，证据等级：`VENDOR_CLAIM × UNVALIDATED`）。
- [ ] **【空缺 2】TorchAir 动态 Shape 图编译成熟度**：在服务场景下，TorchAir 面对连续批处理和动态变长序列时，GE 图重新编译（Re-compile）的开销与缓存淘汰策略尚缺乏大规模 Serving 验证数据（标记为 `UNKNOWN`）。
- [ ] **【空缺 3】真实物理硬件实测指标**：在未实际挂载 Ascend 物理卡的环境下，不得宣称具备 `DEVICE_SMOKE`、`DEVICE_NUMERIC` 或性能压测结论。
