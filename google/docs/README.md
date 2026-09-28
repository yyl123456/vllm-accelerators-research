# Google TPU / XLA 软件栈调研现状与知识库

> **资料角色声明**：
> 本目录归档 Google TPU 与 OpenXLA 在 LLM 推理服务系统中的技术调研。
> **权威文档关系**：Google TPU 路线的唯一权威结论集合为 [`../../20260903-vllm-ai-infra-research/13-tpu/`](../../20260903-vllm-ai-infra-research/13-tpu/)（文档编号 `20260903-130` ~ `135`）。
> **固定源码 Revision**：`google/third_party/tpu-inference/` 严格锁定为 `fd33800041510b957cd2da6199742cce2b5fd113`。

Google 方案是业内**“AOT 静态编译与 SPMD 多芯片并行路线”**的代表范式。

---

## 路线子目录说明与权威对应

- **`aot/`**：**AOT 静态编译与 SPMD 执行路线（主线）**
  - **核心链路**：JAX / PyTorch-XLA 前端 → Tracing 与 HLO IR 转换 → OpenXLA 全图优化与 SPMD 自动分片 → PJRT C ABI 运行时接入 → `libtpu` / TPU Runtime → TPU 硬件。
  - **调度与内存**：静态 Shape 编译与 Shape Bucketing（分桶机制）；基于 SPMD 的大规模网格并行；编译产物缓存（Compile Cache）与复用。
  - **对应权威 Owner**：`20260903-132` (TPUInference-Platform 与 Runtime)、`20260903-133` (TPU-ModelRunner 与 PagedAttention)、`20260903-134` (XLA Shape Bucket 与 CompileCache)。
- **`capture-eager/`**：**解析执行路线（显式空缺/对比参考）**
  - TPU 体系原生强制要求静态图与 XLA 编译，纯 Eager 解释执行仅作为 Debug 模式，性能极低。
- **`comm/`**：**TPU 硬件拓扑与通用通信**
  - 覆盖 TPU 芯片架构（MXU 矩阵乘单元、VPU 向量单元、HBM 显存）、ICI (Inter-Chip Interconnect) 高速直连网络与 2D/3D Torus 环形拓扑、PJRT 统一 Runtime 协议标准。
  - **对应权威 Owner**：`20260903-130` (硬件体系与 Torus 互联)、`20260903-131` (OpenXLA-PJRT 与 libtpu)、`20260903-135` (TPU 版本矩阵与生产契约)。

---

## 调研现状：已证实事实 vs 显式空缺

### 1. 已证实事实（基于真实源码与静态审阅）
- [x] **Shape Bucketing 机制**：已在 `google/third_party/tpu-inference/` 源码中证实其将连续可变的 Token 长度与 Batch 划分为有限离散桶（Buckets），每个桶预先触发 XLA 静态编译并持久化缓存（权威证据：`20260903-134`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。
- [x] **PJRT 统一接口规范**：OpenXLA 通过 PJRT 提供清晰解耦的 C ABI，隔离了编译器输出与设备硬件执行（权威证据：`20260903-131`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。

### 2. 显式空缺与未决问题（待调研项 / UNKNOWN）
- [ ] **【空缺 1】`libtpu` 内部硬件调度与驱动闭源**：Google 仅开源了 PJRT 包装层，核心的 `libtpu.so` 二进制及其与底层 Linux 内核驱动的具体交互协议为闭源实现（标记为 `UNKNOWN`，证据等级：`VENDOR_CLAIM × UNVALIDATED`）。
- [ ] **【空缺 2】跨节点动态扩展与弹性容错**：在大型 Torus 拓扑中，节点故障后的快速重编排机制与编译恢复代价仍缺乏公开系统级验证（标记为 `UNKNOWN`）。
- [ ] **【空缺 3】真实物理 TPU 环境实测**：未直接在物理 TPU VM 上执行模型吞吐与数值对比，无 `DEVICE_SMOKE`、`DEVICE_NUMERIC` 证据产物。
