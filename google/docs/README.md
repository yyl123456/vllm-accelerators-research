# Google TPU / XLA 软件栈调研现状与知识库

> **资料角色声明**：
> 本目录归档 Google TPU 与 OpenXLA 在 LLM 推理服务系统中的技术调研。
> **权威文档关系**：Google TPU 路线的唯一权威结论集合为 [`../../20260903-vllm-ai-infra-research/13-tpu/`](../../20260903-vllm-ai-infra-research/13-tpu/)（文档编号 `20260903-130` ~ `135`）。
> **固定源码 Revision**：`google/third_party/tpu-inference/` 锁定为 `fd33800041510b957cd2da6199742cce2b5fd113`；`google/third_party/torch_tpu/` 锁定为 `d7a3c1cc73a115376a058c7664986f5e34ca8362`（2026-10-22，Google 官方开源 PyTorch 原生 TPU 后端）。

Google 方案是业内**“AOT 静态编译与 SPMD 多芯片并行路线”**的代表范式。

---

## 官方路线侧重与侧重程度说明

- **AOT 静态编译与 SPMD 执行路线（唯一生产主导，侧重程度：★★★★★）**：
  - **定位与实际使用**：Google TPU / XLA 软硬件栈架构的**绝对主导与强制基石**。TPU 架构原生为了张量网格（MXU/Torus）最大吞吐而设计，要求模型在离线或预热期通过 JAX/PyTorch 前端（如 JAX 原生 `flax_nnx` 模块或基于 `torchax` 的 `VllmModelWrapper`，详见 `model_loader.py:657-688`）Lowering 至 StableHLO / HLO IR，由 OpenXLA 编译器完成全图优化与 SPMD 自动分片，生成静态二进制交付 PJRT C ABI（`PJRT_Api`，`pjrt_c_api.h:3030`）/ `libtpu` 执行。
  - **动态 Shape 应对**：强制要求使用 **Shape Bucketing（分桶机制）** 与 **Compile Cache**（`compilation_manager.py:65-74`：`jax_compilation_cache_dir` 挂载 `VLLM_XLA_CACHE_PATH`）将变长输入离散化，杜绝在线动态编译开销。
- **解析执行 / 纯 Eager 路线（仅 Debug / 生产禁用，侧重程度：☆☆☆☆☆）**：
  - **定位与实际使用**：TPU 硬件不支持**无图**的逐算子即时派发。在固定基线 `tpu-inference@fd338000` 中，`tpu_platform.py:163` 显式设置 `simple_compile_backend = "eager"` 仅用于规避 PyTorch 原生 `torch.compile` Inductor 生成，而将完整编译委托给 JAX/XLA；若强行脱离图编译执行逐算子下发，系统会退化为小图逐个启动甚至回退 CPU，性能极度低下，官方文档与工程实践中**严禁在 Serving 生产环境中使用**。
  - **2026-10 校准**：TorchTPU 开源后新增 “strict eager”——逐 op 建 StableHLO 图、以 `-O1` 即时编译并复用 Tier-1 缓存，仍属“每算子一次编译”，不是无编译的解释执行；因此该结论需限定为“不支持无图执行”，而非“不支持逐算子派发”。详见 [`aot/20261023-TorchTPU-开源架构与vLLM接入.md`](aot/20261023-TorchTPU-开源架构与vLLM接入.md)。

---

## 路线子目录说明与权威对应

- **`aot/`**：**AOT 静态编译与 SPMD 执行路线（主线）**
  - **核心链路**：JAX (`flax_nnx`) / PyTorch (`torchax` 桥接包装) 前端 → Tracing 与 HLO IR 转换 → OpenXLA 全图优化与 SPMD 自动分片 → PJRT C ABI (`PJRT_Api`) 运行时接入 → `libtpu` / TPU Runtime → TPU 硬件。
  - **调度与内存**：静态 Shape 编译与 Shape Bucketing（分桶机制：`num_reqs_paddings`、`token_paddings`、`BLOCK_BUCKETS`）；基于 SPMD 的大规模网格并行；编译产物缓存（Compile Cache）与复用。
  - **对应权威 Owner**：`20260903-132` (TPUInference-Platform 与 Runtime)、`20260903-133` (TPU-ModelRunner 与 PagedAttention)、`20260903-134` (XLA Shape Bucket 与 CompileCache)。
- **`capture-eager/`**：**解析执行路线（对比参考，2026-10 后有更新）**
  - TPU 体系仍要求所有计算经 XLA 编译为 executable，不存在“无图解释执行”；但 TorchTPU 的 strict eager 与 Defer & Fuse 已把“逐算子构图 + 即时编译 + 缓存复用”做成官方支持的常规模式，不再是纯 Debug 路径。
- **`comm/`**：**TPU 硬件拓扑与通用通信**
  - 覆盖 TPU 芯片架构（MXU 矩阵乘单元、VPU 向量单元、HBM 显存）、ICI (Inter-Chip Interconnect) 高速直连网络与 2D/3D Torus 环形拓扑、PJRT 统一 Runtime 协议标准（`pjrt_c_api.h`）。分布式通信后端通过 `TpuCommunicator`（`tpu_platform.py:537`）协同。
  - **对应权威 Owner**：`20260903-130` (硬件体系与 Torus 互联)、`20260903-131` (OpenXLA-PJRT 与 libtpu)、`20260903-135` (TPU 版本矩阵与生产契约)。

---

## TorchTPU（2026-10 开源）：PyTorch 原生路径

Google 于 2026-10 公开 `google-pytorch/torch_tpu`（PyPI `torch_tpu==0.2.0`），官方 FAQ 明确 **PyTorch/XLA 已废弃并由 TorchTPU 取代**，且无 JAX runtime 依赖。它与既有 `tpu-inference`（已改名 vLLM JAX）构成两条并行前端：

| 维度 | TorchTPU 事实 | 源码锚点 |
|---|---|---|
| 接入方式 | out-of-tree `PrivateUse1` 后端，经 `torch.backends` 入口点自动加载并重命名为 `tpu` | `build_files/entry_points.bzl:17-19`；`src/torch_tpu/_loader.py:256,387` |
| 执行模型 | ATen 算子先落成 `DeferredOp`，后台单物化线程成图/编译，单执行线程下发 PjRt | `csrc/eager/device_buffer.h:415`；`csrc/eager/materialize.cc:291-360` |
| 编译 | StableHLO → OpenXLA（O1/O2），跳过 TorchInductor | `csrc/common/compilation_spec.h:33-37` |
| 缓存 | Tier-1 进程内 / Tier-2 主机共享内存 / Tier-3 远端对象存储 | `csrc/common/compilation_cache.h:284`；`env_vars.h:203,207` |
| 动态 shape | Bounded Dynamism，128 对齐分桶（最多 8 桶） | `src/torch_tpu/_internal/compile/dynamic/dynamic_compiler.py:123-151` |
| vLLM 接入 | 独立仓库 `vllm-project/vllm-torchtpu` 提供 Platform 插件；本仓库提供 `torch.compile(backend="tpu")` | `src/torch_tpu/_internal/compile/__init__.py:31` |

> **专题分析**：[`aot/20261023-TorchTPU-开源架构与vLLM接入.md`](aot/20261023-TorchTPU-开源架构与vLLM接入.md)（含四种执行模式、与 13-tpu 既有结论的冲突校准、开闭源边界）。
> **路线再定位**：TorchTPU 属“**B 接入 + A 编译**”合流型——接入层是完全的 PrivateUse1 + Python dispatch，编译层是 XLA 全图 AOT 产物 + 多级缓存；没有 CUDA Graph 式 replay，执行图每次由 DeferredOp 图即时重建（缓存命中则只复用 executable）。

---

## 调研现状：已证实事实 vs 显式空缺

### 1. 已证实事实（基于真实源码与静态审阅）
- [x] **双模型加载后端与 Torchax 桥接**：`tpu-inference/tpu_inference/models/common/model_loader.py:657-688` 证实 TPU 支持 `flax_nnx` 与 `vllm` 两种模型实现；对于 PyTorch 模型通过 `torchax`（`models/vllm/vllm_model_wrapper.py:277, 461`）将 PyTorch 算子转为 JAX 表达式并纳入全图 `@jax.jit` 编译（权威证据：`20260903-132`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。
- [x] **Shape Bucketing 机制与预编译池**：已在 `tpu_inference/runner/compilation_manager.py:55-98, 250-290` 与 `tpu_inference/runner/utils.py:149-212` 源码中证实其将请求数、Token 长度与 Block 划分为离散阶梯桶（Buckets），支持通过多线程池（`NUM_PRECOMPILE_WORKERS`）在服务启动前完成 AOT 预编译并写入持久化缓存目录 `VLLM_XLA_CACHE_PATH`（权威证据：`20260903-134`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。
- [x] **PJRT 统一 C ABI 接口规范**：`xla/xla/pjrt/c/pjrt_c_api.h:3030` 明确声明了 `PJRT_Api` 结构体，包含 Client、Device、Buffer、LoadedExecutable 的全生命周期函数指针，解耦编译器与底层硬件驱动（权威证据：`20260903-131`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。
- [x] **Platform 契约与通信封装**：`tpu_platform.py:155-208, 537` 明确声明了 `PlatformEnum.TPU`、`dispatch_key = "XLA"`、`simple_compile_backend = "eager"` 以及 `TpuCommunicator` 分布式通信器（权威证据：`20260903-132`，等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`）。
- [x] **TorchTPU 原生后端已开源**：`torch_tpu@d7a3c1cc` 证实 `PrivateUse1` 注册、DeferredOp、单物化/执行线程、O1/O2 双档编译、三级缓存与 bounded dynamism 均由本地源码直接支撑，闭合了 `20-vllm-torchtpu-xla.md` 时代的 5 项“无法从源码确认”（等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`；硬件实测仍缺）。

### 2. 显式空缺与未决问题（待调研项 / UNKNOWN）
- [ ] **【空缺 1】`libtpu` 内部硬件调度与驱动闭源**：Google 仅开源了 PJRT 包装层，核心的 `libtpu.so` 二进制及其与底层 Linux 内核驱动的具体交互协议为闭源实现（标记为 `UNKNOWN`，证据等级：`VENDOR_CLAIM × UNVALIDATED`）。
- [ ] **【空缺 2】跨节点动态扩展与弹性容错**：在大型 Torus 拓扑中，节点故障后的快速重编排机制与编译恢复代价仍缺乏公开系统级验证（标记为 `UNKNOWN`）。
- [ ] **【空缺 3】真实物理 TPU 环境实测**：未直接在物理 TPU VM 上执行模型吞吐与数值对比，无 `DEVICE_SMOKE`、`DEVICE_NUMERIC` 证据产物。
