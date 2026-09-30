# 跨主流 AI 加速器集合通信算子实现与优化策略深度调研报告

> **调研日期**：2026-09-30  
> **涉及核心源码基线**：
> - **Huawei Ascend**：`huawei/third_party/torch_npu`、`vllm-ascend`、CANN HCCL 头文件规范（`hccl.h`、`hccl_types.h`）
> - **Tenstorrent**：`tenstorrent/third_party/tt-metal` (`c634b1ca`)、`vllm-tt-plugin` (`f6995475`)
> - **Google TPU / OpenXLA**：`google/third_party/xla`、`tpu-inference` (`fd338000`)
> - **Qualcomm Cloud AI**：`qualcomm/third_party/vllm-qaic` (`3212cc67`)、`efficient-transformers`
> - **NVIDIA (参照基准)**：`nvidia/third_party/nccl` (`12df1a11`)
> - **跨框架参考**：`modular/third_party/modular` (`e700d92f`)
> 
> **权威文档索引对应**：`20260903-vllm-ai-infra-research/` 之 `10-ascend/20260903-100`、`11-qaic/20260903-110`、`12-tenstorrent/20260903-120`、`13-tpu/20260903-130`、`14-decisions/20260903-140-141`。

---

## 1. 调研背景与优化目标

在 LLM 大模型分布式训练与高并发 Serving 场景中，随着模型参数量激增与多卡/多机张量并行（TP）、流水线并行（PP）、专家并行（EP/MoE）以及上下文并行（CP）的广泛使用，**集合通信（Collective Communication）的时延和带宽开销已成为最主要的系统瓶颈之一**。

本调研面向“自研或深度优化集合通信算法库（如优化 AllReduce、ReduceScatter、AllGather、AllToAll）”，全面剖析业界主流芯片厂商的集合通信算子底层实现原理、网络拓扑匹配与前沿软硬件协同优化策略。

---

## 2. 各厂商集合通信实现与架构剖析

### 2.1 华为昇腾 (Huawei Ascend HCCL & MC2 体系)

#### 1. 核心架构与原语规范
- **HCCL 原语规范**：定义于 [`huawei/third_party/torch_npu/third_party/hccl/inc/hccl/hccl.h`](../../huawei/third_party/torch_npu/third_party/hccl/inc/hccl/hccl.h)。原生支持 `HcclAllReduce`、`HcclAllGather`、`HcclAllGatherV`、`HcclReduceScatter`、`HcclAlltoAll`、`HcclAlltoAllV` 及 `HcclBatchSendRecv`。
- **硬件拓扑底座**：基于片间 HCCS（Huawei Cache Coherent System）高速私有总线与跨节点 RoCEv2（RDMA over Converged Ethernet）。
- **确定性计算支持**：HCCL 提供 `HcclSetConfig(HCCL_DETERMINISTIC)`，满足金融/科研的高精度确定性需求。

#### 2. 深度算子融合优化：MC2 (Mega-MoE Communication & Computation)
在昇腾大模型推理中，针对 MoE 架构的专家并行（EP）通信，华为不仅使用常规的 AllToAll，还在底层开源了专有的 **MC2** 软硬件融合通信算子库：
- **源码位置**：[`huawei/third_party/vllm-ascend/csrc/mc2/`](../../huawei/third_party/vllm-ascend/csrc/mc2/) 与 [`moe_comm_method.py:262-297`](../../huawei/third_party/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py#L262-L297)。
- **核心算子**：
  * `torch_npu.npu_moe_distribute_dispatch_v2`（Token 分发）
  * `torch_npu.npu_moe_distribute_combine_v2`（结果归并）
- **优化机制（通信与计算重叠 Overlap）**：
  * **分级流水通信（Hierarchy Comm Alg）**：通过 `comm_alg="hierarchy"`，先在单机 8 卡 HCCS 环内做局部聚合，再通过跨机 RoCE 交换，避免全网 Crossbar 冲突。
  * **Mask 驱动的稀疏搬运**：利用 `mc2_mask` 在硬件层面直接跳过未激活 Expert 的 Token 通信，彻底消除无效 Payload 传输。
  * **Workspace 硬件锁**：在 Ascend C 核函数中通过 `HcclMC2WorkSpace` 维护板载状态，结合 `AscendC::Mc2CcTilingConfig` 实现计算流水（FFN/Matmul）与 HCCS 数据搬运的周期级精确重叠。

---

### 2.2 Tenstorrent (TT-Metal CCL 体系：片上 Mesh 与片间 Ethernet)

#### 1. 核心架构与编程模型
Tenstorrent 不依赖传统的 Host-Centric 驱动，而是采用**分布式 RISC-V 计算阵列（Tensix 核心）+ 原生以太网通道（Ethernet Core / ERISC）**架构：
- **源码位置**：[`tenstorrent/third_party/tt-metal/ttnn/cpp/ttnn/operations/ccl/`](../../tenstorrent/third_party/tt-metal/ttnn/cpp/ttnn/operations/ccl/)。
- **原生多拓扑支持**：在 [`ccl_common.hpp:37-64`](../../tenstorrent/third_party/tt-metal/ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp#L37-L64) 中，TT-Metal 原生抽象了 `LineTopology`、`RingTopology` 与 `Mesh/Torus` 2D 拓扑。系统根据物理硬件拓扑状态（`is_axis_straight`、`is_axis_wrap_wired`）自动选择线性（Line）还是闭环（Ring）算法。

#### 2. 核心优化策略：ERISC 硬件流水线与零拷贝转发
- **Data Movement 内核独立**：每个 Tensix 核心分为计算核与数据搬运核。CCL 算法直接通过专用 ERISC 核心执行 `ccl_send`、`ccl_wait_completion` 等微操作（UOPs）。
- **Fabric EDM（Event-Driven Mover）**：引入 Fabric 路由器机制（`fabric_edm_types.hpp`），数据包在片上 SRAM（L1 Buffer）与跨芯片 Ethernet 之间流式穿透传输，无需写回全局 DRAM，大幅降低通信延迟。
- **动态 Worker Core 选择**：通过 `choose_worker_cores()`，根据传输链路数量（`num_links`）和拓扑方向，动态在芯片网格中选取最近的计算核建立通信管线。

---

### 2.3 Google TPU / OpenXLA (编译器驱动的 SPMD 集合通信)

#### 1. 核心架构与拓扑抽象
Google TPU 拥有物理级的 2D/3D Torus（环面网格）互联拓扑（ICI，Inter-Core Interconnect）。Google 不走动态调度下发，而是**将集合通信作为一等公民完全交给 XLA 静态编译器进行全局图变换与调度**：
- **源码位置**：[`google/third_party/xla/xla/service/`](../../google/third_party/xla/xla/service/) 与 `backends/gpu/collectives/`。

#### 2. XLA 编译器的通信优化 Pass 矩阵
XLA 内置了强大的集合通信优化 Pass 管线：
1. **`AllReduceCombiner` / `CollectiveCombiner`**：
   * 在 [`all_reduce_combiner.h`](../../google/third_party/xla/xla/service/collective_utils.h)，将多个微小 Tensor 的 AllReduce 算子自动打包合并为一个大的批次通信，彻底抹平每次调用通信驱动的启动时延（Launch Overhead）。
2. **`AllGatherDecomposer` / `Simplifier`**：
   * 在 [`all_gather_decomposer.h:37-60`](../../google/third_party/xla/xla/service/all_gather_decomposer.h#L37-L60)，当硬件不支持某些非对齐维度的 AllGather 时，编译器自动将其等价分解为 `DynamicUpdateSlice` 与 `AllReduce` 的组合，保障算子语义等价性。
3. **`AllReducePromotion` 与代码外提**：
   * 自动探测并消除多余的通信算子，将循环不变的 AllReduce 提升（Code Motion）到 While 循环体外部执行。
4. **SPMD 分区与重叠调度**：
   * 编译器在静态已知拓扑维度的前提下，执行 `AsyncCollectiveAnnotator`，将通信自动拆解为 `AllReduceStart` 与 `AllReduceDone`，并通过计算图依赖重排，把通信延迟完全隐藏在 Matmul 等密集计算节点背后。

---

### 2.4 Qualcomm QCCL 与 NVIDIA NCCL (工业界成熟基准)

#### 1. Qualcomm QCCL (Qualcomm Cloud Collective Library)
- **源码定位**：[`qualcomm/third_party/vllm-qaic/vllm_qaic/distributed/communicator.py`](../../qualcomm/third_party/vllm-qaic/vllm_qaic/distributed/communicator.py)。
- **特点**：在 `QAicCommunicator` 中声明 `dist_backend = "qccl"`。对于板内 4 颗或单卡多 NSP 芯片，QCCL 依赖 PCIe P2P 共享内存与 DMA 进行通信；在跨机时退化为标准 Socket/Gloo 机制。在高通 AOT 模式下，多卡并行主要依赖编译时静态生成的 `mdp_ts_*.json`（Tensor-Slice 分区）将通信切分固化，运行时通信模式完全静态化。

#### 2. NVIDIA NCCL (全功能工业标杆)
- **源码位置**：[`nvidia/third_party/nccl/src/`](../../nvidia/third_party/nccl/src/)。
- **三大核心架构亮点**：
  1. **多拓扑多算法自动探测（`graph/rings.cc` & `graph/trees.cc`）**：
     * **Ring 算法**：适合中大 Payload，带宽利用率高（有效利用 NVLink 双向环），总耗时为 $2(N-1)/N \cdot (S/B)$。
     * **Tree 算法（二项树/双二叉树）**：针对微小数据包（如几 KB 到几百 KB），将延迟项从 $O(N)$ 降到 $O(\log N)$。
  2. **多通道并发（Multi-Channel）**：
     * 自动将单次大通信分割到 16~32 个并行 Channel 中，每个 Channel 对应独立的 GPU SM 线程块与 NVLink 物理通道，打满硬件带宽。
  3. **核函数微架构优化（Primitives & Protocols）**：
     * 提供 `LL`（Low Latency，带 Flag 标志位的流水线）、`LL128`（专为 NVLink 优化的 128 字节缓存行直传协议）以及 `Simple`（大包直接点对点 DMA 协议），在不同数据量下自动切换最优协议。

---

## 3. 跨厂商横向特性对比与架构决策矩阵

| 架构特性 | Huawei Ascend (HCCL/MC2) | Tenstorrent (TT-Metal CCL) | Google TPU (OpenXLA) | NVIDIA (NCCL) | Qualcomm (QCCL) |
|---|---|---|---|---|---|
| **通信控制平面** | CANN 驱动 + TaskQueue | Device 端 RISC-V 独立调度 | XLA 编译器静态生成 SPMD | GPU Kernel 内部驱动 (SM-driven) | 编译器 MDP + Platform SDK |
| **物理拓扑结构** | HCCS 8卡全互联 + RoCEv2 | 2D Mesh / 2D Torus 网格 | 2D/3D Torus 环面网格 | NVLink Switch 全互联 + InfiniBand | PCIe P2P / 专有直连通道 |
| **拓扑路由算法** | 环形环状 + 分级跨机 (Hierarchy) | Line / Ring / 2D Torus 自动降维 | 编译器根据物理 Coordinate 映射 | Multi-Ring + Double-Binary-Tree | 静态 Tensor-Slice 固定映射 |
| **计算与通信重叠** | MC2 算子级深度融合 (Kernel 内) | Compute Core 与 DataMover Core 硬件解耦 | HLO 图级依赖重排 (`AsyncStart/Done`) | CUDA Stream 并发 + 管道切分 | 离线静态规划排布 |
| **小包延迟优化** | 共享内存零拷贝 (Shared Buffer) | 片上 L1 SRAM 寄存器穿透 | `AllReduceCombiner` 自动聚合成大包 | `LL` / `LL128` 标志位轻量级握手协议 | 离线打包合并 |
| **开源开放程度** | 头文件规范开源 / 运行时驱动闭源 / MC2 部分开源 | **全栈完全开源** (Metalium + Kernel C++) | **全栈完全开源** (XLA 编译管线) | **核心库完全开源** (C++/CUDA) | 仅暴露 Python 桥接 / 底层闭源 |

---

## 4. 自研与优化集合通信算法的设计指南（核心建议）

若要自研或深度优化专有加速芯片的集合通信算子，建议采取以下 5 大核心优化策略：

### 1. 算法与拓扑物理亲和（Topology-Aware Algorithm Selection）
* **小包走 Tree / 扁平星型，大包走 Ring / Pipeline**：
  必须像 NCCL/TT-Metal 一样，在 Host 侧或编译期建立拓扑感知图。对于 $S < 256\text{KB}$ 的小数据包（如控制信息、Norm 归一化同步），直接走树状多播（$O(\log N)$ 延迟）；对于 $S \ge 1\text{MB}$ 的大张量（如 TP 权重输出），走 Ring 或 2D 拆解环（打满物理链路双向带宽）。
* **单机内 HCCS/NVLink 与跨机 RDMA 层次化分级（Hierarchical Communication）**：
  严禁跨机所有节点进行平面 AllReduce。必须采用“单机内 ReduceScatter → 跨机节点代表 AllReduce → 单机内 AllGather”的两阶段/三阶段分级算法，将昂贵的跨机流量压缩到 $1/\text{单机卡数}$。

### 2. 算子融合与无感重叠（Compute-Communication Overlap）
* **参考昇腾 MC2 的 Fused-MoE 范式**：
  在 MoE Dispatch 和 Combine 阶段，不要调用通用的 AllToAll 阻塞等待，而是将通信切片与后续专家的 GEMM 计算放入同一个硬件调度流水中（通过软件双缓冲 Ping-Pong Buffer 实现通信与计算互相掩盖）。
* **参考 XLA 的静态 Combine 机制**：
  在编译期或上层框架层引入 Combiner 机制，将短时间窗口内派发的所有细碎梯度或激活通信自动打包为单个 Contiguous Buffer 一次性传输，消除频繁触发下发队列的 CPU 驱动开销。

### 3. 数据协议多档分级（Multi-Tier Protocols: Fast vs Bandwidth）
* **标志位嵌入的无同步协议（Lock-Free / LL Protocol）**：
  在小包传输中，消除显式的 Event 同步或操作系统中断。在目标内存的数据包尾部附带单调递增的 4/8 字节 `Flag`，接收端计算核通过轮询（Polling）该标志位即可判定数据就绪，直接进入计算。
* **零拷贝直传（Direct P2P DMA）**：
  充分利用芯片的片上高速 SRAM（如 TT 的 L1、高通的片上内存、昇腾的 L2/UB），让网络引擎直接将数据 DMA 到计算核近端内存，避免“设备显存 → Host 内存 → 网络卡”的多重拷贝。

### 4. 稀疏性与掩码感知传输（Mask-Aware Communication）
* 在动态路由（MoE、Token Pruning）场景中，引入通信掩码机制（如 `mc2_mask`）。上层先交换路由决策矩阵（极小的控制元数据），通信引擎仅搬运有实际数据的非零 Token 槽位，避免传输大量 Padding 产生的无效带宽消耗。
