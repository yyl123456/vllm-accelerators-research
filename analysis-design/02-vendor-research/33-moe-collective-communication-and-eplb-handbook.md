# MoE 专家并行 AllToAll 集合通信与动态负载均衡优化实战指南

> **调研与编写日期**：2026-09-30  
> **涉及核心源码基线**：
> - **Huawei Ascend**：`vllm-ascend/csrc/mc2/`、`eplb/communicator.py`、`moe_comm_method.py`
> - **vLLM Upstream & DeepSeek**：`vllm/distributed/eplb/`、`eplb_state.py`、`deepseek_v2.py`
> - **Tenstorrent TT-Metal**：`models/demos/gpt_oss/tt/experts_throughput/fused_decode.py`、`ttnn.experimental.moe_gpt`、`ttnn.experimental.deepseek_moe_reduce_scatter`
> - **Google TPU / OpenXLA**：`features/Collective_Communication_Matmul.yml`、`AsyncCollectiveAnnotator`
> 
> **权威文档对应**：`20260903-vllm-ai-infra-research/` 之 `10-ascend/20260903-100`、`12-tenstorrent/20260903-120`、`14-decisions/20260903-140-141`。

---

## 1. MoE 专家并行集合通信面临的核心痛点

在稠密模型（Dense）中，张量并行（TP）主要依赖 **AllReduce** 或 **ReduceScatter + AllGather**，通信模式规则、数据量对称、通信对等。
然而在混合专家大模型（如 DeepSeek-V2/V3/V4、Mixtral、Qwen-MoE）中，引入了专家并行（EP, Expert Parallelism）：

1. **动态非均衡负载与长尾倾斜（Load Imbalance & Hotspot）**：
   * 路由网络（Router / Gate）根据上下文内容动态选择 Top-K 专家；
   * 实际推理中，某些专家会被绝大多数 Token 选中成为“热点专家（Hotspot Expert）”，而部分冷门专家接收 Token 极少；
   * 传统静态 AllToAll 要求所有卡互相同步，**整机集群被迫等待最慢的“热点专家卡”计算与通信完成，产生严重的木桶效应**。
2. **AllToAll 通信爆炸与 Crossbar 拥塞**：
   * 在千卡跨机集群中，每个 Token 都可能被送往任意节点，形成完全的 $N \times N$ 全互联流量（All-to-All Personalized Exchange）；
   * 导致跨机 Spine/Leaf 交换机与网卡出现严重的拥塞丢包与排队等待。
3. **密集小包开销（Fine-grained Small Packet Overhead）**：
   * Token 分发与归并（Dispatch & Combine）伴随着大量的索引重排（Gather/Scatter/Permute），产生海量微小碎片通信，传统集合通信库启动开销难以承受。

---

## 2. 业界主流解决方案代码级深度剖析

### 2.1 动态负载均衡：EPLB (Expert Parallelism Load Balancer) 架构

在 [`third_party/vllm/vllm/distributed/eplb/eplb_state.py:27-150`](../../third_party/vllm/vllm/distributed/eplb/eplb_state.py#L27-L150) 中，工业界给出了成熟的动态负载均衡解答：

#### 1. 核心数学概念：物理专家与逻辑专家解耦
* **逻辑专家（Logical Expert）**：模型结构定义的专家（例如 DeepSeek 有 256 个逻辑专家）；
* **物理专家（Physical Expert）**：在真实卡上实例化的专家实例；
* **冗余专家复制（Redundant Replicas）**：
  引入 $N_{\text{redundant}}$ 个额外插槽（例如 32 个额外实例，总计 288 个物理专家）。
  $$N_{\text{physical}} = N_{\text{logical}} + N_{\text{redundant}}$$
  热点逻辑专家在多张卡上**同时拥有副本（Replicas）**。

#### 2. 运行时滑动窗口统计与在线迁移
* 在 `EplbStats` 中维护 `global_expert_load_window`（形状为 `[window_size, num_layers, num_experts]`）。
* 统计每个专家的实际 Token 访问频次；
* 当检测到专家负载不均达到阈值时，EPLB 调度器在后台异步规划专家权重重排（`rearrange_expert_weights_inplace`），将热点专家克隆到空闲卡上，分散 AllToAll 的目标负载。

#### 3. 华为昇腾对 EPLB 的通信规避适配
在 [`huawei/third_party/vllm-ascend/vllm_ascend/distributed/eplb/communicator.py:9-35`](../../huawei/third_party/vllm-ascend/vllm_ascend/distributed/eplb/communicator.py#L9-L35)：
* 昇腾实现了 `AscendGlooEplbCommunicator`，在权重迁移时完全走 Host CPU 侧的 P2P 通道，**显式绕开昂贵的设备端 HCCL 集合通信缓冲区预留（`needs_profile_buffer_reservation = False`）**，确保线上推理数据流不被重平衡通信打断。

---

### 2.2 通信与计算双流水：DualPipe 范式与硬件重叠

以 DeepSeek-V3 的 DualPipe 和华为昇腾 MC2 融合算子为代表，核心思想是：**“彻底打破先通信完再计算的串行边界，将 AllToAll 通信拆解为微批次（Micro-Batch）或分块，与矩阵乘法完全交错推进”**。

```text
传统串行模式:
[AllToAll Dispatch] ───────────► [GEMM W1/W3] ──► [GEMM W2] ──► [AllToAll Combine]
     (耗时 T_comm)                 (耗时 T_gemm)                    (耗时 T_comm)

DualPipe / 融合重叠流水:
通信流水:  [Dispatch Chunk 0] ──► [Dispatch Chunk 1] ──► [Combine Chunk 0] ──► [Combine Chunk 1]
计算流水:          ▼                     ▼                     ▼                     ▼
             (等待就绪)           [GEMM Chunk 0]        [GEMM Chunk 1]         (等待完成)
```

#### 昇腾 MC2 算子级代码实现
* 在 [`vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp:284-318`](../../huawei/third_party/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp#L284-L318)：
  * 算子下发前，根据 `(M * K * topK * sizeof(int8_t)) * 3 + 10MB` 分配三阶段双缓冲 Workspace；
  * `Mc2CcTilingConfig` 在硬件级别直接为 AI Core 和 Vector Core 生成融合调度指令：
    1. **Stage 1 (Dispatch)**：板载 DMA 接收远端 Token 到 `HcclMC2WorkSpace`；
    2. **Stage 2 (FFN/SwiGLU)**：Cube Core 计算已就绪的 Chunk；
    3. **Stage 3 (Combine)**：计算完毕的结果直接推入硬件发射 FIFO 吐出。

---

### 2.3 Tenstorrent：端到端稀疏流（Sparse Flow）与端侧 ReduceScatter

在 [`tenstorrent/third_party/tt-metal/models/demos/gpt_oss/tt/experts_throughput/fused_decode.py:10-28`](../../tenstorrent/third_party/tt-metal/models/demos/gpt_oss/tt/experts_throughput/fused_decode.py#L10-L28)，Tenstorrent 展现了一种极具参考价值的**全稀疏硬件流水模式**：

1. **废弃稠密中继（Dense Flow Elimination）**：
   * 传统做法：先 AllToAll 全量分发 → 沿专家维度广播复制 → 稠密 Batch Matmul → 全量 Combine。
   * TT 创新做法：`all_to_all_dispatch_metadata` 只提取轻量索引，`ttnn.experimental.moe_gpt` 算子内部直接将 **Tilize 布局变换 + 门控矩阵乘（W0/W1） + SwiGLU 激活 + 环形 AllToAll 环 + 下投影（W2） + Combine** 全部熔炼在同一个计算核中。
2. **端侧快速规约（`deepseek_moe_fast_reduce_nc`）**：
   * 融合输出利用 `deepseek_moe_reduce_scatter` 算子直接在 Mesh 片上网络上执行原位规约散播，避免数据在 Host 与 Device 之间多次往返倒腾。

---

## 3. MoE 专属集合通信算子自研设计法则 (Blueprint)

若要为自研 AI 芯片打造一套专用于 MoE 架构的高性能集合通信算子，必须遵循以下 **四大约束法则**：

### 1. 分层级拓扑感知路由（Hierarchical Two-Stage Routing）
* **禁止扁平 AllToAll**：严禁让集群中所有节点直接发起 $N \times N$ 全互连连接。
* **两阶段层次化分发**：
  * **Intra-Node（节点内）**：充分利用节点内超高带宽（NVLink / HCCS / NoC），走 Full-Mesh 直连极速交换；
  * **Inter-Node（节点间）**：每个节点由代表卡（Leader Rank）对本节点送往同一远端节点的 Token 进行**批量打包压缩（Batch Packing）**，通过 RDMA 集中交换，到达远端后再在节点内二阶段二次分发。跨机网络连接数从 $O(P^2)$ 降低到 $O(M^2)$（$M$ 为物理节点数）。

### 2. 掩码感知与非均等稀疏传输（Sparse Mask-Aware DMA）
* **元数据先行**：在发送庞大的 Hidden States 数据之前，先通过极低延迟的通道同步一张极其轻量的专家路由布尔掩码（`mc2_mask`，仅占几十字节）；
* **硬件级跳跃搬运（Strided Scatter/Gather DMA）**：
  板载 DMA 引擎必须支持条件过滤指令：仅当该槽位对应的 Mask 为 `True` 时才触发物理内存搬运，彻底丢弃全量 Padding 补齐产生的巨大无效带宽开销。

### 3. Ping-Pong 硬件状态机与微批次切片（Micro-Batch Tiling）
* 将单次 Step 的输入 Batch 拆分为至少 2 个 Micro-Chunk（$C_0, C_1$）；
* 计算核心与通信引擎绑定双向硬件信号量（Hardware Semaphore）：
  * 通信引擎完成 $C_0$ Dispatch $\to$ 触发原子自增信号量 $\to$ 唤醒计算核开始 GEMM；
  * 计算核计算 $C_0$ 的同时，通信引擎全速搬运 $C_1$；
  * 从而将通信延迟的绝大部分（甚至 100%）完全隐藏于 GEMM 计算阴影之中。

### 4. 冗余物理专家动态映射机制（EPLB-Aware Addressing）
* 集合通信底层地址解析层（Address Generator）必须原生支持“一对多物理映射表”；
* 当上层路由产出的目标是逻辑专家 $E_{\text{logical}}$ 时，通信引擎根据本地 EPLB 负载表，自动轮询（Round-Robin）或哈希派发到具有相同权重的空闲物理副本卡，从通信发射端天然消除目标节点的排队热点。
