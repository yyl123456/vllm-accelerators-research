# 大模型长文本上下文并行 (CP) 集合通信与底层算子实现实战指南

> **调研与编写日期**：2026-09-30  
> **涉及核心源码基线**：
> - **vLLM Upstream**：`vllm/v1/attention/ops/pcp.py`、`cp_common.py`、`dcp.py`
> - **Huawei Ascend**：`vllm-ascend/attention/context_parallel/common_cp.py`、`attention_cp.py`、`mla_cp.py`、`_npu_attention_update`
> - **NVIDIA / Megatron-LM 架构**：RingAttention (Striped Attention) 与 DeepSpeed-Ulysses 拓扑映射
> - **Google TPU / OpenXLA**：SPMD Sharded Attention、`CollectiveCombiner`
> 
> **权威文档对应**：`20260903-vllm-ai-infra-research/` 之 `06-vllm-distributed/20260903-65-PCP与DCP并行切分与KV映射`、`10-ascend/20260903-100`、`14-decisions/20260903-140-141`。

---

## 1. 核心问题背景：为什么长文本（128K~1M）必须依赖上下文并行？

在单卡显存能够容纳模型权重的场景下，随着长文本（Context Window）扩展至 128K、256K 乃至 1M Token：
1. **显存爆炸（Memory Wall）**：
   * 即使使用 GQA（分组查询注意力），单次请求的 KV Cache 显存占用也随序列长度 $S$ 线性暴增；
   * Attention 激活值（Activation Memory）在反向传播或超长 Prefill 时达到 $O(S^2)$ 或 $O(S)$，单卡无法容纳。
2. **算力墙（Compute Wall）**：
   * 单卡处理 1M Token 的 Prefill 耗时可能达到数十秒，违反真实服务时延 SLO。
3. **张量并行（TP）的扩展瓶颈**：
   * TP 切分的是注意力头数（Attention Heads）。当头数较少（如 GQA 只有 8 个 KV 头），TP 的扩展上限被硬性锁死在 8，无法在 16/32/64 卡集群上继续平摊长文本负载。

因此，**上下文并行（Context Parallelism, CP）成为破除 TP 限制、沿 Sequence 序列维度物理切分的唯一解法**。

---

## 2. 三大主流 CP 通信范式数学建模与代码链路

### 2.1 范式一：DeepSpeed-Ulysses (AllToAll 维度转置)

#### 1. 核心数学机制
Ulysses 利用两次双向 **AllToAll** 集合通信，将序列切分与头数切分进行正交维度转置：
1. **输入状态**：$Q, K, V$ 沿序列维度切分，每个 Rank 持有完整头数的一部分序列：
   $$\text{Shape: } \left[\frac{S}{P}, H, D\right]$$
2. **AllToAll 转置（Seq $\to$ Head）**：
   通过 AllToAll 集合通信交换数据，变为沿注意力头数切分，每个 Rank 持有完整的上下文序列：
   $$\text{Shape: } \left[S, \frac{H}{P}, D\right]$$
3. **本地局部 Attention 计算**：
   每个 Rank 在完整的序列上执行标准局部 FlashAttention；
4. **AllToAll 逆转置（Head $\to$ Seq）**：
   Attention 输出再次通过 AllToAll 转置回原始序列切分排布：
   $$\text{Shape: } \left[\frac{S}{P}, H, D\right]$$

#### 2. 通信耗时与优缺点
* **总传输量 (Per Rank)**：
  $$2 \times \frac{P - 1}{P} \cdot S \cdot H \cdot D$$
* **核心优势**：直接复用单卡极速 FlashAttention/Fused Attention 算子，代码极其精简；
* **核心缺陷**：
  * 受限于注意力头数（$H$ 必须能被 $P$ 整除）；
  * 在大规模跨机节点时，两次全连接 AllToAll 会造成严重网络拥塞。

---

### 2.2 范式二：RingAttention (双向环 P2P 流水线)

#### 1. 核心数学机制
RingAttention 完全不改变头数分布，将 $Q, K, V$ 物理切分在各卡上，每卡持有 $\frac{S}{P}$。
通过构建双向环形通信，在 $P$ 步内循环推进：
1. 每一步，Rank $i$ 持有本地的 $Q_i$ 不动；
2. 计算当前的局部注意力和 Log-Sum-Exp 归一化项（LSE）；
3. **P2P 非阻塞通信**：通过异步 `Send/Recv` 将 $K_j, V_j$ 传递给相邻节点，同时接收前驱节点的 $K_{j-1}, V_{j-1}$；
4. 随着环的旋转，利用 **Online Softmax（在线归纳更新算法）** 持续融合新的局部结果，直到转满 $P$ 步。

#### 2. 在线 Softmax 融合数学公式
设第 $k$ 步计算出的局部最大值为 $m_{\text{curr}}$，输出为 $O_{\text{curr}}$，历史累积最大值为 $m_{\text{prev}}$，历史输出为 $O_{\text{prev}}$：
$$m_{\text{new}} = \max(m_{\text{prev}}, m_{\text{curr}})$$
$$O_{\text{new}} = O_{\text{prev}} \cdot e^{m_{\text{prev}} - m_{\text{new}}} + O_{\text{curr}} \cdot e^{m_{\text{curr}} - m_{\text{new}}}$$
$$l_{\text{new}} = l_{\text{prev}} \cdot e^{m_{\text{prev}} - m_{\text{new}}} + l_{\text{curr}} \cdot e^{m_{\text{curr}} - m_{\text{new}}}$$
最终归一化输出：$O = \frac{O_{\text{final}}}{l_{\text{final}}}$。

#### 3. 通信耗时与优缺点
* **总传输量 (Per Rank)**：
  $$(P - 1) \cdot \frac{S}{P} \cdot 2 \cdot H_{KV} \cdot D = \frac{P-1}{P} \cdot 2 S \cdot H_{KV} \cdot D$$
* **核心优势**：
  * **突破头数限制**：即使 GQA 只有 1 个 KV 头，也能扩展到上百张卡；
  * **通信计算 100% 隐藏**：只要单步 Chunk 的 Attention 计算时间大于 P2P 传输时间，通信延迟完全被掩盖在计算之后；
* **核心缺陷**：需要对底层 Attention Kernel 进行深度定制，支持累积状态传入与因果掩码（Causal Mask）块对角线跳跃。

---

### 2.3 范式三：Prefill CP (PCP) 与 Decode CP (DCP) 生产级混合架构

在上游 vLLM 与华为昇腾 vllm-ascend 生产落地中，采取了目前业界最先进的**“Prefill 与 Decode 分离切分策略”**：

#### 1. Prefill Context Parallel (PCP)：前缀并行
* **源码位置**：[`third_party/vllm/vllm/v1/attention/ops/pcp.py:11-46`](../../third_party/vllm/vllm/v1/attention/ops/pcp.py#L11-L46)。
* **逻辑**：仅在超长提示词 Prefill 阶段开启序列切分；
* **KV 收集与槽位重构**：
  在 `_gather_prefill_cache_inputs` 中：
  ```python
  gathered_prefills = tuple(
      pcp_group.all_gather(tensor[num_decode_tokens:].contiguous(), dim=0)
      for tensor in tensors
  )
  ```
  在 Prefill 算子执行完后，通过一次高效的 `pcp_group.all_gather` 将切分的物理 KV 写回全局槽位表，避免了在复杂 Attention 算子内部旋转环的麻烦。

#### 2. Decode Context Parallel (DCP)：解码期切分与硬件归一化算子
* **源码位置**：[`huawei/third_party/vllm-ascend/vllm_ascend/attention/context_parallel/common_cp.py:121-155`](../../huawei/third_party/vllm-ascend/vllm_ascend/attention/context_parallel/common_cp.py#L121-L155)。
* **逻辑**：在 Decode 阶段，针对已有数十万 Token 的长上下文，将历史 KV Cache 物理切分给各卡（`num_computed_tokens_of_dcp`）；
* **昇腾硬件归一化专用算子**：
  华为在底层开发了硬件级融合指令 `_npu_attention_update`：
  ```python
  def _merge_dcp_attention_output(self, attn_output, softmax_lse, head_size):
      return _npu_attention_update(
          head_size,
          _process_attn_out_lse(attn_output, softmax_lse, dcp_size=self.dcp_size),
          dcp_size=self.dcp_size,
      )
  ```
  各卡独立计算切片 Attention，产出局部 `attn_output` 和局部 `softmax_lse`（Log-Sum-Exp），通过底层轻量通信直接在 NPU 硬件中原子融合成最终真实输出，完全省去了繁琐的手写归一化 Kernel。

---

## 3. 长文本集合通信算子自研设计法则 (Blueprint)

若要为自研分布式推理引擎构建一套高性能长文本 CP 算子库，建议采纳以下 **“四大工程设计法则”**：

### 1. 混合分级拓扑映射（Hierarchical Hybrid CP）
* **机内 Ulysses + 跨机 RingAttention**：
  * 在单机 8 卡内（NVLink / HCCS 带宽高达数百 GB/s），部署 **Ulysses AllToAll**，享受单卡 FlashAttention 原生高性能与极简实现；
  * 在跨机网络间（受限于 RoCE/Infiniband 400G/800G 带宽），部署 **RingAttention**，将长上下文切分为大块在机间做点对点环形流水，打满网卡物理全双工带宽并隐藏延迟。

### 2. 因果三角形掩码动态跳过（Causal Mask Pruning）
* 在自回归（Causal）语言模型中，下三角矩阵意味着后半段 Token 无法看见前半段 Token：
  * 在 Ring 旋转过程中，约有一半的 Block 属于完全在掩码之外的无效计算（全零注意力）；
  * **通信裁剪策略**：通信调度器必须维护一张静态 Block 依赖拓扑图，对于目标卡不需要的 KV 块，直接跳过 P2P 发送，**物理网络流量直接减少 50%**。

### 3. 双缓冲异步流水线（Double-Buffered P2P Pipeline）
* 在执行 RingAttention 算子时，显存分配必须采用双缓冲（`Ping-Pong Buffer`）：
  * **Buffer A**：参与当前步的 Attention GEMM 计算；
  * **Buffer B**：通过异步通信流（Async Stream）在后台接收下一张卡传来的 KV；
  * 当计算完成时，切换两个 Buffer 指针，实现通信与计算时延的极致掩盖。

### 4. 硬件级在线 Softmax 状态更新算子（Fused Online Softmax）
* 避免在 Python 或通用 PyTorch 层处理局部 Attention 输出与 LSE 值的融合；
* 必须在底层 C++/CUDA/Ascend C 层实现类似 `_npu_attention_update` 的原生融合算子：传入 `[O_local, LSE_local]`，内部完成跨卡 ReduceScatter/AllReduce 聚合，直接在片上 SRAM 中完成数值校准后输出最终结果。
