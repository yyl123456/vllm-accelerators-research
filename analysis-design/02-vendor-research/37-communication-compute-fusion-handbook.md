# 通信计算融合 (Communication-Compute Fusion) 算子自研设计与优化手册

> **调研与编写日期**：2026-09-30  
> **涉及核心源码基线**：
> - **NVIDIA CUTLASS / vLLM**：`third_party/vllm/csrc/libtorch_stable/cutlass_extensions/vllm_collective_builder.cuh`、CUTLASS 3.x Hopper TMA & Cluster 异步传输
> - **Huawei Ascend CANN**：`vllm-ascend/csrc/gmm/`（Grouped Matmul）、`csrc/mc2/`（`Mc2CcTilingConfig`、AI Core/Vector Core 流水编排）
> - **Google TPU / OpenXLA**：`xla/backends/gpu/transforms/collectives/collective_fusion.cc`、`collective_emitter.cc`、HLO 算子级图融合
> 
> **权威文档对应**：`20260903-vllm-ai-infra-research/` 之 `05-vllm-compile/20260903-51-Inductor与CustomOp协同`、`10-ascend/20260903-100`、`14-decisions/20260903-140-141`。

---

## 1. 核心技术痛点：传统 Stream 异步通信的局限性

在传统的大模型并行优化中，最常见的通信重叠做法是使用 **双 CUDA/NPU Stream**：
```python
# 传统粗粒度 Stream 异步
with torch.cuda.stream(comm_stream):
    dist.all_gather(gathered_x, x)
# 宿主机必须在此等待或通过 Event 强行同步
torch.cuda.current_stream().wait_stream(comm_stream)
out = torch.matmul(gathered_x, weight)
```
**这种粗粒度 Stream 异步存在三大不可忽视的硬伤**：
1. **显存放大（Memory Bloat）**：必须为 `gathered_x` 分配完整的中间临时显存（可达数百 MB），显存压力剧增；
2. **GPU/NPU SM 利用率断崖**：通信未完成前，矩阵乘法的计算单元（Tensor Core / Cube Core）完全处于饥饿空闲状态；
3. **驱动发射延迟（Launch Overhead）**：每次启动 Kernel 都有微秒级 CPU 下发时延。

**终极解法：内核级通信计算深度融合（Kernel-Level Communication-Compute Fusion）**。即把 AllGather / ReduceScatter 与 GEMM 矩阵乘法合并进同一个硬件核函数中，**边收数据、边算矩阵乘、边发结果**！

---

## 2. 业界两大融合范式深度剖析

### 2.1 范式 A：AllGather-GEMM 融合（以 TP 前向传播为例）

#### 1. 传统分离执行 vs 内核融合流转
* **传统分离**：所有卡通过 Ring/Tree AllGather 将输入序列 $X_i$ 完整汇聚成 $X = [X_0, X_1, \dots, X_{P-1}]$，然后再调用 GEMM 计算 $Y = X \times W$。
* **AllGather-GEMM 内核融合**：
  * 将大矩阵权重 $W$ 沿行或列划分为与卡数 $P$ 对应的 $P$ 个分块；
  * **Step 0（本地零通信先算）**：当网络引擎还在向对端传输数据时，计算核心直接读取本地已有的 $X_{\text{local}}$，立刻启动第一块矩阵乘法 $Y_0 = X_{\text{local}} \times W_0$；
  * **Step 1 ~ P-1（流水接力）**：网络引擎异步通过 P2P/NVLink 将相邻卡的 $X_{\text{peer}}$ 搬入片上共享内存（Shared Memory / L1 SRAM）。计算核一检测到信号量翻转，立刻无缝切入计算下一块 $Y_k = X_{\text{peer}} \times W_k$；
  * **显存节约**：片上仅需一块极小的双缓冲切片（Ping-Pong Tile），**全局中间临时张量显存开销直接降为 0**！

---

### 2.2 范式 B：GEMM-ReduceScatter 融合（以 TP 输出投影为例）

#### 1. 内核融合流转与 Epilogue 劫持
* **传统分离**：本地先算出完整的 GEMM 结果 $Y_{\text{local}} = X \times W_{\text{local}}$，写回全局 HBM 显存；再启动 ReduceScatter 算子从显存读取并切片规约。
* **GEMM-ReduceScatter 融合（Epilogue Fusion）**：
  * 在 CUTLASS 3.x（见 `vllm_collective_builder.cuh`）或 Ascend C 的 GEMM 尾声阶段（Epilogue），**计算结果根本不落盘写回全局显存**；
  * 处于寄存器中的累加器（Accumulator Registers）在执行完缩放（Bias/Scale/Activation）后，直接通过 DMA 写入对端卡的环形接收 Buffer 中；
  * 对端卡的计算核在同一循环中将传入的切片与本地累加器原位累加，直接产出最终分块输出。

---

## 3. 编译器自动融合机制：OpenXLA 的 `CollectiveFusion` Pass

在 [`google/third_party/xla/xla/backends/gpu/transforms/collectives/collective_fusion.cc:53-85`](../../google/third_party/xla/xla/backends/gpu/transforms/collectives/collective_fusion.cc#L53-L85)，Google OpenXLA 展现了如何在编译器计算图层面自动化执行通信融合：

1. **图模式匹配（Graph Pattern Matching）**：
   * 编译器自动扫描 HLO 计算图，识别形如 `AllReduce(Dot(A, B))` 或 `Dot(AllGather(A), B)` 的拓扑子图。
2. **重写为单核指令（`CreateCollectiveFusionInstruction`）**：
   * 突破常规算子无法融合副作用操作（Side-effect）的限制；
   * 自动生成包含通信与矩阵乘的嵌入计算图（Embedded Computation `fused_computation`），交由 `collective_emitter.cc` 直接生成针对特定硬件架构（如 Hopper TMA、TPU ICI）的原生融合汇编。

---

## 4. 自研通信计算融合算子设计指南 (Blueprint)

若要为专有 AI 芯片自研通信与计算融合内核，必须严格遵守以下 **四大工程设计法则**：

```text
┌────────────────────────────────────────────────────────┐
│ 4. 死锁主动防御：环形资源对称锁定与严格保序下发               │
├────────────────────────────────────────────────────────┤
│ 3. 寄存器与片上内存复用：Accumulator 原生直发与零显存驻留      │
├────────────────────────────────────────────────────────┤
│ 2. 硬件双缓冲流水编排：Ping-Pong 信号量与 Warp 角色专业化分工   │
├────────────────────────────────────────────────────────┤
│ 1. Tiling 粒度对齐：通信 Chunk 尺寸必须精确整除 GEMM Block Tile│
└────────────────────────────────────────────────────────┘
```

### 1. Tiling 粒度严格几何对齐（Geometry Alignment）
* **禁止通信与计算采用不同粒度**：
  * 设 GEMM 计算核的单次分块为 $[M_{\text{tile}}, N_{\text{tile}}, K_{\text{tile}}]$；
  * 每一个通信数据包（Network Packet / DMA Burst）的有效载荷大小，必须是 $M_{\text{tile}} \times K_{\text{tile}} \times \text{sizeof(dtype)}$ 的整数倍；
  * 确保数据每到达一个完整包，计算核心无需等待后续拼包，立刻就能满载喂饱乘加单元（MMA / Cube）。

### 2. 线程角色专业化分工（Warp Specialization）
* 在 Hopper / Blackhole / 昇腾 等现代芯片上，切忌让同一个计算线程同时负责网络通信与数学乘加；
* 必须划分 **计算组（Compute Warps）** 与 **通信/搬运组（DMA / Transfer Warps）**：
  * 通信组专职负责轮询标志位、发射 NoC/DMA 指令、管理双缓冲换页；
  * 计算组专职负责从片上 SRAM 加载并执行 Tensor Core 乘加，两者通过硬件屏障（Hardware Barrier / Mbarrier）异步通信，杜绝计算流被通信轮询分支（Branch Divergence）拖慢。

### 3. Epilogue 寄存器级发射（Register-Direct Transfer）
* 在矩阵乘法的 Epilogue 阶段，乘加累加器通常存放在 FP32/FP16 物理寄存器堆中；
* 融合内核应直接调用芯片专有指令（如 NVLink Direct Write 或 Ascend UB 搬运），直接把寄存器数据投递至网卡发送队列，彻底省去中间局部显存的写入与二次加载。

### 4. 环形通信死锁规避（Deadlock Avoidance）
* 在多卡同时执行 AllGather/ReduceScatter 融合算子时，若所有卡都在等待对端释放 Buffer，极易引发死锁；
* 必须在软件层面保证**先发射本地已知计算，后进入通信流水**；且所有 Rank 必须严格遵循全局相同的单向环推进次序（`Rank i -> Rank (i+1)%P`），确保依赖图无环。
