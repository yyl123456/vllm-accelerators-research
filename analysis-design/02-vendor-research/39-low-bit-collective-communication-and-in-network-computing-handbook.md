# 低精度量化集合通信、硬件网内计算 (SHARP/CCU) 与网络拥塞流控实战指南

> **调研与编写日期**：2026-09-30  
> **涉及核心源码基线**：
> - **Huawei Ascend**：`vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/`（`moe_v2_fullload_dynamic_quant.h`）、`hccl_types.h`（`HCCL_ACCELERATOR: CCU/AIV`）、`HCCL_DATA_TYPE_FP8E4M3/HIF8`
> - **Google TPU / OpenXLA**：`google/third_party/xla/xla/backends/gpu/tests/sub_byte_collectives.hlo`（Sub-byte 4-bit 规约/Bitcast 通信）
> - **NVIDIA (SHARP & NCCL)**：Mellanox/NVIDIA Quantum InfiniBand 硬件网内规约架构（SHARP）、`NCCL_NET_GDR_LEVEL`
> 
> **权威文档对应**：`20260903-vllm-ai-infra-research/` 之 `09-hardware-foundations/20260903-90-统一硬件模型与拓扑`、`10-ascend/20260903-100`、`14-decisions/20260903-140-141`。

---

## 1. 低比特量化通信 (Low-Bit Quantized Collectives)

在大模型跨机通信受限于物理网络带宽（如 400Gbps/800Gbps）时，将通信数据从 FP16/BF16（16-bit）压缩到 FP8（8-bit）甚至 FP4/INT4（4-bit），**可将物理传输体积直接压缩 50% ~ 75%**。

### 1.1 在线动态量化通信流水线
以华为昇腾在 MoE 融合通信中的代码实现为例（见 [`moe_v2_fullload_dynamic_quant.h:20-60`](../../huawei/third_party/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h#L20-L60)）：

```text
[本地高精度 Tensor (BF16)]
   │
   ▼ 1. 向量单元就地动态量化 (AscendC TQue Vector Engine)
   │  - 每行/每组计算绝对最大值: scale = max(|X|) / MaxQuantVal
   │  - 向量乘加归一化: X_quant = round(X / scale)
[低精度 Payload (FP8 / INT8)] + [轻量 Scale 向量 (FP32)]
   │
   ▼ 2. 压缩物理链路传输 (HCCS / RoCE 仅搬运 50% 字节量)
[对端网卡 DMA 接收]
   │
   ▼ 3. 硬件近端反量化或直接乘加 (Dequant Epilogue / Fused MMA)
[还原高精度或直接送入 Int8/FP8 Tensor Core 运算]
```

### 1.2 OpenXLA 中的 Sub-byte（4-bit / 8-bit）位转换优化
在 [`google/third_party/xla/xla/backends/gpu/tests/sub_byte_collectives.hlo:3-12`](../../google/third_party/xla/xla/backends/gpu/tests/sub_byte_collectives.hlo#L3-L12)：
* XLA 编译器在处理低于 1 字节的算子（如 `s4[4, 16]`、`f4e2m1fn[80, 20]`）时，不执行昂贵的数值类型提升（`convert`）；
* 而是通过 **`bitcast` 将两个 4-bit 元素静态拼接打包成一个标准的 `s8[4, 8]` 字节流**，直接调用底层硬件通信引擎；在对端接收后再通过无损 `bitcast` 还原回 4-bit 张量，实现极致的网络线速饱和。

---

## 2. 硬件网内计算与专用通信加速核

### 2.1 NVIDIA SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)
* **传统主机规约的缺陷**：在 Tree/Ring AllReduce 中，每张卡既要收发数据，又要消耗 GPU 显存带宽和计算核心去执行 `Sum/Max` 累加；
* **SHARP 网内计算机制**：
  * 将 AllReduce 规约算子直接下沉到 **Mellanox Quantum InfiniBand 交换机芯片内部（ASIC）**；
  * 各卡仅向交换机上报自己的局部张量；
  * **交换机芯片内的 ALU 硬件逻辑在数据穿过交换背板的瞬间完成加和规约**；
  * 交换机直接将最终结果多播（Multicast）回发给所有卡。
* **收益**：网络跳数直接减半，GPU 完全零算力消耗，通信延迟大幅降低。

### 2.2 华为昇腾 CCU (Collective Communication Unit) 专用硬件通信核
在 [`huawei/third_party/torch_npu/third_party/hccl/inc/hccl/hccl_types.h:95`](../../huawei/third_party/torch_npu/third_party/hccl/inc/hccl/hccl_types.h#L95)：
* HCCL 明确定义了硬件加速器枚举：
  ```c
  typedef enum {
      HCCL_ACCELERATOR,  // 0: default, 1: CCU, 2: AIV, 3: AICPU_TS...
  } HcclConfig;
  ```
* **CCU 硬件定位**：昇腾芯片上独立于 AI Core（Cube 矩阵核）与 AI Vector（向量核）之外的**硬线通信加速单元**。
* **职责**：专职负责板载 HCCS/RoCE 数据搬运过程中的原位校验、简单规约和描述符排队，**彻底解脱 AI Core，让计算核 100% 专注于大矩阵乘法**。

---

## 3. 大规模 AI 集群网络层流控与拥塞控制 (RoCEv2 / InfiniBand)

在千卡/万卡跨机大模型集群中，集合通信的崩溃或性能断崖往往不是算法代码的问题，而是底层物理网络发生了**拥塞崩溃（Congestion Collapse）**：

### 3.1 核心流控与拥塞防御三大支柱

| 机制名称 | 核心运作原理 | 解决的致命痛点 |
|---|---|---|
| **PFC (Priority Flow Control)** | 基于优先级的逐跳流控：当接收端交换机 Buffer 超过高水位线时，反向发送 Pause 帧暂停上游发送 | 避免物理丢包导致的昂贵 TCP/RDMA 重传等待 |
| **DCQCN 拥塞控制** | 结合 ECN（显式拥塞通知）与反向 CNP 报文：网络轻微拥塞时平滑降低发送端网卡速率 | 彻底防止 PFC 频繁触发引发的**“PFC 环路死锁与广播风暴”** |
| **Packet Spraying (数据包喷淋)** | 动态逐包轮询（Per-Packet ECMP）：将单次大流分散到全网所有等价链路条目上 | 避免传统基于五元组哈希（ECMP）导致的某些核心链路被打爆而其余链路空闲的**“链路极化（Hash Collision）”** |

---

## 4. 自研低精度网内计算与网络通信架构法则 (Blueprint)

若要在自研集群或通信库中引入低比特通信与网内硬件加速，必须遵循以下 **四大约束原则**：

```text
┌────────────────────────────────────────────────────────┐
│ 4. 随机舍入无偏估计：Stochastic Rounding 防梯度累积漂移      │
├────────────────────────────────────────────────────────┤
│ 3. 硬件自适应回退：无 SHARP/CCU 时无感降级为 GPU 本地规约    │
├────────────────────────────────────────────────────────┤
│ 2. 块级动态缩放 (Block-wise Scaling)：保持高动态范围与精度    │
├────────────────────────────────────────────────────────┤
│ 1. 偶数字节对齐保证：Sub-byte 必须本地 Bitcast 规整后发射     │
└────────────────────────────────────────────────────────┘
```

1. **数值下溢防御与动态缩放（Block-wise Scaling）**：
   在执行 FP8/FP4 通信前，切忌全张量使用全局单一 Scale。必须按 32 或 64 个元素为一组（Block/Tile）计算局部绝对最大值，将高频动态范围锁定在微块内，防止奇异值（Outliers）导致全量精度崩塌。
2. **随机舍入（Stochastic Rounding）保证无偏估计**：
   在 FP16 $\to$ FP8/INT8 量化阶段，不能使用简单的向最近取整（Round-to-Nearest），因为大量微小梯度的舍入误差单向累加会导致模型训练发散；必须引入硬件伪随机数生成器（PRNG）执行随机舍入，维持统计期望无偏。
3. **网内计算拓扑严格白名单探测**：
   在初始化阶段主动探测物理交换机是否支持网内硬件聚合（SHARP / In-Network Computing）。若交换机不支持或处于跨不同可用区（AZ）网络，立即通过软件 Fallback 到标准 Host-driven 环算法，避免由于固件不兼容引发静默数据错误。
