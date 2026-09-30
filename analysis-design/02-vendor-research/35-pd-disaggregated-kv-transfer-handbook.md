# 大模型 P/D 分离部署与跨节点 KV Cache 高速通信实战指南

> **调研与编写日期**：2026-09-30  
> **涉及核心源码基线**：
> - **Huawei Ascend**：`vllm-ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`、`sfa_pd_rd2h/`、`ascend_store/`
> - **Qualcomm QAIC**：`vllm-qaic/distributed/kv_transfer/kv_connector/v1/qaic_connector.py`、`_kv_dma_handoff.py`
> - **vLLM Upstream & NVIDIA**：`vllm/distributed/kv_transfer/`（`KVConnectorBase_V1`）、`NIXL / LMCache / UCX` 接口
> - **Google TPU**：`features/DCN-Based_P-D_disaggregation.yml`、PJRT C ABI 跨 Slice 传输
> 
> **权威文档对应**：`20260903-vllm-ai-infra-research/` 之 `08-production/20260903-82-PD分离与KV传输`、`10-ascend/20260903-100`、`11-qaic/20260903-110`、`14-decisions/20260903-140-141`。

---

## 1. 核心业务痛点：为什么必须实施 P/D 分离部署？

在传统同节点部署（Colocated Serving）中，Prefill（首字计算）与 Decode（逐字生成）共享同一组计算卡，导致无法调和的系统矛盾：
1. **计算特征严重冲突（Compute vs Memory Bound）**：
   * **Prefill 阶段**：Compute-Bound，具有极高的计算强度（Compute Density），需要打满 Tensor Core / NSP / AI Core 算力；
   * **Decode 阶段**：Memory-Bound，算力需求极低，完全受限于显存带宽（HBM Bandwidth）；
   * 两者混跑会导致 Decode 生成时延（TPOT, Time Per Output Token）发生严重抖动。
2. **长文本传输墙（Transfer Wall）**：
   * 实施 P/D 分离后，Prefill 节点算出的完整 KV Cache（如 128K 上下文可达数 GB）必须跨卡/跨物理机在毫秒级时延内送达 Decode 节点。
   * **如果网络传输耗时大于重新 Prefill 的时间，P/D 分离便失去工程意义**。

---

## 2. 跨厂商主流 P/D KV 传输网络实现剖析

### 2.1 华为昇腾：三轨并行体系 (Mooncake / SFA RD2H / AscendStore)

在 [`vllm-ascend/distributed/kv_transfer/`](../../huawei/third_party/vllm-ascend/vllm_ascend/distributed/kv_transfer/) 中，昇腾根据网络拓扑和硬件介质设计了 3 套不同的传输连接器：

#### 1. Mooncake Connector (基于跨节点 RDMA 零拷贝引擎)
* **源码位置**：[`kv_p2p/mooncake_connector.py:25-78`](../../huawei/third_party/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L25-L78)。
* **机制**：
  * 基于 `mooncake.engine.TransferEngine`，建立跨节点 RoCEv2 直连；
  * **内存池注册（`RegisterRegions`）**：在启动时将 NPU 显存的物理 Block 区域通过驱动注册为 RDMA 内存键（Remote Key）；
  * **无 CPU 中继直传**：Prefill 节点产出 KV 后，通过控制面 ZMQ 握手交换目标 Decode 节点的物理 Block 虚拟地址，网卡通过 RDMA Write 直接将数据写入 Decode 节点的 NPU 显存，实现真正的端到端**零拷贝（Zero-Copy）**。

#### 2. SFA RD2H (Remote Device to Host 异步流水线)
* **源码位置**：[`kv_p2p/sfa_pd_rd2h/connector.py`](../../huawei/third_party/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/connector.py)。
* **适用场景**：当 Decode 节点显存极其紧张、无法一次性接收全量 KV 时，数据流经 Host CPU 共享内存做二级缓冲，分层级流水写入 NPU。

#### 3. AscendStore (多层池化分布式缓存)
* **源码位置**：[`kv_pool/ascend_store/ascend_store_connector.py`](../../huawei/third_party/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py)。
* **特点**：提供 `attention_fence` 状态栅栏，支持将多轮对话的公用 Prefix KV Cache 沉降到中央分布式内存池中，支持多个 Decode 节点并发订阅。

---

### 2.2 高通 QAIC：基于共享内存与专属物理 DMA Handoff

在 [`vllm-qaic/distributed/kv_transfer/kv_connector/v1/qaic_connector.py:19-58`](../../qualcomm/third_party/vllm-qaic/vllm_qaic/distributed/kv_transfer/kv_connector/v1/qaic_connector.py#L19-L58)：

#### 1. 跨进程共享内存通道（PSM）
* 高通为单机多卡/多卡箱部署设计了专有的共享内存命名规范：
  ```text
  Format: psm_{uuid16}_{counter}_{pid}
  Example: psm_a3f9c2b1e4d78956_42_4d2
  ```
* 利用 POSIX Shared Memory 机制，绕过 Docker 容器网络壁垒，实现进程间微秒级通信。

#### 2. 槽位级 DMA 直传 (`_kv_dma_handoff.py`)
* 回顾高通 AOT 模式的 `batch_index` 槽位机制：Prefill 产物并不按细粒度 Block 离散存放，而是连续的整段 Context。
* **Handoff 机制**：Prefill 节点直接向底层驱动提交 DMA 搬运指令，将该槽位的连续 Buffer 映射至 PCIe P2P 总线，瞬间镜像复制给 Decode 节点的对应物理 Slot，开销极低。

---

### 2.3 Google TPU：DCN (Data Center Network) 跨 Slice 传输

在 [`google/third_party/tpu-inference/.buildkite/features/DCN-Based_P-D_disaggregation.yml`](../../google/third_party/tpu-inference/.buildkite/features/DCN-Based_P-D_disaggregation.yml)：
* **拓扑分级**：TPU 内部通过光路互联（ICI, Optical Circuit Switch）走 3D Torus，跨 Pod/跨机架则走数据中心网络（DCN）；
* **P/D 分离策略**：在超大规模集群中，Prefill 在大型高算力 TPU Pod（如 v5p-512）上集中执行，通过 DCN 异步多播管道下发给分布式 Decode 边缘节点。

---

## 3. 自研 P/D 跨节点通信架构实战设计指南 (Blueprint)

若要自研一套支撑超低延迟 P/D 分离的高速 KV 传输子系统，必须遵循以下 **四项核心原则**：

```text
┌────────────────────────────────────────────────────────┐
│ 4. 控制与数据解耦：ZMQ/gRPC 极速元数据握手 (仅几十字节)       │
├────────────────────────────────────────────────────────┤
│ 3. 内存直接注册：驱动级 RDMA / GPUDirect 内存预打桩 (MR Pool) │
├────────────────────────────────────────────────────────┤
│ 2. 算子流水重叠：Layer-wise 分层逐层抢先发送 (Early Handoff)  │
├────────────────────────────────────────────────────────┤
│ 1. 物理拓扑亲和：单机内 PCIe/NVLink 共享内存 + 跨机 RoCE 零拷贝│
└────────────────────────────────────────────────────────┘
```

### 1. 算子流水重叠：逐层抢先发射 (Layer-wise Streaming)
* **禁止全模型推理完才发送**：
  若等待 64 层 Transformer 全部算完才发起传输，网络将经历漫长的空闲期，并在瞬间爆发流量洪峰。
* **推荐做法（Layer-wise Handoff）**：
  在 Prefill 节点执行完第 0 层的 Attention 时，**立即触发后台异步流将第 0 层的 KV Cache 注入 RDMA 管道发送**。当全模型第 63 层计算完毕时，前 62 层的 KV 数据早已抵达 Decode 节点显存，实现近乎零等待的瞬时接力。

### 2. 驱动级 RDMA 预打桩（Pinned Memory Buffer Pool）
* **拒绝运行期临时注册**：
  在 Linux/Infiniband 中，调用 `ibv_reg_mr` 锁定显存/内存物理页的开销极高（单次毫秒级）。
* **静态预注册**：
  在服务初始化阶段，预先将整个 KV Cache 显存池全部注册进 RDMA 网卡硬件映射表，运行期间仅传递内存偏移量（Offsets）与长度（Lengths），做到纯零拷贝直接内存写入。

### 3. 控制面与数据面彻底解耦
* **控制面（Metadata Plane）**：基于轻量级 ZMQ 或自研无锁共享内存队列，只负责交换请求 ID、Token 长度及物理块映射表（数百字节，微秒级）；
* **数据面（Data Plane）**：数据流完全脱离 CPU 用户空间，由网卡硬件 DMA 控制器全速搬运。
