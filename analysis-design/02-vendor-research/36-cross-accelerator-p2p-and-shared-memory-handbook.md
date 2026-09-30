# 异构加速卡 P2P 点对点通信与硬件共享内存设计实战手册

> **调研与编写日期**：2026-09-30  
> **涉及核心源码基线**：
> - **NVIDIA CUDA IPC**：`third_party/vllm/vllm/distributed/weight_transfer/ipc_engine.py`、`rebuild_cuda_tensor`、CUDA Driver API（`cuMemExportToShareableHandle`）
> - **Huawei Ascend NPU IPC**：`vllm-ascend/distributed/weight_transfer/npu_ipc_engine.py`、`packed_npu_ipc_consumer/producer`、`ASCEND_RT_VISIBLE_DEVICES` 物理芯片映射
> - **Tenstorrent NoC P2P**：`tenstorrent/third_party/tt-metal/`、`ccl_command.hpp`（`noc_transfer_info`）、片上 SRAM 原生直读写
> 
> **权威文档对应**：`20260903-vllm-ai-infra-research/` 之 `09-hardware-foundations/20260903-90-统一硬件模型与拓扑`、`10-ascend/20260903-100`、`14-decisions/20260903-140-141`。

---

## 1. 核心问题背景：为什么集合通信底层必须依赖 P2P 与硬件共享内存？

在大模型训练、RLHF（强化学习在线对齐）以及高并发 Serving 场景中，许多关键链路（如权重热同步 Weight Transfer、单机多卡 KV 交换、多进程并行分发）如果走传统的网络套接字（Socket）或通用集合通信规约，会面临严重开销：
1. **多重内存拷贝（Double-Copy Overhead）**：数据从 GPU/NPU 显存拷贝到 Host 内存，再通过 Linux IPC 拷贝到对端进程，最后再搬上卡；
2. **内核态上下文切换（Context Switch）**：传统 IPC 依赖系统调用和页缓存。
3. **物理拓扑孤岛（Topology Blindness）**：在未探测硬件拓扑的情况下跨 NUMA 节点传输，导致 PCIe 饱和与 CPU 内存总线锁死。

因此，**利用硬件总线（NVLink / HCCS / NoC）的 P2P（Peer-to-Peer）直接互联，并通过驱动级 IPC 句柄实现“零拷贝跨进程共享内存”，是实现微秒级通信的终极底座**。

---

## 2. 主流芯片架构的 P2P 共享内存底层实现拆解

### 2.1 NVIDIA：CUDA IPC 与统一虚拟寻址 (UVA)

#### 1. 底层驱动机制
* **UVA（Unified Virtual Addressing）**：为所有 CPU 和 GPU 统一分配 64 位全局虚拟地址空间；
* **内存句柄导出导入（Driver API）**：
  1. **生产者（Producer）**：调用 `cudaIpcGetMemHandle()`（底层基于 `cuMemExportToShareableHandle`），将一块设备显存导出为一个轻量的 64 字节 `cudaIpcMemHandle_t`；
  2. **跨进程传递**：将该 Handle（通常是十六进制或 Base64 字符串）通过轻量控制通道传给消费者；
  3. **消费者（Consumer）**：调用 `cudaIpcOpenMemHandle()` 将该 Handle 导入，映射为本地进程可直接指针寻址的 GPU 显存指针，**实现真正的跨进程零拷贝直读直写**。

#### 2. vLLM 中的工程打包优化（`ipc_engine.py`）
在 [`vllm/distributed/weight_transfer/ipc_engine.py:28-79`](../../third_party/vllm/vllm/distributed/weight_transfer/ipc_engine.py#L28-L79)：
* **Packed Buffer 打包传输**：如果一个大模型有几百个小权重 Tensor，导出几百个 IPC Handle 会带来巨大的解析与所有权管理开销；
* vLLM 设计了 `PackedBufferImporter`：生产者在显存中预分配一块连续的巨型打包缓冲区（`DEFAULT_PACKED_BUFFER_SIZE_BYTES`，如 2GB），将所有小张量排布其中，**全局仅导出一个 IPC Handle**，消费者根据 `tensor_sizes` 偏移量进行微秒级切片挂载。

---

### 2.2 华为昇腾：HCCS 总线与 NPU IPC 跨进程共享引擎

在 [`vllm-ascend/distributed/weight_transfer/npu_ipc_engine.py:63-120`](../../huawei/third_party/vllm-ascend/vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py#L63-L120)：

#### 1. 物理芯片 ID 与逻辑设备索引解耦
* 在昇腾集群中，每个进程看到的 `torch.accelerator.current_device_index()` 是逻辑卡号（0~N）；
* 昇腾 NPU IPC 核心要求**同一物理芯片才能无锁互通**；
* 源码实现了 `npu_generate_uuid`：
  ```python
  physical_chip_id = get_physical_chip_id(logical_device)
  return f"{get_ip()}-{physical_chip_id}"
  ```
  结合环境变量 `ASCEND_RT_VISIBLE_DEVICES` 进行映射，确保同机共驻留的训练进程与推理进程精准匹配到同一个物理卡 UUID。

#### 2. NPU 序列化重构（`reduce_tensor`）
* 利用 PyTorch 与昇腾私有驱动打通的 `torch.multiprocessing.reductions.reduce_tensor`；
* 导出 NPU 板载内存的 IPC 共享描述符，在消费者侧无感重构出真实的 `torch.Tensor` 对象，整个过程数据完全保留在昇腾 HBM 显存内部，零 Host 内存搬运。

---

### 2.3 Tenstorrent：NoC 片上直接寻址与网格 P2P 穿透

Tenstorrent 的设计更为激进——**硬件在物理层面就是一个统一编址的 2D 晶圆网格网络（Network-on-Chip, NoC）**：
* **全局物理坐标寻址（Physical Coords）**：
  在 [`tenstorrent/third_party/tt-metal/ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp:62-68`](../../tenstorrent/third_party/tt-metal/ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp#L62-L68)：
  ```cpp
  struct noc_transfer_info {
      uint64_t noc_addr;
      size_t noc_transfer_size_bytes;
  };
  ```
  `noc_addr` 直接编码了目标芯片上的 `(X, Y)` 物理网格坐标以及该 Tensix 核心的 L1 SRAM 偏移地址。
* **无需软件 IPC 句柄**：芯片内的任意计算核或数据搬运核可以直接向对端核的 `noc_addr` 发起硬件级别的原子写，彻底抹除了传统 GPU/CPU 复杂的虚拟页表映射和 IPC 句柄握手流程。

---

## 3. 自研跨加速卡通用 P2P 共享内存与通信架构法则 (Blueprint)

若要设计一套统一兼容 NVIDIA、华为昇腾、Tenstorrent 及其他自研 AI 芯片的通用 P2P 共享内存子系统，推荐采纳以下 **四大自研架构法则**：

```text
┌────────────────────────────────────────────────────────┐
│ 4. 连续打包：巨型 Packed Buffer 规整 (消除碎片 Handle)    │
├────────────────────────────────────────────────────────┤
│ 3. 物理对齐：物理 UUID 唯一寻址 (解决多容器逻辑卡号冲突)   │
├────────────────────────────────────────────────────────┤
│ 2. 句柄闭环：Export/Import 状态生命周期追踪 (避免野指针)   │
├────────────────────────────────────────────────────────┤
│ 1. 拓扑探测：CanAccessPeer 物理连通性矩阵白名单           │
└────────────────────────────────────────────────────────┘
```

### 1. 严格的拓扑连通性探测（Topology Pre-flight Check）
* 在尝试打开 P2P 之前，必须在启动期调用硬件 API（如 `cudaDeviceCanAccessPeer` 或 CANN 拓扑探测接口）；
* 只有位于同一 NVLink / HCCS 域内，或处于同一 PCIe 根复合体（Root Complex / PCIe Switch）下的卡对，才允许激活 P2P 路径；
* 跨 NUMA 或无硬件直连通道的节点，必须自动降级为跨机 RDMA 或 Socket，严禁强开 P2P 导致总线锁死。

### 2. 物理 UUID 寻址规范（Physical ID Normalization）
* 在 Kubernetes/Docker 容器化部署中，每个 Pod 的 `CUDA_VISIBLE_DEVICES` 或 `ASCEND_RT_VISIBLE_DEVICES` 往往都被映射为 `0, 1`；
* 通信库严禁使用“逻辑卡号”作为握手路由键；
* 必须统一采用 `Host_IP + Physical_Bus_ID / Chip_UUID` 组合作为全集群唯一的硬件身份标识。

### 3. Packed Buffer 聚合搬运机制
* 绝不为细碎 Tensor（如几十 KB 的 Bias 或 Norm 权重）频繁创建 IPC Handle；
* 维护固定容量（如 1GB ~ 2GB）的常驻物理环形页，所有需要 P2P 传输的数据统一 memcpy 拼接，仅导出单张元数据清单与 1 个主 Handle，极大降低通信控制面开销。

### 4. 显式的跨进程生命周期栅栏（Lifecycle Memory Fence）
* 消费者在导入 Handle 并使用期间，生产者的原始物理内存**绝不可释放或被 Allocator 重新分配**；
* 必须引入握手三段式：
  $$\text{Producer Export} \longrightarrow \text{Consumer Import \& Compute} \longrightarrow \text{Consumer Ack Release}$$
  只有当收到消费者的释放确认消息后，生产者才可归还或复用该物理 Buffer。
