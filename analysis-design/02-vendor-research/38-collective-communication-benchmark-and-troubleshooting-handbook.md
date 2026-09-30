# 集合通信算子性能压测、拓扑基准与死锁故障诊断实战手册

> **调研与编写日期**：2026-09-30  
> **涉及核心源码基线**：
> - **NVIDIA NCCL**：`src/debug.cc`、`src/graph/`（环与树拓扑发现）、NCCL-Tests 带宽推导公式
> - **Huawei Ascend**：`torch_npu/profiler/`（MSPTI 算子追踪）、`hccl.h`（`HcclGetCommAsyncError`）、`collect_hccl_info.py`
> - **vLLM Benchmarks**：`benchmarks/kernels/benchmark_device_communicators.py`（CustomAllreduce vs SymmMem vs PyNccl）
> - **Tenstorrent TT-Metal**：`tests/ttnn/unit_tests/operations/ccl/`、BugChecker 规则 `ccl-ring-buffer-mismatch.md`
> 
> **权威文档对应**：`20260903-vllm-ai-infra-research/` 之 `10-ascend/20260903-100`、`11-qaic/20260903-116-QAIC版本Tuple、故障语义与验证`、`14-decisions/20260903-142`。

---

## 1. 集合通信性能基准评测：AlgBW 与 BusBW 数学折算体系

在评估集合通信算子优劣或压测集群硬件时，最核心的两个指标是 **算法带宽（AlgBW）** 与 **总线有效带宽（BusBW）**。两者有严格的物理与数学对应关系：

### 1.1 数学定义与物理意义
* **Payload 尺寸 $S$**：用户传入算子的实际数据字节数（例如一个形状为 `[4096, 8192]` 的 BF16 张量，$S = 4096 \times 8192 \times 2 = 64\text{MB}$）；
* **耗时 $t$**：通信算子端到端耗时（通常以微秒 $\mu s$ 或毫秒 $ms$ 计量）；
* **算法带宽（Algorithm Bandwidth, AlgBW）**：从用户算法视角衡量的吞吐：
  $$\text{AlgBW} = \frac{S}{t}$$
* **总线带宽（Bus Bandwidth, BusBW）**：反映物理硬件链路真实数据吞吐的能力（剔除算法本身的冗余系数）：
  $$\text{BusBW} = \text{AlgBW} \times \text{Factor}_{\text{collective}}$$

---

### 1.2 主流集合通信算子带宽折算因子矩阵

折算系数 $\text{Factor}$ 由物理传输步数推导得出（$P$ 为参与通信的卡数/Rank 数）：

| 算子名称 | 单卡实际物理搬运量 (Per Rank) | 折算系数公式 ($\text{Factor}$) | 8卡集群折算比 ($P=8$) | 极限折算比 ($P \to \infty$) |
|---|---|---|---|---|
| **AllReduce** | $2 \cdot \frac{P - 1}{P} \cdot S$ | $2 \cdot \frac{P - 1}{P}$ | $\mathbf{1.75\times}$ | $\mathbf{2.00\times}$ |
| **ReduceScatter** | $\frac{P - 1}{P} \cdot S$ | $\frac{P - 1}{P}$ | $\mathbf{0.875\times}$ | $\mathbf{1.00\times}$ |
| **AllGather** | $\frac{P - 1}{P} \cdot S$ | $\frac{P - 1}{P}$ | $\mathbf{0.875\times}$ | $\mathbf{1.00\times}$ |
| **AllToAll** | $\frac{P - 1}{P} \cdot S$ | $\frac{P - 1}{P}$ | $\mathbf{0.875\times}$ | $\mathbf{1.00\times}$ |
| **Broadcast** | $S$ (Root 发送) / $\frac{S}{P}$ | $\approx 1.0$ (树形多播) | $\mathbf{1.00\times}$ | $\mathbf{1.00\times}$ |

> **关键准则**：当评测自研通信算子时，若声称“AllReduce 达到了 350 GB/s 的 BusBW”，对应 8 卡机器上的物理单向总线利用率为 $350 / 1.75 = 200\text{ GB/s}$。评测报告必须显式区分两个指标，严禁混淆。

---

## 2. 集合通信死锁（Deadlock）与挂起（Hang）根因与诊断体系

在大规模分布式推理与训练中，集合通信挂起（Hang）是最隐蔽、排查代价最高的系统故障。其底层根因主要分为以下 4 类：

### 2.1 常见挂起根因剖析
1. **控制流分叉导致调用不一致（Rank Desynchronization）**：
   * 某一 Rank 满足特殊条件（例如命中缓存、遇到异常）跳过了集合通信调用，而其余所有 Rank 均进入了阻塞等待，导致整个拓扑死锁。
2. **环形缓冲区队列死锁（Circular Buffer Mismatch）**：
   * 常见于自研或自定义 Ring 算法（见 TT-Metal 规则 `ccl-ring-buffer-mismatch.md`）：Rank 0 等待 Rank 1 释放空间，Rank 1 等待 Rank 2，形成循环资源依赖（Dining Philosophers 问题）。
3. **硬件静默丢包或链路断连（Silent Link Degradation / Flapping）**：
   * RoCE 网卡 PFC（基于优先级的流控）发生死锁风暴，或者光模块产生误码丢包，硬件重传超时触发底层驱动阻塞。
4. **CUDA/NPU Stream 与 Event 交叉死锁**：
   * 错误地让通信流等待计算流的 Event，而计算流内部又依赖了尚未发射的通信结果。

---

### 2.2 跨厂商排查工具与看门狗（Watchdog）实践

#### 1. NVIDIA NCCL 诊断排查
* **环境变量日志定位**：
  在 [`nvidia/third_party/nccl/src/debug.cc:43-85`](../../nvidia/third_party/nccl/src/debug.cc#L43-L85)：
  ```bash
  # 开启全局调试级别与子系统
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=COLL,GRAPH,P2P,ENV
  # 开启死锁看门狗与超时报警 (默认超时通常为数分钟)
  export NCCL_COMM_BLOCKING=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  ```
* **NCCL Flight Recorder (Dump 分析)**：
  在崩溃或超时时，驱动自动向 `/tmp/nccl_trace_*` dump 导出每个 Channel 的最后执行步数（`connStepPtr`），快速揪出到底是哪一张卡掉队（Straggler Rank）。

#### 2. 华为昇腾 HCCL / MSPTI 诊断排查
* **异步状态轮询**：
  在 [`huawei/third_party/torch_npu/third_party/hccl/inc/hccl/hccl.h:413`](../../huawei/third_party/torch_npu/third_party/hccl/inc/hccl/hccl.h#L413)，通过 `HcclGetCommAsyncError(comm, &asyncError)` 获取异步硬件错误码（如 `HCCL_E_TIMEOUT`、`HCCL_E_ROCE_TRANSFER`）。
* **动态 Profiler 追踪**：
  利用 `torch_npu.profiler`（基于 MSPTI 接口），抓取 Timeline 时间线，清晰展示 AI Core 计算与 HCCS 通信流的时间跨度，直观定位等待间隙（Gap Bubble）。

---

## 3. 自研集合通信库故障防御与稳定性设计准则 (Blueprint)

若要自研一套高健壮性的工业级通信算子库，建议内置以下 **四项防御机制**：

```text
┌────────────────────────────────────────────────────────┐
│ 4. 拓扑与步数一致性校验：通信前握手全局递增 SeqNo           │
├────────────────────────────────────────────────────────┤
│ 3. 非侵入式 Flight Recorder：环形内存飞行记录仪 (Trace Dump)│
├────────────────────────────────────────────────────────┤
│ 2. 硬件异步超时看门狗：独立后台监控线程 (Watchdog Heartbeat)  │
├────────────────────────────────────────────────────────┤
│ 1. 软件防呆门禁：严格对称性编译期与运行期 Assert 检查         │
└────────────────────────────────────────────────────────┘
```

1. **全局执行序列号校验（Sequence Number Guard）**：
   在每次下发集合通信前，各卡在 CPU 侧或通过轻量网络交换当前操作的自增序列号（`op_seq_id`）。若发生分叉（如 Rank 0 正在调第 10 次，Rank 1 正在调第 11 次），立刻提前抛出清晰异常并退出，严禁进入底层死锁。
2. **环形内存飞行记录仪（In-Memory Flight Recorder）**：
   在共享内存中开辟一段固定大小（如 1MB）的循环缓冲区，无锁记录最后 1000 次通信调用的参数、数据量、时间戳和完成状态。当发生 Hang 故障时，由独立的看门狗线程将其 dump 成 JSON/Trace 文件供快速定位。
3. **分级优雅超时退出（Graceful Timeout & Abort）**：
   严禁底层 C++ 驱动使用死循环（`while(true)`）轮询硬件标志位。必须在循环内加入时间步计数器；超过阈值（如 30 秒）自动触发 `HcclCommDestroy` 或 `ncclCommAbort`，释放物理通道资源并向框架返回清晰的 `TIMEOUT` 错误码。
