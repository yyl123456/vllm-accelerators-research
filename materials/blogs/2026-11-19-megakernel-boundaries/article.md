# MegaKernel 不是万金油：它赢在 Decode 的缝隙，也会输在 Prefill、动态调度和多节点通信

作者：DeepPrompting
发布时间：2026-10-17T11:08+00:00
来源：https://mp.weixin.qq.com/s/Lj8HkN-yy4jc8tWlEgCJJw

>

把更多算子塞进一个 Kernel，确实可以减少 Launch、同步和 HBM 往返。但 Kernel Boundary 不只是开销，也提供资源专用化、故障隔离、调度自由和成熟 Library Kernel。消除边界，等于接管这些责任。

过去两年，MegaKernel 成了 GPU 推理系统里最容易制造兴奋感的词之一。

它的直觉非常诱人：既然一个 Decode Step 要启动几百甚至上千个 Kernel，为什么不把它们放进一个长期驻留的 Kernel？数据留在 Register 和 Shared Memory，GPU 自己调度 Task，Host 不再逐个 Launch，中间结果也不用反复写回 HBM。

这个方向是真实有效的。问题在于，很多讨论悄悄完成了三次外推：

1.

从“小 Batch Decode 有效”外推到“Prefill 也有效”。

1.

从“减少 Kernel Boundary”外推到“端到端 Serving 更快”。

1.

从“单卡或单节点成功”外推到“多租户、多节点和动态业务也成功”。

公开证据并不支持这些无条件外推。

Ada-MK 的生产方案明确让 TensorRT-LLM 继续执行 Prefill，只在 Decode 切换到 MegaKernel；Event Tensor 的同一套框架中，动态调度在规则 TP 负载下只有其单 Launch、阶段间全局 Barrier 基线的 **0.82–0.89 倍相对性能**；一个多节点 MoE MegaKernel 在固定每 GPU 工作量的 Weak-scaling 实验中，8 节点延迟最多达到单节点归一化理想值的约 **10×**。[1][2][5]

这篇文章不是要证明 MegaKernel 没用，而是回答三个更有工程价值的问题：

-

它究竟适合什么？

-

哪些场景不适合？

-

它把原来的成本转移到了哪里？

>

**结论先行：MegaKernel 是高收益、高约束的专用执行路径，不是默认终局。最合理的系统通常是 CUDA Graph、成熟 Library Kernel、局部 Fusion Island 和少量 Persistent Scheduling 的组合。**

---

## 一、先把四个概念拆开

很多争论没有意义，是因为大家把 Kernel Fusion、Persistent Kernel、MegaKernel 和 CUDA Graph 当成一件事。

*图 1：四种执行模型优化不同成本。MegaKernel 的增量价值，必须相对 CUDA Graph 和局部 Fusion 证明，而不是只和 Eager Launch 比。*

### 1. Kernel Fusion

把相邻的 Producer、Elementwise、Reduction 或 Epilogue 合进一个 Kernel，主要收益是减少中间张量物化和少量 Launch。

FlashAttention 更接近“IO-aware Algorithm + 深度融合 Kernel”：它重写 Attention 的分块和 Online Softmax，避免完整 Attention Matrix 落入 HBM。它的成功不等于“任意控制流都应该融合”。[9][10]

### 2. Persistent Kernel

启动一组长期驻留的 CTA 或 Warp，让它们循环消费 Task，而不是每项工作重新 Launch。

关键是生命周期和取任务方式，不是融合范围。一个 Persistent Kernel 可以只执行一种算子。

### 3. MegaKernel

本文把 MegaKernel 定义为：

>

**跨越多个逻辑阶段、算子类型或动态任务类别，并在设备端承担 Task Dispatch、依赖追踪、同步或状态管理的执行体。**

它不只是更大的 CUDA Function，而是在 Kernel 内放进一个小型 Runtime。

### 4. CUDA Graph

CUDA Graph 保留多个 Kernel，但把创建、依赖和重复提交从每轮执行中移走。它能显著降低 Host 提交成本，同时保留每个 Kernel 独立的 Thread Block、Register、Shared Memory 和专用实现。[6][7]

所以正确的 Benchmark 不应是：

```
Eager 多 Kernel vs. MegaKernel

```

而应是：

```
Eager
vs. CUDA Graph
vs. 局部 Fusion + CUDA Graph
vs. Persistent / MegaKernel

```

否则，一个 Graph Replay 就能解决的收益，会被错误记到 MegaKernel 名下。

---

## 二、MegaKernel 的经济账：少了什么，又多了什么

多 Kernel 路径可以抽象为：

Tmulti = Thost + Tlaunch + Tcompute + Tmaterialize + Tsync - Toverlap

MegaKernel 路径则变成：

Tmega = Tuseful + Tdevice scheduler + Tresource penalty + Timbalance + Tsync + Tresidual memory

它能节省：

-

Host Launch 和设备调度间隙。

-

跨 Kernel 的中间 HBM 写回与读取。

-

粗粒度全局同步。

-

一部分 CPU 调度开销。

-

Tile 级计算与通信的重叠机会。

它也会新增：

-

设备端 Queue、Atomic、Polling 和 Fence。

-

更长的 Live Range、更高 Register 压力和 Spill。

-

Shared Memory 对 Pipeline Depth 的约束。

-

不同阶段被迫共享 Launch Geometry 和资源预算。

-

动态 Task 的负载不均衡与 Wave Tail。

-

更大的代码体积、分支和 Instruction Fetch 压力。

-

编译、Autotuning、验证、可观测性和移植成本。

-

长驻留 Kernel 对多租户公平性和抢占的影响。

净收益是：

ΔT = Glaunch + Gmaterialize + Glocality + Goverlap - Cscheduler - Cresource - Cimbalance - Cengineering

只有在目标 Workload、硬件和并发条件下 `ΔT > 0`，它才值得采用。

---

## 三、第一个边界：Decode 适合，不等于 Prefill 也适合

MegaKernel 最理想的战场通常是小 Batch Decode：

-

每次只生成少量 Token。

-

GEMV 和小算子多。

-

Launch 与 Kernel 间 Gap 占比高。

-

权重与 KV Cache 访问更偏 Memory-bound。

-

单算子很难独立吃满 GPU。

Prefill 则常常相反：Token 多、矩阵大、Arithmetic Intensity 高，成熟 GEMM 和 Attention Kernel 已能充分利用 Tensor Core。此时 Launch 只占很小比例，牺牲专用 Kernel 的代价可能大于融合收益。

### 支持 MegaKernel 的团队，自己也做了阶段切分

Ada-MK 在 NVIDIA L20 上服务 Qwen3-1.7B 和 Qwen2.5-1.5B。它报告小 Batch、短序列场景相对原生 TensorRT-LLM 最高 **23.6%** 的吞吐提升；但生产架构没有让 MegaKernel 接管整个推理过程，而是：[1]

-

Prefill：继续使用 TensorRT-LLM 原生 Fused Operators。

-

Decode：切换到 MegaKernel。

-

Embedding、KV Cache 管理和业务逻辑：继续复用原 Runtime。

![Ada-MK Prefill 与 Decode 边界](https://mmbiz.qpic.cn/mmbiz_png/J8mNWJkqUePdlTuZBOzSGS9OmuNvicacO9WcskUKo9g88KC9ShbT1dfhD7ialW9Nzd2MPbZVh4BPrFfNdmOqyauLkG3jNmn3AyXnG8E0Qgv20/640?wx_fmt=png&from=appmsg)

*图 2：Ada-MK 论文第 6 页。作者明确写道 MegaKernel 在 Decode 有显著优势，但在 Prefill 的大规模并行计算中落后于成熟 TensorRT-LLM，因此采用 Phase-adaptive Hybrid Engine。来源：Dong et al., CC BY 4.0，页面裁剪。[1]*

作者给出的原因也很直接：

-

Decode 偏 IO-bound，融合更容易隐藏访存并减少 Launch。

-

Prefill 偏 Compute-bound，大规模计算已经接近饱和。

-

作者在 Qwen2.5-1.5B、TensorRT-LLM 的 Nsight Profile 中记录到 1655550 次 Launch、累计约 3.3 秒，并估计 Launch Time 占该次端到端推理的 **14.6%**；Prefill 是一次非自回归 Forward，不是“只启动一个 Kernel”，但 Launch 成本相对大规模计算更容易被摊薄。

这不是反对 MegaKernel，而是最有力的适用性证据：

>

**同一个模型、同一块 GPU、同一个系统，Prefill 和 Decode 的最优执行机制都可能不同。**

### 大 Batch 时，系统调度可能重新反超

Ada-MK 在 CSL 数据集上，Batch 1–8 领先；到 Batch 16，vLLM 反而比 Ada-MK 快 **3.5%**。作者将其归因于高并发长上下文下系统调度优势开始显现。[1]

这个数字只适用于论文的 L20、Qwen3-1.7B、GPTQ-W4A16 和 CSL Offline Fixed-batch 配置，不能外推为“Batch 16 是通用拐点”。作者将结果解释为高并发长上下文下系统调度优势开始显现，但论文没有用 Scheduler Ablation 单独验证这个因果关系。它与下面的解释一致：

>

**随着 Batch、序列长度和并行度增加，Launch Saving 的边际价值会下降，Batching、成熟 GEMM 与 Runtime Scheduling 的价值会上升。**

---

## 四、第二个边界：动态调度不是免费的

MegaKernel 经常把“CPU 调度”换成“GPU 内部调度”。但调度没有消失，它变成了：

-

Global-memory Queue。

-

Atomic Push / Pop。

-

Event Counter。

-

Spin Wait。

-

Runtime Dispatch Branch。

-

Work Stealing 与终止检测。

Event Tensor 是一项认真解决动态 MegaKernel 问题的工作。它同时支持 Static Scheduler 和 Dynamic Scheduler，也因此给出了一组很有价值的对照实验。[2]

![Event Tensor 调度权衡](https://mmbiz.qpic.cn/sz_mmbiz_png/J8mNWJkqUeMGPadMWjLNKbiaIRiakqw31xBHUnqLMgbd7XYEH0ROQ1054bqfaxcJkVGNwXRCMiahVDeE94lttlfeCrJPmoianM6qJCsf2t6Y6EI/640?wx_fmt=png&from=appmsg)

*图 3：Event Tensor Tables 2–3。在不规则 MoE 中，Dynamic Scheduler 随 Token 数增加可以胜过 Static；在规则 Qwen3-32B TP=4 中，Dynamic 只有 unfused-megakernel 基线的 0.82–0.89 倍相对性能，而 Static 为 1.06–1.09 倍。该基线仍是单次 Persistent-kernel Launch，只用一个 Event 在阶段间模拟全局 Barrier，并非 Eager/CUDA Graph 多 Kernel。来源：Jin et al.，页面裁剪；本裁剪图按 CC BY-SA 4.0 再许可。[2]*

结果不是“动态调度不好”，而是明确的 Crossover：

|

Workload

|

Static

|

Dynamic

|

结论

|

MoE，1 Token

|

1.03×

|

0.95×

|

Task 太少，Queue 开销不划算

|

MoE，1024 Tokens

|

1.04×

|

1.08×

|

不规则路由的负载均衡开始覆盖调度成本

|

Qwen3-32B TP=4，Batch 1

|

1.09×

|

0.83×

|

相对性能低 17%；对应耗时约长 20.5%

|

Qwen3-32B TP=4，Batch 16

|

1.06×

|

0.82×

|

相对性能低 18%；对应耗时约长 22.0%

这组实验击中了 MegaKernel 最常见的误区：

>

**把 Scheduler 搬到 GPU，不代表 Scheduler 成本变成零。**

当 Task 可预测时，预先计算的静态队列更便宜；当任务完成时间不可预测时，Dynamic Scheduler 理论上可能通过更好的 Load Balance 赚回 Queue 成本。表 2 直接验证的是 MoE Token Routing，不能据此把同一收益外推到任意 Ragged Sequence 或通信抖动。

### 同一个 MoE Batch，也不存在一个万能 Kernel

RaMP 在 H200 上研究 8 种 MoE 架构，发现最佳配置不仅取决于 Batch，还取决于每轮实际 Expert Routing Histogram。[3]

论文报告：

-

作者在 OLMoE/Qwen3、四类下游任务、每模型 12 个 Prompt 的样本中，观察到每层只有 **8%–14%** 的 Expert 活跃。

-

超过 **96%** 的观测满足论文定义的 Entropy-normalized Balancedness `β < 0.7`。

-

同样 Batch Size 下，不同路由可产生相差 **1.6–10×** 的 CTA Grid。

-

每个模型有 **134–268** 个有效 Kernel 配置。

-

在论文评测中，Static/Default Dispatch 相对其 134 个候选配置中的最佳值可留下 **10%–70%** 的 Kernel Throughput 未实现。

所以“做一个 Fused MoE MegaKernel”并没有结束优化问题。它只是把问题变成：这一步应该选什么 `bm`、`bn`、Warp 数、Pipeline Depth、Split-K 和 Swizzle？

>

**Fusion 可以消除算子边界，却不能消除 Workload Distribution。**

---

## 五、第三个边界：最弱的 Microkernel 会污染整个 MegaKernel

成熟的多 Kernel 系统可以分别调用 cuBLAS、CUTLASS、FlashAttention、NCCL 和特定架构的最佳实现。Whole-model MegaKernel 如果内置的 GEMV、GEMM 或 Attention 较弱，就可能用“少 Launch”换来“每个核心算子都更慢”。

AutoMegaKernel 提供了一个直接案例。[4]

![AutoMegaKernel 带宽利用率](https://mmbiz.qpic.cn/sz_mmbiz_png/J8mNWJkqUePxOpIHBGiazggn5n9jvV5hQEa4Nu59auMEXlfBsBqZ9SrT1ib3EicZJ6h4GwSbzcb8NSicAH55z2nJ5Ds8tcDPJb6o0xeoZtoDOTU/640?wx_fmt=png&from=appmsg)

*图 4：AutoMegaKernel Table 7。在该实现和测试负载中，A100 只达到实测内存带宽峰值的 12.5%–17.7%，H100 为 4.8%–8.6%。作者指出早期 GEMV 没有随训练级 GPU 的峰值带宽扩展。来源：Jaber & Jaber, CC BY 4.0，页面裁剪。[4]*

论文报告该版本 AutoMegaKernel 在 A100 和 H100 上落后于 CUDA-graphed cuBLAS。Table 7 的 Batch-1、Decode Position 0、Empty KV 测试中，其 H100 路径只达到实测带宽峰值的 **4.8%–8.6%**，A100 为 **12.5%–17.7%**；这些数据不覆盖长上下文 Attention。在 RTX 5090 的另一个 622.9MB 模型、BF16 GEMV Kernel-only 测试中，后续调优实现达到约 63% 实测峰值，同机同权重的 CUDA-graphed cuBLAS BF16 约为 90%。[4]

论文其他优势结果还包含 W8A16 MegaKernel 对 BF16 cuBLAS 的非等精度比较，因此不能把量化收益全部归因于融合。

这不能证明 H100 不适合 MegaKernel，也不能证明所有 Whole-model Fusion 都输给 cuBLAS。它证明的是：

>

**当 Library Kernel 已经接近硬件效率上限时，消除 Launch 无法自动补偿较弱 Microkernel 和细粒度同步。**

论文中的两个消融也值得注意：

-

Software Pipeline Depth 2 相对基线测得 **0.98×**。

-

一个 SM Load Balancer 测得 **0.83×**。

这两项来自未锁频测试，作者都标为 Run-to-run Noise，不能当作确定性回退；它们至少没有给出可确认的增益，也说明更多 Pipeline、更多“智能调度”不能只凭设计推断为加速。

### CUDA Graph 抬高了 MegaKernel 的门槛

NVIDIA 公布的特定 Ampere 直线型 Graph 测试中，CUDA 12.6 的重复 CPU Launch 接近：

Trepeat-launch CPU API ≈ 2.5 μs + 1 ns × Nnode

这是 NVIDIA 在 RTX 3060、CUDA 12.6 上，对已经上传、至少约 10 个 Node 的 Straight-line Kernel-only Graph 测得的重复 Launch CPU API Overhead，不是 Device Runtime 或完整 Time-to-solution；首轮 Launch、多 Root、非 Kernel Node 和 Ampere 以前硬件也不适用。但它说明，现代基线已经不是“一次 Kernel 启动几微秒，再乘几百次”的简单模型。[7]

如果 CUDA Graph 已经压低 Host 提交，而多个专用 Kernel 能分别使用最佳资源配置，MegaKernel 必须证明的就不再是“少 Launch”，而是：

-

真正减少多少 HBM 中间物化？

-

解锁多少跨 Operator Pipeline？

-

是否足以覆盖资源折中和设备端调度？

---

## 六、第四个边界：跨节点时，细粒度通信可能反向串行化

单节点内，MegaKernel 可以用 NVLink 上的 Tile-level Put-with-Signal，把通信和 Expert Compute 细粒度重叠。

但跨节点后，同一个抽象会进入 RDMA Transport、NIC Ordering 和 CPU Proxy。融合不能让这些层消失。

一项针对多节点 MoE MegaKernel 的研究在 Perlmutter A100 集群上发现：[5]

-

在固定每 GPU 工作量、总问题规模随 GPU 数增长的 Weak-scaling 实验中，Communication-bound Qwen3-30B 在 8 节点、32 张 A100 上的延迟最多达到单节点归一化理想值的约 **10×**；这不是固定总 Workload 在 8 节点绝对慢 10×。

-

96 个并发传输、8 节点时，Put-with-Signal 吞吐只有 Put-only 的 **2%**。

-

4KB 消息的 Aggregate Fence Time 从 2 节点的 **0.96ms** 增至 8 节点的 **6.1ms**。

-

小消息中 Fence 可占通信时间的 **98%**。

![多节点 MegaKernel 通信串行化](https://mmbiz.qpic.cn/mmbiz_png/J8mNWJkqUePdJ1vjiaGnEhiaQ9BiabzXmXrgneyiabywWm6bOaG8WnYqqtdP2e3mBCl4vGRlzEpib9iczbdswCzLPJdt44ZTPkDnSeFsS2NPB60Ug/640?wx_fmt=png&from=appmsg)

*图 5：多节点 MegaKernel 中，Put 与 Signal 之间的正确性 Fence 会让 Proxy 等待 NIC Pipeline 排空。Perseus 通过分离 Signal 并把 Ordering 下推 NIC 来减少串行化。来源：Oh & Singh, CC BY 4.0，页面裁剪。[5]*

根因不是“MegaKernel 这个名字”，而是它依赖的细粒度通信语义：每个 Tile 的 Signal 必须在对应 Payload 可见后到达。Proxy-based RDMA 用 Fence 保证顺序，Fence 又会排空 NIC Pipeline，于是本来为了增加重叠的细粒度信号，反而制造了串行化。

论文提出 Perseus 后，在 Proxy-based Transport 上获得最多 **10.3×** 端到端加速，并将示例中的 Fence 数量降低 8×。[5]

这个案例的工程含义是：

>

**Distributed MegaKernel 的性能边界不仅在 GPU Kernel，还在 NIC、Proxy、Ordering Semantic、Fabric 和每个 Expert 可用于隐藏通信的计算量。**

因此以下场景不适合只靠 MegaKernel 解决：

-

EP All-to-All 已经位于关键路径。

-

Rank 间 Token 分布高度不均衡。

-

每个 Expert 的 Compute 太小，无法隐藏网络延迟。

-

Transport 不支持高并发、低成本的 Device-initiated Ordering。

-

多节点 Tail Latency 由最慢目标或 Fence Drain 决定。

---

## 七、第五个边界：一个 Launch Configuration 装不下所有阶段

独立 Kernel 可以分别选择：

-

Threads per Block。

-

Register Cap。

-

Shared-memory Carveout。

-

Tensor Core Tile。

-

Pipeline Depth。

-

Cluster Shape。

-

Warp Specialization。

Whole-model MegaKernel 则容易被同一个 Launch Envelope 约束。

Mirage 的一项 2026 年 9 月开放 Feature Proposal 建议把“整个 Decode 一个 Persistent Kernel”改为“每个 Model Slice 一个 Persistent Kernel”，理由非常明确：一个 Launch 会为其中所有阶段固定 Threads per Block、Register Cap 和 Cluster Shape。截至本文核验时，它是 Runtime V2 的设计提案，不是已发布实现或受控 Benchmark。[16]

他们列出的目标配置甚至互相冲突：

|

Slice

|

Threads

|

Register / Cluster 需求

|

KDA

|

512

|

128-register cap

|

MoE

|

256

|

最高 255 registers

|

Attention

|

3 Warpgroups

|

Cluster Size 2、TMA Multicast、特定 Tensor Core Path

这不是 MegaKernel 的失败，而是设计粒度的收缩：

>

**更大的优化域不等于更好的资源配置。合理边界可能是 Model Slice、Layer 或 Fusion Island，而不是整个 Decode。**

### Register 与 Local-memory Spill

CUDA 官方文档说明，Register、Shared Memory、Thread 和 Barrier 共同限制每个 SM 可驻留的 CTA；无法保留在 Register 的值会 Spill 到 Local Memory，而 Local Memory 物理上位于 Device Memory。[6][8]

公开工程记录给出了具体征兆：

-

Mirage 的 Qwen3 MPK A100 Demo Issue 报告了 **224 Registers/Thread** 和 **2272-byte Stack Frame**；同一份 `ptxas` 输出显示 0-byte Explicit Spill Load/Store，Issue 作者怀疑部分间接索引数组进入 Local Memory，并指出同一 Kernel 中 Worker 与 Scheduler 共享高 Register Allocation。[17]

-

Triton 的一个 Fused Dot Regression 从 **34 Registers** 增至 **255 Registers**，并产生 Spill Load/Store。[18]

-

XLA 一个大型 Complex128 Fusion 在 A100 上因 Unroll 和 Register Allocation Cliff 出现 **1.4–3.6×** 回退。[19]

后两个不是 Whole-model MegaKernel，它们是机制证据：融合扩大代码和 Live Range 后，资源使用可能跨过离散阈值，而不是平滑变差。

### Shared Memory 与 Pipeline Depth

Ada-MK 将 L20 的 128KB Combined L1/Shared Capacity 与论文所引用 Hopper 的约 227KB Shared-memory Capacity 对比，并报告可实现 Pipeline Stage 从理想的 4 降至 2、Duty Cycle 损失超过 30%；实际 Pipeline 还受 Per-block/Per-SM Allocatable Limit 约束。其 K-dimension Split 的核心作用，就是把峰值 Shared Memory 降低 50%。[1]

一个截至核验时尚未得到维护者回复或独立复现的 Mirage Issue Reporter 还报告：在 SM86 上，一个三阶段 Tile 需要 110592 Bytes Shared Memory，而 Worker Budget 只有 98304 Bytes，首轮 Decode 触发非法 Shared-memory Write；降为两阶段后需求变成 73728 Bytes。[20]

这说明“代码可编译”与“性能可移植”是两回事。跨 A100、H100、B200、Ada 或 AMD 时，TMA、Tensor Core ISA、Shared Memory、Cluster、Barrier 和 Warp Size 都可能改变最优边界。

---

## 八、第六个边界：长期驻留会改变公平性与故障半径

Persistent MegaKernel 的优势来自“不要退出”。副作用也来自“不要退出”。

### 1. 多租户与高优先级任务可能排队

NVIDIA 在 Green Context 文档中给出的动机案例是：Kernel A 先启动并占满所有 SM 资源，后到的低延迟 Kernel B 即使位于另一个 Stream，也要等 A 的 Block 释放资源；为 B 保留 SM 可以让它立即开始，但会拉长 A。[11]

所以 Persistent Kernel 并非必然独占 GPU，但系统必须主动选择：

-

少 Launch 一些 CTA，保留 Capacity。

-

使用 MIG、MPS 或 Green Context 分区。

-

加入 Yield Point 或缩短驻留周期。

-

接受单 Workload 峰值下降，换取多租户 SLO。

经典 Persistent-thread 调度研究在 GTX 1080 Ti 的合成 Compute、Memory、Branch 和 Special-function Kernel 共驻留实验中，测到执行时间膨胀上限约为 **1.45–1.8×**；同时，Interleaving 又能提高系统利用率和吞吐。[12] 具体倍率不能直接搬到 H100/B200 LLM Serving，但它证明公平性与吞吐是两项目标，不会自动一致。

### 2. 死锁从 Kernel Boundary 进入设备 Runtime

普通 CUDA Grid 不保证 Block 调度顺序。一个 Block 不能安全等待另一个尚未获得 Residency 的 Block；Whole-grid Barrier 需要 Cooperative Launch 等特殊机制，并受可共同驻留数量约束。[6]

Mirage 曾在 B200 上出现一个典型死锁：[13]

-

GPU 有 148 个 SM。

-

系统启动 128 个 Worker Block 和 80 个 Scheduler Block。

-

如果 80 个 Scheduler 先各占一个 SM，只剩 68 个 Worker 能启动。

-

已启动 Worker 等待分配给未启动 Worker 的 Task，形成永久等待。

这是一个 Placement Race：PR 通过强制 Scheduler 先启动稳定复现，普通运行是否 Hang 取决于放置顺序。修复方式是把 4 个 Scheduler Warp 打包进一个 Block，让 80 个 Scheduler 只占 20 个 SM，为 128 个 Worker 留出 Residency。[13]

这不是“MegaKernel 必然死锁”，而是说明：

>

**Kernel Boundary 原本提供的完成语义被消除后，Progress、Residency、Barrier 和 Queue Correctness 都要由新 Runtime 重新证明。**

### 3. Debugging 和 Profiling 变难

多个逻辑阶段进入一个 Kernel 后：

-

Nsight Timeline 不再自然显示每个 Operator Boundary。

-

Illegal Access 可能在错误发生很久后才暴露。

-

一个 Queue 或 Event 错误可以 Hang 整个 Decode。

-

Per-stage Fallback、灰度和归因更困难。

-

Profiler Replay 可能与状态化 Kernel 的 Host 依赖冲突。[8]

AutoMegaKernel 的约 **1150 行 `schedule/ir.py`** 包含 IR 与 Validator，检查 DAG、Wait 可满足性、Per-SM Queue Order、Happens-before、KV 顺序和输出可达性。在 7160 个真实 Lowering、注入 Mutant 与 Random DAG 的混合样本中，独立 Oracle 标记 6091 个 Unsafe，论文报告对这些样本 Zero False Accept；这不是形式化证明。[4]

这恰恰说明，MegaKernel 不只是性能项目，也是正确性验证项目。

---

## 九、哪些场景不适合 MegaKernel

### 1. 大 GEMM 或 Prefill 已经主导关键路径

如果 Tensor Core 利用率高、Kernel 间 Gap 小，优先优化 GEMM Shape、精度、Layout、Batching 和成熟 Library。最多融合 Epilogue，不要轻易放弃 cuBLAS/CUTLASS 的专用实现。

### 2. CUDA Graph 已经消除了主要 Launch 开销

先用 Graph 建立强基线。若剩余收益只来自少数中间张量，优先做局部 Fusion，不必直接构建 Whole-model Runtime。

### 3. Shape、功能和控制流组合巨大

高风险因素包括：

-

Continuous Batching 与 Ragged Sequence。

-

多 LoRA / Adapter。

-

多种 Sampling Policy。

-

Speculative Decoding 与 Rollback。

-

Structured Output 和 Tool Calling。

-

请求取消、超时与 Early Stop。

-

多种 Quantization 和 KV Layout。

可以支持，不等于低成本支持。通常更适合 Shape Bucket、多个专用 Kernel 和 Graph Replay。

### 4. 多租户、公平性和 P99 优先

如果要支持高优先级请求插队、多模型共卡和严格隔离，应保留较短的可调度边界，或限制 Persistent Grid 的资源份额。

### 5. 通信或跨 Rank Straggler 才是主瓶颈

先解决 Expert Placement、All-to-All、Collective、Topology 和负载均衡。单 Rank Kernel TFLOPS 不会自动减少暴露在关键路径上的网络时间。

### 6. 模型结构仍在快速变化

MegaKernel 会把模型结构、Layout、Scheduler 和硬件紧耦合。算法、量化格式和业务能力仍频繁变化时，维护债务可能大于性能收益。

### 7. 团队不能承担双路径

生产 MegaKernel 应保留 Library/Graph Fallback。这意味着两套执行路径、正确性测试、性能 CI、硬件矩阵和事故处置。如果只能维护一条路径，组合式方案通常风险更低。

---

## 十、副作用清单：被“更少 Kernel”隐藏的账单

|

副作用

|

机制

|

应观测指标

|

Occupancy 下降

|

Register/Shared Memory 增加，Resident CTA 减少

|

Registers/Thread、Theoretical/Achieved Occupancy

|

Spill

|

Live Range 过长，值进入 Local Memory

|

Local Load/Store Bytes、Stack Frame

|

专用化损失

|

多阶段共享 Block Size、Register Cap、Cluster

|

各阶段独立最佳配置与共同配置差距

|

动态调度开销

|

Queue、Atomic、Polling、Fence

|

Scheduler Cycles、Queue Depth、Atomic Stall

|

Load Imbalance

|

Ragged、MoE Skew、Early Stop

|

Per-worker Time CV、Wave Tail、SM Work Distribution

|

I-cache / Code Size

|

分支和 Microkernel Body 膨胀

|

Instruction Fetch Stall、Binary Size

|

公平性下降

|

Persistent CTA 长期占据资源

|

多租户 Slowdown、P99、Jain Fairness

|

移植成本

|

ISA、Shared Memory、Cluster 与 Warp 差异

|

跨代性能、Compile Failure、Variant Count

|

冷启动成本

|

AOT/JIT、Autotune、Binary Cache

|

Compile Time、Warmup、Cache Size

|

故障半径扩大

|

内部 Runtime Hang 影响整步执行

|

Timeout、Hang Hunter、Per-stage Progress Counter

|

数值语义变化

|

Reduction Order、FMA、量化位置改变

|

ULP/Error、Token Agreement、任务质量

还有一个常被忽略的成本：Variant Explosion。

Nvariant ≈ Nshape × Ndtype × Narchitecture × Nfeature

RaMP 的单个模型已经需要 134–268 个有效配置，每个配置约 5 秒编译，并需要 10–24 分钟的模型级 Profiling。[3] 这不是不可接受，但它说明“一个 MegaKernel”在部署系统里往往是一族 Binary、Cost Model 和 Dispatch Policy。

---

## 十一、如何判断它值得做

*图 6：先证明关键路径、Graph 基线和局部 Fusion 仍不够，再讨论 Whole-model MegaKernel。编辑图。*

推荐顺序是：

```
Profile 真实流量
  → 找 E2E 关键路径
  → 建立 CUDA Graph 强基线
  → 做局部 Fusion / IO-aware Algorithm
  → 保留成熟 Library Kernel
  → 只有剩余瓶颈需要跨阶段设备调度时，再做 MegaKernel

```

### 一个最低限度的上线门槛

不要只报 Kernel TFLOPS。至少验证：

1.

目标流量分布加权后的 E2E Latency 或 SLO Goodput 改善。

1.

Prefill、Decode、小中大 Batch 和长尾 Sequence 都有结果。

1.

对比 Eager、CUDA Graph、局部 Fusion 和 MegaKernel 四条路径。

1.

关键生产 Shape 不出现不可接受的回退。

1.

P99、多租户公平性、请求取消和超时仍满足 SLO。

1.

至少两个目标 GPU 代际上收益可解释。

1.

正确性、数值误差和业务功能矩阵通过测试。

1.

有可靠 Fallback，能在新 Shape、新 GPU 或事故时切回。

### Amdahl 上限仍然成立

如果 MegaKernel 能优化的区域只占端到端时间比例 `p`，该区域加速 `s` 倍，则：

SE2E = 1 / ((1 - p) + p / s)

即使该区域无限快：

Smax = 1 / (1 - p)

候选区域只占 20% 时，理论端到端上限也只有 1.25×。如果它原本与通信重叠，真实收益还会更小。

---

## 十二、真正合理的终局：不是一个超级 Kernel，而是一组 Fusion Islands

MegaKernel 最有价值的地方，是迫使 GPU 系统重新审视 Kernel Boundary。

但 Boundary 不只是历史包袱。它还提供：

-

不同阶段独立的资源配置。

-

成熟 Library Kernel 的复用。

-

Runtime 的 Batching 与优先级调度。

-

故障隔离和可回退性。

-

清晰的 Profiling 与测试边界。

-

跨硬件的组合式移植。

所以更现实的架构往往是：

```
CUDA Graph
  ├─ Fusion Island A：Norm + Projection Epilogue
  ├─ Library Kernel B：cuBLAS / CUTLASS GEMM
  ├─ Persistent Island C：动态微任务或 Decode Slice
  ├─ Library Kernel D：FlashAttention
  └─ Communication：NCCL / Device API

```

MPK 的开放 Proposal 建议从 Whole-decode Runtime 转向 Model-slice Persistent Kernel，Ada-MK 把 Prefill 留给 TensorRT-LLM，RaMP 为同一个 MoE Kernel 保留数百个运行时配置，这些路线都指向同一件事：

>

**MegaKernel 的成熟，不是融合边界无限扩大，而是系统学会在正确的位置停止融合。**

最终判断不应从“还能塞进去多少算子”开始，而应从三个可验证的问题开始：

1.

哪一段关键路径时间能被真正消除？

1.

哪些 HBM 字节和同步边界能被真正消除？

1.

为此失去的资源专用化、调度自由和工程可控性值不值得？

只有当这三个问题都能用目标硬件、目标流量和最强现实基线回答时，一个更大的 Kernel 才可能成为一个更快的系统。

>

**MegaKernel 不是万金油。它是一笔交换：用专用化、耦合和运行时责任，换 Launch、物化和同步。真正优秀的系统，不是把所有边界都消灭，而是知道哪些边界昂贵，哪些边界仍然有价值。**

---

## 参考资料

[1] Dong et al. Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference, 2026.

[2] Jin et al. Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel, MLSys 2026.

[3] Sharma and Datta. RaMP: Runtime-Aware Megakernel Polymorphism for Mixture-of-Experts, 2026.

[4] Jaber and Jaber. AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis, 2026.

[5] Oh and Singh. Eliminating Hidden Serialization in Multi-Node Megakernel Communication, 2026.

[6] NVIDIA. CUDA Programming Guide.

[7] NVIDIA. Constant Time Launch for Straight-Line CUDA Graphs and Other Performance Enhancements, 2024.

[8] NVIDIA. Nsight Compute Profiling Guide.

[9] Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, NeurIPS 2022.

[10] Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, ICLR 2024.

[11] NVIDIA. CUDA Programming Guide: Green Contexts.

[12] Zou et al. RTGPU: Real-Time GPU Scheduling of Hard Deadline Parallel Tasks with Fine-Grain Utilization, 2021.

[13] Mirage. Prevent split-launch deadlocks by packing schedulers, 2026.

[14] FusionStitching. Boosting Memory Intensive Computations for Deep Learning Workloads, 2020.

[15] Zhang et al. MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators, SC 2024.

[16] Mirage. Feature Request: Megakernel library, 2026.

[17] Mirage. Non-Optimal Register Utilization, 2025.

[18] Triton. FMAs now using a lot more registers than necessary and often spilling, 2025.

[19] OpenXLA. Loop-fusion unroll factor 4 makes large complex128 fusions 1.4–3.6× slower, 2026.

[20] Mirage. Out-of-bounds shared-memory write in cutlass linear kernel, 2026.
