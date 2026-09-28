# 硬件特性对 Prefill、Decode、KV、MoE 的影响

## 文档契约

- **唯一问题**：如何从硬件资源推导四类关键workload的不同优化优先级。
- **In scope**：phase/resource mapping、bottleneck shifts、parallelism、决策问题。
- **Out of scope**：单项硬件机制（91-94）、vendor最终判断。

| workload | 典型压力 | 首要硬件问题 | 软件决策 |
|---|---|---|---|
| prefill | 大GEMM、attention、activation | matrix吞吐、HBM、SRAM tile | chunk/batch、fusion、PP/CP |
| decode | 小GEMM、反复读weights/KV | memory BW、launch、collective latency | continuous batch、graph/trace、TP度 |
| KV | 容量、分页随机读写、迁移 | DRAM容量/BW、DMA、layout | block/quant/offload/P-D |
| MoE | route skew、expert GEMM、all-to-all | bisection、local memory、small GEMM | EP placement、EPLB、capacity |

prefill/decode同机共享weights但资源曲线不同；统一batch策略会互扰。P/D拆分允许不同SKU/parallelism，却增加KV transfer。是否拆分要用SLO goodput而非单phase峰值。

更大memory可容纳更多KV/weights，不保证decode更快；更高matrix TOPS不保证small-batch利用；高link peak不保证all-to-all tail；大SRAM不保证compiler能安排实际tile。每个硬件特点都需经software mapping转化为收益。

架构选择按模型族分层：dense/GQA/MLA/MoE/hybrid/MM；再按context、batch、quant、SLO。不存在一个“最适合vLLM”的静态排名。

验收用phase-separated microbench定位上界，再用混合open-loop workload验证干扰、队列和goodput。结论必须注明瓶颈证据；若只知道官方规格，保留为假设而非性能结论。

## Bottleneck 转移

量化 weights/KV 可增加 dequant/vector 压力；扩大 batch 提高 matrix 利用却增加排队/KV；增大 TP 降低 compute 但增加 collective；P/D 降低 phase 干扰却引入 KV transfer deadline。

决策要给 sensitivity：batch、context、prefix hit、spec acceptance、expert skew、link contention 改变时瓶颈是否切换。平均 workload 不适用于 tail-heavy 流量。

`INFERENCE`：选型是 workload 分布下的多瓶颈优化；`RECOMMENDATION`：同时报告 phase roofline 与混合 goodput；`UNKNOWN`：vendor 相对位置需统一实测。
