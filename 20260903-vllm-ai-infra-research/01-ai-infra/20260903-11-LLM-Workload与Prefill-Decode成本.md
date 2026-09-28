# LLM Workload 与 Prefill/Decode 成本

## 本篇唯一主问题

为什么在线自回归 LLM 的 prefill 和 decode 是两类不同工作负载，它们如何决定 batching、KV、编译和硬件需求？

## In scope

decoder-only Transformer 的计算/内存访问形状，prefill/decode 的工作量、batching 价值、长度分布、MoE/多模态/speculative decoding 对 workload 的改变。

## Out of scope

不定义 SLO，不设计 admission policy，不将理论工作量换算为任何硬件的 tokens/s。

## 基本变量

对单个请求，定义：

- `L_p`：prompt token 数；
- `L_o`：实际生成 token 数；
- `L = L_p + t`：decode 第 `t` 步时的已有 context；
- `B`：某个 iteration 的活动 sequence 数；
- `H`：hidden size，`N_l`：layer 数，`N_h/N_kv`：query/KV head 数；
- `b_w`、`b_kv`：weight 和 KV element 的 byte 数；
- `E`：MoE expert 数，`k`：每 token 选中 expert 数。

生产 workload 必须使用 `(L_p, L_o)` 的联合分布，而不是两个独立均值。长 prompt 可能与短 output 相关，chat 应用的长 context 又可能与更长回答相关；打乱这种关系会改变峰值 KV 和排队行为。

## Prefill 的计算形状

prefill 一次处理多个 prompt token。在简化的 dense Transformer 中：

- 线性层的主要工作可形成大的 matrix multiplication，通常更容易提高计算单元利用率；
- causal attention 的 score/value 工作随 context 长度增长，原始表达有近似二次序列工作量；
- FlashAttention 类算法通过 tiling/recomputation 避免物化完整 attention matrix，但不把所有计算变成线性；
- 每层产生 prompt token 的 K/V，必须写入可被后续 decode 寻址的布局；
- 长 prompt 的单次运行可以阻塞已在 decode 的请求，导致 inter-token latency 尖峰。

因此，prefill 常被粗略称为 compute-heavy，但这不是恒真命题。当 context 足够长、KV 写入/读取、互联或内存容量成为主导时，prefill 也可以受内存和通信限制。

## Decode 的计算形状

标准自回归 decode 每次为每个活动 sequence 生成一个新 token：

- 权重在每次 iteration 都要被消费，单请求时 matrix-vector 形状通常难以充分复用权重读取；
- 将多个 sequence 批处理可把形状推向 matrix-matrix，摊薄权重和 launch overhead；
- attention 要读取每个 sequence 已存在的 KV，读取量随 context 长度增长；
- batch 内 sequence 的 context 长度、block table 和采样参数不同，会引入 ragged metadata 和分支；
- 每次只产生少量输出，host scheduling、input packing、launch、D2H 和 sampling 更容易变成显著占比。

因此 decode 经常是 memory/launch-limited，但 MoE、大 batch、超长 context、低比特量化或特定互联会改变瓶颈。

## 为什么 iteration-level scheduling 重要

Orca 把生成模型的多 iteration 特性作为核心，提出以 iteration 而非整个 request 为调度粒度。[原始 USENIX 论文页](https://www.usenix.org/conference/osdi22/presentation/yu)

这一改变允许新请求在旧请求尚未生成完时加入 batch，但也引入新的不变式：

- 请求的 model/KV/RNG/output 状态必须跨 iteration 保持；
- batch slot 可以变化，不能被当作永久 request identity；
- 每次调度只承诺部分 token progress，不承诺整个 request 完成；
- 长 prefill 和连续 decode 的竞争变成 scheduler 策略问题。

## Chunked prefill 改变了什么

Sarathi-Serve 将长 prefill 分块，用更均匀的 iteration 减少 prefill 对 decode 的停顿，其原始论文将目标表述为吞吐-延迟权衡。[论文](https://arxiv.org/abs/2403.02310)

chunking 不只是把 `L_p` 除以块长：

- 每块之间要保留 prompt progress 和已写 KV；
- 中间 chunk 不应向用户公布生成 token，也不应错误推进 RNG；
- 块大小同时影响计算效率、decode 被阻塞时间、KV 分配和 graph shape；
- 静态图或 model-specific program 可能无法从任意中间状态恢复 prefill；
- pipeline parallelism 中的 chunk 形状影响 stage balance 和 bubble。

所以“开启 chunked prefill”是 scheduler、runner、model implementation、KV 和 compiler 的组合 claim。

## P/D 解耦的 workload 前提

DistServe 将 prefill 和 decode 放到不同资源上，目的是解除两阶段干扰，并分别优化 TTFT 与 TPOT；该论文也强调 placement 必须考虑 KV 传输带宽。[论文](https://arxiv.org/abs/2401.09670)

解耦有效的必要条件不只是有两组设备：

- prefill 和 decode 的资源比例能匹配 workload；
- KV 可以带着 model/layout/dtype/rank metadata 安全传输；
- transfer latency 和带宽不抵消干扰隔离的收益；
- producer/consumer 对 cancellation、retry、timeout 和 backpressure 有一致协议；
- 两类 replica 能独立 autoscale，又不在短时间尺度产生负载震荡。

## 特殊 workload 对两阶段模型的修正

### Speculative decoding

draft 一次提出多个 token，target 验证并接受其中一部分。每次 target iteration 的 query token 数大于 1，但只有 accepted token 能提交到输出和 KV 语义。成本取决于 acceptance length、draft overhead 和 verification 形状，不能用标准 one-token decode 简化。

### MoE

MoE 的激活参数少于总参数，但 token routing 会产生 expert 负载不均和 all-to-all。小 decode batch 更容易产生瘦形 expert GEMM；大 batch 可提高局部效率，同时增大通信和尾部 expert 压力。

### Multimodal

vision/audio encoder 工作不能简单归为 decoder prefill。它可以有独立设备、cache、batching 和执行时间；encoder output 又会改变 decoder input length 或 cross-attention 状态。

### Pooling/embedding

这类请求通常没有自回归 decode，如果与 generation 请求混合计算“平均 tokens/s”，指标的语义会发生变化。

## 一个可用的 workload 描述

最少应包含：

```text
request type distribution
arrival process and burstiness
joint prompt/output length distribution
model + tokenizer + dtype/quantization
sampling/logprob/grammar/speculative settings
multimodal sizes and cacheability
LoRA/tenant cardinality and locality
deadline/SLO class
concurrency and cancellation rate
warm/cold cache and session reuse
```

只给出“模型名 + 平均输入/输出 + 并发数”可以做微基准，但不足以复现生产排队、cache locality 和 tail latency。

## 证据边界

- Orca、Sarathi-Serve、DistServe 的机制和论文实验是其各自版本/硬件/workload 下的原始证据。
- 本篇对当代 vLLM 的影响是 workload 分析框架；具体是否实现由后续源码章证明。
- 任何论文报告的倍数不用于本研究的跨硬件性能排名。

## 架构决策点

1. workload 是否保留 `(L_p, L_o)` 联合分布与 burstiness？
2. prefill/decode 是在同一 device step 混合、分时，还是分离部署？
3. chunk size 由延迟、计算效率、graph shape 还是 KV 约束决定？
4. 特殊 workload 是否使一 token/步的成本模型失效？
5. 资源采购是在优化 prefill 还是 decode，它与业务 SLO 一致吗？

## 本篇输出契约

后续 SLO、admission、瓶颈和 vendor 文档可使用本篇的 workload 变量与 prefill/decode 分类，但不再重复其机制。
