# Ascend Runner、Attention、KV 与 Sampling

## 文档契约

- **唯一问题/Owner**：NPU执行数据面如何实现 vLLM Runner→Attention/KV→Sampling 合同。
- **依赖输入**：31-36 KV；42/44/46 Runner/attention/sampling；102 platform selection。
- **唯一输出**：Ascend-specific data-path map和组合风险。
- **Out of scope**：graph capture（104）、release支持表（105）。
- **源码基线**：vllm-ascend `354635...` 的 worker、attention、sample、distributed/kv_transfer。

## 1. Runner

固定SHA含 `NPUWorker(WorkerBase)`、V1 `NPUModelRunner(GPUModelRunner)` 及v2/310P/xlite variants。这表明它大量复用upstream runner contract，同时替换NPU input batch、block table、device ops、warmup/profile与graph manager。继承不保证零差异：patch目录还修改upstream worker/model behavior。

## 2. Attention family

源码存在通用 `AscendAttentionBackend/Impl`、FA、MLA、SFA、DSA、sparse与context-parallel implementations。selection必须按model attention kind、SoC、dtype/quant、layout、prefill/decode、CP和graph求交；一个“Ascend attention”名称不能概括全部。

KV contract包括paged block table、可能的C8/quant layouts、hybrid/Mamba states、offload与Mooncake/AscendStore等connectors。transfer/offload只能在layout/scales/group mapping和completion一致时使用。

## 3. Sampling/spec

plugin提供Ascend sampler、top-k/top-p、penalties与rejection sampler，并有310P/v2 variants。device sampling可减少D2H/host work，但必须保持46定义的processor顺序、RNG和logprob语义。spec methods存在源码不代表所有engine/model/SKU组合支持；官方文档的feature表还会随release变化。

## 4. Scheduler extensions

源码有recompute、short-request-first、batch-job-aware、DyntraLB等schedulers。它们改变preemption/admission/fairness，不能只作为kernel优化。每个policy要对upstream request/KV counters和abort语义做回归。

## 5. 风险与验证

高风险：NPU/上游InputBatch排列漂移；patch后block table ownership；attention layout选择；async copy/ACLGraph与KV reuse；device RNG；HCCL rank failure。验证从greedy single-rank开始，再prefix/preempt/spec/MM/LoRA、TP/CP/MoE、P/D逐项升级，逐token对reference。

## 6. 证据边界

- `SOURCE_IMPLEMENTED`：上述classes/files在固定SHA存在。
- `INFERENCE`：它们对应公共contracts的映射来自源码结构。
- `UNKNOWN`：无NPU运行，numeric、feature combinations和performance均未验证。

