# Sampling、LogitsProcessor 与 RNG

## 文档契约

- **唯一问题**：hidden states 如何在一组确定顺序的变换后产生可复现 token、logprob 与 finish evidence。
- **In scope**：logits projection、processors、penalties、top-k/p/min-p、temperature、RNG、logprobs。
- **Out of scope**：structured grammar 组合（49）、spec verification（47）、HTTP output（80）。
- **依赖输入**：42 的 sample positions；43 的 logits contract；61 的 TP shard。
- **唯一输出/Owner**：普通 sampling 的 processor 顺序、RNG ownership、token/logprob commit。
- **相邻篇不得重述**：49A 只拥有 grammar mask；47 只拥有 speculative verification。
- **证据基线**：vLLM `bb363db9...` 的 `v1/worker/gpu/sample/`、`v1/sample/` 与 sampling config。

## 1. sampling pipeline

典型顺序为取得需要采样位置的 hidden states、LM head/logits、模型或用户 logits processors、bias/bad words/penalties、temperature 与截断策略、随机/贪心选择、logprob 提取。顺序是 API 语义；交换 temperature 与 top-p 或 penalty 会改变分布。

## 2. batch RNG ownership

RNG 必须按 request/sequence 拥有，不能依赖动态 batch row 或全局 launch 顺序，否则其他请求加入/完成会改变结果。preemption、async run-ahead、spec rejection 后应定义 RNG 消耗是否回滚。仅设置全局 seed 不能保证跨 backend bitwise reproducibility。

## 3. distributed sampling

若 logits shard 在 TP ranks，系统需选择 gather 后采样、distributed top-k，或 driver sampling 后 broadcast token。各方案决定通信、数值归约顺序与 RNG owner。所有 ranks 必须在下一 step 使用同一 accepted token。

## 4. logprob 契约

要区分 raw model logprob、processor 后 distribution、top-k returned logprobs、prompt logprobs 与 sampled-token logprob。文本 stop 截断后 token/logprob 对齐仍需保持。浮点归约差异可能改变近 tie token，质量验证应同时有 deterministic greedy 与统计 sampling tests。

## 5. 固定源码与 commit 顺序

`vllm/v1/sample/sampler.py::Sampler` 与 `metadata.py::SamplingMetadata` 持有主采样路径；runner 构造 per-row temperature/top-k/top-p、seed/generator、logprob 与 prompt/output token metadata。structured/spec 分别由 49A/47 持有状态，本篇只定义最终 token/RNG commit。

```text
logits -> processors/masks -> distribution transform
 -> RNG draw/top-k/top-p -> sampled token + logprob
 -> accepted/validated -> advance request token and RNG epoch
```

device sampling 或 distributed vocab sampling 必须保持同一顺序；fallback 若改变 processor coverage、reduction order 或 RNG advance 不能静默。`SOURCE_IMPLEMENTED`：Sampler/SamplingMetadata；`INFERENCE`：RNG advance 应与 token commit 同事务；`RECOMMENDATION`：记录 seed、draw epoch、sampling path；`UNKNOWN`：跨 backend bitwise/distribution equivalence。

## 6. 验收

固定 request seed 在 batch churn/preemption 下复现；processors 顺序 golden；NaN/Inf/all-masked logits fail closed；TP 多 rank 一致；greedy 跨 backend tolerance；随机 sampling 做分布检验。吞吐优化不得跳过启用的 processor 或少算 logprobs。

**架构结论**：sampling 是用户可观察语义边界，不是 attention 后的小尾巴。插件若交给私有 runtime 采样，必须声明 processor、RNG 与 logprob 的兼容子集。
