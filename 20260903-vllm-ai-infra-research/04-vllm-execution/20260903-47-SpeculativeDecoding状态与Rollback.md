# Speculative Decoding 状态与 Rollback

## 文档契约

- **唯一问题**：draft tokens 如何被 target 验证，并使 RNG、KV、output 与 counters 只提交接受前缀。
- **In scope**：proposer、verification/rejection sampling、lookahead slots、state commit/rollback、metrics。
- **Out of scope**：普通 sampling（46）、KV基本分配（31-35）、具体 speculative 算法性能论文综述。
- **依赖输入**：24 的 request progress；35 的 lookahead slots；46 的 RNG/sampling。
- **唯一输出/Owner**：draft→verify→accept 的联合 commit/rollback 合同。
- **相邻篇不得重述**：各 KV 文档只消费 finalized/temporary state 分类。
- **证据基线**：vLLM `bb363db9...` 的 `v1/spec_decode/`、GPU `spec_decode/`、scheduler/request fields。

## 1. 事务模型

一次 iteration 中 proposer 产生 `k` draft tokens；target 对它们及必要 bonus position 计算概率；verifier 接受最长合法 prefix并可能采样 correction/bonus token。对外只提交 accepted tokens，未接受 tail 的临时 state 必须丢弃或标为可重新 prefill。

## 2. 多状态守恒

需要同步提交：output token history、target KV、draft KV/hidden state、positions、RNG streams、`num_computed/scheduled/in_flight`、logprobs、stop detection。只裁剪 output 而保留 rejected KV 会在后续 step 污染结果。

lookahead slots 是容量 reservation，不等于 finalized cache。prefix caching 必须排除可重新 prefill 的尾部；preemption 或 async failure 也要以 plan sequence 识别哪些 state 已 commit。

## 3. 算法/硬件组合

draft model、MTP/EAGLE、n-gram 等 proposer 的 weights/state 不同。target backend需支持一次验证多个 positions；静态 graph/AoT profile 需覆盖 `k+1` shapes。若硬件 small-batch launch 慢，额外 draft/verify可能降低吞吐。

## 4. correctness

严格 speculative sampling 应保持 target distribution；实现需证明 acceptance/rejection math、RNG independence 和 correction distribution。贪心模式也需处理 tie/precision 差异。acceptance rate 高不等于无偏。

## 5. 故障矩阵

- zero/all/partial accept；
- EOS/stop 位于 draft 中间；
- preempt/abort between propose and commit；
- target Future failure 与 stale draft output；
- P/D transfer 含 draft groups；
- quantized draft/target 和不同 parallel degree。

## 6. 固定源码对象与原子边界

V1 主路径位于 `vllm/v1/spec_decode/` 与 `vllm/v1/worker/gpu/spec_decode/`，runner 管理 draft token IDs、verify inputs 和 accepted output；scheduler 为 lookahead/额外 tokens 预留资源。不同 proposer（draft model、n-gram、MTP 等）不改变 accepted-length 作为 commit point。

| 状态 | verify 前 | commit 后 |
|---|---|---|
| target KV | 可含候选窗口写入 | 截断/保留到 accepted boundary |
| draft state | 已推进 | 依据下一轮协议同步/重建 |
| RNG | draw 尚未最终归属 | 只按接受语义推进 |
| streamed output | 不可见 | accepted tokens 才可见 |

`SOURCE_IMPLEMENTED`：spec decode modules、draft/accepted data flow；`INFERENCE`：所有状态必须共享 accepted epoch；`RECOMMENDATION`：测试 0/partial/full accept 与 abort 交错；`UNKNOWN`：vendor device sampler 的 rollback fence。

## 7. 评价

报告 acceptance length distribution、target calls saved、draft cost、额外 KV/workspace、ITL p99、质量一致性和 fallback rate。**UNKNOWN**：某插件能跑普通 decode 不推出它支持 spec decode；需要完整状态 rollback 与组合证据。
