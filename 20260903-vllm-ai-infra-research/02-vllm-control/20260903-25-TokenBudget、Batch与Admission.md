# Token Budget、Batch 与 Admission

## 本篇唯一主问题

vLLM V1 Scheduler 用哪些 token、sequence、encoder、LoRA 和 KV 限制建立一个 `SchedulerOutput`，这些限制中哪些是安全上限，哪些是调度策略？

## In scope

running-first/waiting scheduling、`max_num_scheduled_tokens`、`max_num_batched_tokens`、`max_num_seqs`、encoder budget、KV allocation/preemption interaction、LoRA cardinality、chunk/alignment guards 和 `SchedulerOutput` 构造。

## Out of scope

不设计外部 admission/fairness policy，不解释 KV allocator 内部，不讨论 device kernel batch layout。

**固定源码基线**：vLLM `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` 的 `vllm/v1/core/sched/scheduler.py`、`output.py::SchedulerOutput` 与 scheduler config。

## Scheduler 不显式分“prefill phase”与“decode phase”

`schedule()` 的源码注释明确：它用 `num_computed_tokens` 追赶 `num_tokens_with_spec + output placeholders`，这一表达统一覆盖 chunked prefill、prefix caching、speculative decoding 等。`SOURCE_IMPLEMENTED`

这是控制面的统一语义，不意味每个 runner/device program 都能在同一 batch 混合 prefill/decode。Tenstorrent plugin 就使用特化 scheduler 禁止一个 TT step 混合两者。

## 两个 token 预算

### `token_budget = max_num_scheduled_tokens`

限制本 iteration 向 request progress 分配的 scheduled tokens。它包括 prompt/decode/spec-related 工作的统一计数，但不是真实 device-time 或 FLOP/byte 成本。

### `input_budget = max_num_batched_tokens`

对输入 batch token 和 speculative draft slot 组合施加限制。每请求安排时会减去 `num_new_tokens + draft_slots`。

两个预算的区分表明：scheduler 已需要分开 logical progress 与 input tensor/slot 形状约束。后端如果还有 profile/bucket/padding/program 限制，不能假设 token budget 自然表达它们。

## 单个 running request 的可调度 token 推导

概念化顺序：

1. 计算 request 待追赶 progress：`tokens_with_spec + placeholders - computed`；
2. 应用 long-prefill threshold；
3. 取 token/input budget 最小值；
4. 限制不超过 `max_model_len - computed - sampled_per_step`；
5. 应用 Mamba/hybrid block alignment；
6. 安排 encoder inputs 并减少 encoder compute/cache budget；
7. 为 multi-module MTP 预留 prefill lookahead；
8. 向 KV manager 请求新 slots；
9. 若无 slots，选 preemption victim 并回滚本轮已预留 budget；
10. 成功后记录 blocks/tokens/spec/encoder 状态。

这条链证明 token budget、KV capacity、encoder budget 和 model-specific alignment 是串联的可行性检查，不是一个单维 batch-size knob。

## Running-first 与 waiting admission

Scheduler 先尝试为 `running` 请求分配 progress，然后在本轮没有 preemption 且未 pause 时处理 waiting/skipped requests。该策略倾向保持已运行请求的 decode cadence，但不是绝对：

- long prefill 可被 threshold/chunking 分割；
- DP prefill balancing 可在非 cadence step 延后 prefill；
- 某 running request 受 encoder/alignment 阻塞时可跳过；
- KV 不足时可 preempt running victim；
- priority policy 会改变 waiting 顺序和 victim 选择；
- structured grammar/remote KV/streaming 等状态未就绪的 request 不能直接 admission。

## `max_num_seqs` 不等于实际并发用户数

`max_num_seqs` 限制 scheduler/model runner 可同时管理的 sequence slots。实际可运行请求数还受：

- KV blocks/context lengths；
- 每 request parallel samples/beam-like children；
- multimodal encoder cache；
- active LoRA 数；
- speculative draft slots；
- backend static batch/profile/trace 形状；
- PP/async 的 in-flight slots；
- workspace 和 communication buffers。

所以它是一个结构性上限，不是容量保证。

## Encoder budget

multimodal/encoder-decoder request 可以需要独立 encoder compute budget 和 encoder cache。Scheduler 的 `_try_schedule_encoder_inputs()` 可以减少本轮 decoder token work、安排外部 cache load，或因容量不足返回 0 token。

因此 `max_num_batched_tokens` 不能单独解释 MM workload；一个 image embed 数可与文本 token 账本不同。

## LoRA cardinality

Scheduler 从本轮 scheduled running requests 收集正 `lora_int_id`，并断言不超过 `lora_config.max_loras`。对 waiting request 的接管需要考虑已活跃 LoRA 集合，以避免 runner 同时加载超出能力。

这是对 batch feature cardinality 的 hard guard，但不定义 tenant fairness 或 adapter cache locality。

## Preemption 回滚当前轮预算

当 `allocate_slots()` 失败时，Scheduler 可以选 victim。若 victim 已在当前轮被加入 `scheduled_running_reqs`，必须：

- 从 scheduled list 移除；
- 把它的 token/input budget 加回；
- 移除它的 new blocks/spec tokens；
- 若安排了 encoder input，恢复 encoder budget；
- 然后执行 request-level preemption。

这是“schedule 计划构造本身也需要 transaction rollback”的源码证据。它不等于 device 已提交后的 rollback，后者由 async/preemption 文档处理。

## Hard safety bound 与 policy knob

| 限制 | 主要性质 | 为什么不是单一分类 |
|---|---|---|
| max model length | shape/position/KV 安全上限 | 可由 effective KV fitting 修正 |
| max scheduled/batched tokens | batch/input 上限 + perf policy | 大小同时影响延迟和效率 |
| max sequences | runner/metadata capacity | 不保证 KV 可容纳 |
| KV allocation | 硬容量 | 失败可触发 policy preemption |
| long-prefill threshold | latency/fairness policy | 又受 model alignment 约束 |
| encoder budget/cache | 硬资源 + perf policy | external load/cache hit 改变实际工作 |
| max LoRAs | runner/model adapter capacity | locality/tenant policy 不在其中 |

## 为什么这不是全局 Admission Controller

Scheduler 可以接管 waiting request 并因本地资源延后它，但它不自动拥有：

- 跨 replica/backend 的 model/feature routing；
- tenant quota 与计费；
- 外部 gateway queue 和 request retry budget；
- client-observed deadline/goodput prediction；
- fleet capacity、autoscaling 和 rolling-upgrade headroom；
- vendor profile/firmware 的全局健康和 artifact inventory。

因此它是 engine-local iteration scheduler 与资源 admission 的组合，不应被文档扩大成完整平台 admission plane。

## 架构决策点

1. backend 的实际工作成本能否用 scheduled token 近似？
2. profile/bucket/padding/trace 限制在 platform validation、scheduler 还是 runner 处理？
3. hard capacity 失败时是等待、preempt、reject 还是 fallback？
4. `max_num_seqs` 与 KV/encoder/LoRA/in-flight capacity 如何联合配置？
5. priority/FCFS 与 long-prefill/encoder skip 导致的非严格顺序是否符合 SLO？
6. SchedulerOutput 是否暴露了后端需要的全部 shape/cost metadata？

## 证据结论

本篇预算、顺序、分配和 rollback 路径为 `SOURCE_IMPLEMENTED`。它们在四类 vendor 上的成本精度、性能最优性与全局 SLO 公平性为 `UNKNOWN`。

## 本篇输出契约

`26`–`28` 使用这里已构造的 schedule plan 和 reservation 概念；KV 物理分配由 `03`组文档展开。
