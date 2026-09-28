# EngineCore 生命周期与 Step 协议

## 本篇唯一主问题

`EngineCore` 如何初始化 scheduler/executor/KV 资源，并通过 `step()` 在调度决策、异步执行和状态更新之间维持协议？

## In scope

`vllm/v1/engine/core.py` 的初始化阶段、`add_request/abort_requests/step/step_with_batch_queue`、scheduler-executor 接口、输出构造和快照/重置边界。

## Out of scope

不展开 scheduler 内部算法、KV 块分配、executor RPC 实现或 sampler。

**固定源码基线**：vLLM `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` 的 `vllm/v1/engine/core.py::EngineCore` 及其 `step/step_with_batch_queue`。

## EngineCore 的组件图

```text
EngineCore
  ├─ VllmConfig / effective platform config
  ├─ Executor
  │    └─ Worker(s) / ModelRunner(s)
  ├─ KV cache configs and scheduler KV managers
  ├─ Scheduler or platform-selected scheduler class
  ├─ KV/encoder-cache connectors and event publishers
  ├─ batch queue / async execution bookkeeping
  ├─ DP coordinator/wave state where configured
  └─ utility, profile, sleep, reset, shutdown controls
```

EngineCore 是调度与执行的编排 owner，但不执行 model math。Scheduler 拥有请求和逻辑资源决策，Executor/Worker 拥有对物理设备的执行。

## 初始化是一条必须整体成功的链

概念顺序：

1. 读取并保存 vLLM/model/cache/parallel/scheduler 等 config；
2. 创建 executor，初始化 workers 和 distributed environment；
3. 加载 model；
4. 从 worker/backend 获得 KV cache spec 和可用内存/profile 信息；
5. 构造 KV cache configs，可能调整 effective max model length；
6. 在 worker 上分配/初始化物理 KV；
7. 构造 scheduler、connector、event publisher 和异步/batch queue 状态；
8. 完成 warmup/capture/profile 后，通过 `EngineCoreReadyResponse` 向 frontend 回传 effective capacity/config。

该顺序的关键不变式：Scheduler 不得在物理 KV 容量未知时开始接管请求；frontend 不应只依赖 desired `max_model_len/max_num_seqs`，而应使用 ready response 中的 effective 值。

## Add request 的 core 边界

`EngineCoreRequest` 进入 core 后被转换为 core `Request`，添加 block hashes/structured-output/KV connector 相关状态，然后交给 scheduler。请求初始状态通常为 `WAITING`；若 structured output grammar 尚未就绪，则可以先处于等待 grammar 状态。

`abort_immediately` 请求仍被加入 scheduler 后立即 abort，源码注释说明这是为了让 connector-side cleanup 通过标准 `request_finished` hook 执行，例如 D 侧 admission 拒绝时释放 P 侧 prefill blocks。`SOURCE_IMPLEMENTED`

这是“失败请求也可能必须进入状态机才能正确清理”的具体例子。

## 基本 step 协议

```text
Scheduler.schedule()
  -> SchedulerOutput
  -> Executor.execute_model(SchedulerOutput, non_block=...)
  -> ModelRunnerOutput or Future
  -> Scheduler.update_from_output(SchedulerOutput, ModelRunnerOutput)
  -> EngineCoreOutput(s) + SchedulerStats
  -> frontend OutputProcessor
```

### Schedule

Scheduler 根据 waiting/running request、token budget、KV/encoder/connector 状态产生计划。这个输出不是 model tensor，而是从控制面投影到 runner 的 execution descriptor。

### Execute

Executor 将同一 plan 分发给相关 worker/rank。`non_block=True` 可返回 future-like 对象，但其完成究竟表示 worker result、host readback 还是真实 device completion，需由具体 executor/backend 契约确定。

### Update

Scheduler 结合原 `SchedulerOutput` 和 model output，更新 request progress、output token、finish/preempt/KV/connector 状态，产生 frontend 可消费的 `EngineCoreOutput`。

不变式：只有与某个 scheduler plan 对应的 result 才能更新该轮状态；如果允许 run-ahead，必须有 sequence/placeholder/in-flight 计数防止 stale output 改写已重置请求。

## Batch queue 与 run-ahead

`step_with_batch_queue()` 允许在前一个 model execution 尚未完全退休时调度/提交后续 batch，用队列保持 `(Future, SchedulerOutput)` 对应。这可以重叠 host scheduling 和 device work，但会改变状态语义：

- `num_computed_tokens` 可乐观地包含 in-flight token；
- request 保留 placeholder/stale/in-flight/last schedule sequence 等字段；
- KV block free 可能需要延迟到不再有旧 writer；
- preemption/reset/abort 必须处理已提交但尚未应用的输出；
- future 返回顺序必须与 scheduler plan 序列一致，或有显式 reorder 协议。

## `EngineCoreOutput` 是状态 delta

它包含某 request 的 `new_token_ids`、logprobs/pooling output、finish/stop reason、events、KV/EC transfer params、trace headers、prefill/spec metrics、routed experts、NaN count、MM cache miss 和 sampling mask。

这是 delta 而非完整 frontend response：

- detokenization、stop string 处理和 request output formatting 在 frontend；
- `finished` 只由 `finish_reason is not None` 决定；
- stop string 可由 frontend 发现后再 abort core；
- DP/multi-client 外层 `EngineCoreOutputs` 还携带 engine index、scheduler stats、timestamp/wave/utility 信息。

## Pause、sleep、reset 不是同一操作

- pause generation 有 `abort/wait/keep` 模式：分别终止、等待或冻结队列；
- sleep 可释放不同 level 资源，frontend 还会清 MM cache；
- reset prefix cache 可选择是否处理 running requests/connector；
- reset encoder cache 源码文档明确用于权重更新后避免复用 stale vision embeddings；
- shutdown 是进程、executor、device 和 IPC 生命周期的终点。

任何后端替换 Engine/Worker 时，不只要支持 forward，还要定义这些 utility/lifecycle 操作。

## 失败语义

- scheduler 在 execute 前失败：尚未应发生新 device work，但可能已做 reservation；
- worker execute 失败：需区分 request-level 可重试错误和 executor permanent failure；
- future 失败：必须知道已提交 work 是否还会写 buffer/KV；
- output update 失败：device 可已成功，但控制面状态尚未提交；
- frontend output channel 失败：core 可仍在运行，需显式 abort/cleanup。

静态源码证明存在这些边界，不证明所有 backend 已经通过每个 fault injection。

## 架构决策点

1. ready response 中哪些 effective config 必须与 frontend/router 同步？
2. scheduler reservation 何时提交，execute 失败如何 rollback？
3. future 完成的 backend-specific 定义是什么？
4. batch queue 的最大 run-ahead 和状态序列如何防止 ABA/reuse？
5. pause/sleep/reset/weight update 对各类 cache/artifact/in-flight work 有何失效协议？
6. EngineCoreProc/Executor 永久失败是否可与 request-level error 区分？

## 证据结论

EngineCore 组件、step/batch queue、request/output fields 和 lifecycle APIs 为 `SOURCE_IMPLEMENTED`。各 vendor future 的真实 device completion、全故障路径 rollback 和不同 utility 组合的实机安全性为 `UNKNOWN`。

## 本篇输出契约

`24`–`28` 分别展开 request state、budget/admission、preemption/abort、output 和 async completion；它们不重写 step 总体链路。
