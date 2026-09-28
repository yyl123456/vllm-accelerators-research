# Async Scheduling、Future 与 Completion

## 本篇唯一主问题

vLLM 中 `non_block=True`、batch queue、output placeholder 和 runtime event 分别表示哪一级异步，什么证据才能宣称物理 device work 已完成？

## In scope

executor Future、host RPC completion、batch run-ahead、request placeholder/in-flight counters、result retirement、buffer/KV reuse fence 和 backend completion contract。

## Out of scope

不解释 Python asyncio 入门，不展开 TT 具体 async decode，不解释 CUDA/ACL/TT/TPU event API 细节。

**固定源码基线**：vLLM `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` 的 EngineCore batch queue、Executor Future 与 scheduler update 路径。

## “异步”至少有六个边界

```text
1. frontend coroutine does not block
2. EngineCoreClient message is queued
3. Executor RPC returns a Future
4. Worker accepted/submitted device commands
5. Device commands completed and writes are visible
6. D2H/readback/sampling/finalization completed and result is consumable
```

一个 Future 可以在 3、4、5 或 6 中任一点完成，取决于 concrete executor/worker/runtime。类型注解 `Future[ModelRunnerOutput]` 只证明 host 程序将在未来获得一个结果，不定义 device visibility。

## Executor 层契约

`Executor.execute_model(scheduler_output, non_block=False)` 通过 `collective_rpc("execute_model")` 从第一个 worker result 取 model-runner output；`non_block=True` 返回 Future。`sample_tokens()` 也有相同的 sync/async overload。`SOURCE_IMPLEMENTED`

但 abstract interface 没有说明：

- Future 是所有 rank 返回后完成，还是 driver rank 返回；
- worker `execute_model` return 前是否 synchronize device stream；
- output tensor 是 host-visible、device tensor 或 deferred wrapper；
- collective 错误是在 Future 上抛出还是异步浮现；
- cancellation Future 是否尝试取消 device work。

这些必须由 concrete executor/backend 文档和测试补充。

## Batch queue 为什么需要 placeholder

若 EngineCore 在 step `n` 的 result 未应用前就为 step `n+1` 调度同一 request，它必须乐观假定前一步会产生若干 output token。否则新 step 会重复计算同一位置。

placeholder 表示这个未来 progress reservation：

- scheduler 在构建下一步时把它纳入待追赶 token 计算；
- result 退休时用真实 accepted token 替换/消费保留；
- spec decoding 可一次产生不定长 accepted output，所以不能只用布尔值；
- preemption/abort/reset 可让返回 result 变 stale，需决定交付、丢弃或只退休 counter。

## Happens-before 链

安全的最小关系是：

```text
schedule plan S_n created
  happens-before device work W_n consumes its block table/input slots
W_n writes KV/output
  happens-before result R_n is declared consumable
R_n validated against request generation/sequence
  happens-before Scheduler applies progress/output delta
all writers using old block/slot retired
  happens-before block/slot reuse by another request
```

在异步 pipeline 中，`S_(n+1)` 可以早于 `R_n` 退休，但不得违反消费和 reuse 边。

## 为什么 host result 不总等于 device completion

常见 runtime 是 asynchronous command queue：host 发起 kernel/DMA 后立即返回。只有在以下任一条件成立时，host 才可以将数据视为可消费：

- runtime API 本身是 synchronous；
- 记录并等待了 stream/event/fence；
- D2H operation 具有隐式 synchronization 并已完成；
- tensor/runtime 定义了在访问时自动等待；
- backend deferred wrapper 在 `get_output()/ensure_finalized()` 中完成等待和转换。

这些行为可在源码中寻找，但某些 device fault 只会在同步点浮现，仍需 runtime fault test。

## Completion 应是分级的

| 级别 | 可安全做什么 | 不一定可做什么 |
|---|---|---|
| submitted | 记录 command/future | 读 output、reuse input/KV |
| device executed | 依 runtime 语义使用 device result | host 读取、采样已完成 |
| writes visible | 后续 stream/rank 在有序依赖下使用 | frontend 已可见 |
| host consumable | 读 token/logits/metadata | scheduler 已提交 |
| control committed | 更新 request/KV/RNG/output state | client 已收到 |
| client visible | 计入 API 线性化/SLO | device resource 必已释放 |

## Abort/preemption 下的 completion

取消 Future 不保证已提交 device work 停止。安全策略通常是：

1. 阻止新 work 使用旧 request generation；
2. 将在途 result 标 stale/drop；
3. 保留其写入的 physical resources；
4. 在正确 event/future/fence 退休后再 free/reuse；
5. 幂等消费 placeholder/counters；
6. 只对尚未公布的状态做 rollback/recompute。

如果 backend 不能取消 device command，这个模式仍可以正确，但需要足够的资源 quarantine 和 completion signal。

## 异步与性能的真实收益

异步只在两段可重叠工作且关键路径变短时改善性能。例如：

- host schedule/pack 与前一个 device step；
- D2H readback 与下一个 decode submit；
- PP stage/microbatch；
- communication 与局部 compute。

若后续每次都立即 wait，只是 API 形式为 Future 而没有 overlap。若 run-ahead 增加了 stale work、内存和 tail latency，throughput 收益还需用真实 trace 验证。

## Backend completion contract 清单

每个 backend 必须回答：

1. `execute_model` return/Future completion 表示哪一级？
2. 输出 object 包含 host tensor、device tensor 还是 deferred handle？
3. KV write 和 output readback 是同一 stream/queue 吗？
4. 跨 rank collective 完成如何聚合？
5. 哪个 fence 允许 block/input slot/trace buffer reuse？
6. error 在 submit、event wait、readback 还是下次 sync 浮现？
7. abort/shutdown 是 drain、cancel 还是终止进程？
8. completion callback 是否可重入/只执行一次？

## 证据结论

abstract executor Future、batch queue 和 request async fields 为 `SOURCE_IMPLEMENTED`。“Future 代表真实 device completion”不是从 abstract interface 可得的通用结论；每个 backend 需单独证明，无实机 fault/visibility 测试时保留 `UNKNOWN`。

## 本篇输出契约

KV/physical reuse、executor 和 vendor async 文档使用本篇的六级 completion 语言，只写本层映射，不重写异步定义。
