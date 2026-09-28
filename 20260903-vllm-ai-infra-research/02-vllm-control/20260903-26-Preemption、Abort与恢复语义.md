# Preemption、Abort 与恢复语义

## 本篇唯一主问题

vLLM 如何区分资源不足导致的 preemption、客户端/系统 abort、请求级 error 和恢复，已提交输出与 KV 如何不被错误重放？

## In scope

preemption victim、request status/progress reset、stale/in-flight output、deferred block free、frontend/core abort 传播、streaming/P-D cleanup、恢复不变式。

## Out of scope

不解释 worker/process restart 容灾，不展开 KV allocator，不定义 HTTP disconnect 协议。

**固定源码基线**：vLLM `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` 的 V1 scheduler、request status、EngineCore abort 与 deferred free 路径。

## 四个不可混用的概念

| 概念 | 目标 | 请求以后是否可继续 | 必须处理的状态 |
|---|---|---|---|
| preemption | 临时回收资源 | 是 | logical progress、KV、in-flight output、RNG |
| abort | 永久终止 request/session | 否，除非新 session update | frontend collector、scheduler、KV、runner slot |
| request error | 结束当前 request | 视错误可重试性 | finish reason、connector cleanup、对外错误 |
| worker/executor failure | 故障域可超过 request | 通常不能直接原地继续 | process group、device memory、全部 in-flight work |

## Preemption 的触发和 victim

当 Scheduler 为当前 request 调用 `KVCacheManager.allocate_slots()` 返回 `None`，它可从 running 中选 victim：

- priority policy：选择 priority 更差/到达更晚的 request；
- 其他 policy：从 running 末尾弹出；
- 若 victim 是本轮已计划 request，先撤销 token/block/encoder/spec reservation；
- 然后 `_preempt_request()` 修改状态、记录 event/计数，并释放或重置相关资源。

preemption 是缓解 KV 压力的机制，但会付出 recompute/queue delay，且 async 下不能忽略已提交 output。

## 为什么 progress reset 容易错

一个请求可同时存在：

```text
A. 已向客户端公布的 output token
B. 已被 scheduler 应用但尚未公布的 token
C. 已在 device/worker 计算但尚未被 scheduler update 的 result
D. 已预留 placeholder，但 device result 尚未完成
E. speculative candidate，尚未验证接受
F. 已写入物理 KV 但尚未能被安全重用的状态
```

恢复时必须明确从哪个逻辑 token 重建。把 `num_computed_tokens` 简单置零可能导致已公布 token 重复，或将旧 in-flight result 应用到新 block table。

## vLLM 为 async/preemption 保留的状态

`Request` 包含：

- `num_output_placeholders`：未退休输出保留；
- `num_stale_output_tokens`：preemption 时已在途中、返回后不应改变 reset counters 的 token；
- `drop_stale_output`：same-step preempt + resume/prefix reset 时应丢弃 stale output；
- `num_in_flight_tokens`：async/PP run-ahead 已乐观计入 computed progress 的 token；
- `next_decode_eligible_step`：V2+PP+async 下保持 worker broadcast slot ring cadence；
- `last_sched_seq`：对 deferred block freeing 建立 fence。

这些字段是为了表达 B–F 之间的差异。`SOURCE_IMPLEMENTED`

## Abort 是 frontend 和 core 的两阶段动作

`OutputProcessor.abort_requests()` 接受 external 或 internal request ID：

- external ID 可映射到多个 internal child request；
- parent request 可递归 abort parallel-sampling children；
- frontend `RequestState` 被移除，LoRA state 收到 finished；
- async collector 收到 `FinishReason.ABORT` 的最终 output；
- 返回 internal IDs，由 `AsyncLLM/LLMEngine` 向 `EngineCoreClient.abort_requests` 传播。

这个顺序先结束 frontend view，再请求 core 清理。它保证用户不再等待，但不自动证明 device command 已取消。core/worker 必须单独保证 stale output 不再改变 request state，且 buffer 重用前无旧 writer。

## 已 abort request 的输出

`OutputProcessor.process_outputs()` 如果找不到 `request_states[req_id]`，会忽略已 abort request 后到达的 core output。`SOURCE_IMPLEMENTED`

“忽略”是 frontend 语义，不是物理安全：该 output 可能已经由 device 计算并写过 KV/buffer。Scheduler/runner 仍需完成退休与回收。

## Stop string 发现是一个反向 abort 路径

core 返回 token 后，frontend detokenizer 可检出 stop string，把 finish reason 设为 STOP。如果 core output 自身还未 finished，`OutputProcessor` 将 request ID 放入 `reqs_to_abort`，上层 step 再向 EngineCore abort。

这说明部分终止条件的 authority 在 frontend text 层，core 只在后续收到 abort 后停止。所以可能存在一个小窗口：frontend 已决定 finish，core/device 还有工作在途中。

## P/D 与 connector cleanup

`abort_immediately` 允许一个在 D 节点 admission 被拒绝的 request 先进入 scheduler 再立即 abort，以确保 standard request-finished hook 通知 connector 释放 P 侧状态。

这是一个分布式 transaction cleanup 路径。但它不自动证明 network partition、duplicate abort、producer crash 和 consumer timeout 都能清理；这些问题在 KV transfer 专篇保留。

## 恢复不变式

1. 已公布 token 不重复、不改写。
2. 已接受 spec token 与对应 KV/RNG 一起提交；被拒绝 token 不污染恢复状态。
3. stale output 必须被识别，不可应用到新 request generation/block table。
4. logical block free 与 physical reuse 之间存在 completion fence。
5. frontend abort 、core abort、runner slot free 和 connector cleanup 最终收敛，重复消息不产生二次释放。
6. 恢复的 params、LoRA、MM、grammar、RNG 与原 request generation 一致。

## 新 backend 的必测故障矩阵

| 时机 | 动作 | 验收 |
|---|---|---|
| waiting | abort | 无 KV/runner slot，collector 终止 |
| scheduled before submit | preempt/abort | reservation 完整回滚 |
| device in flight | abort | stale output 不公布，旧 writer 退休后才 reuse |
| output ready before update | preempt/reset | generation/sequence 能阻止错应用 |
| partial streaming output | worker failure/retry | 无重复前缀，或明确不自动重试 |
| P/D transfer | D reject/timeout | P-side KV 最终清理，重复 cleanup 幂等 |
| PP/async run-ahead | preempt | cadence/placeholder/KV fence 一致 |

## 架构决策点

1. backend future/event 在哪个时刻允许 logical block 和 physical buffer reuse？
2. preemption 重建的成本是否纳入 fairness/admission？
3. external/internal/parent/child request ID 的 abort 是否幂等？
4. frontend stop 与 core stop 的竞态是否可观测？
5. worker permanent failure 后是否禁止在已部分 streaming 的 request 上透明重试？
6. connector cleanup 有无 timeout、lease、idempotency key 和 orphan reaper？

## 证据结论

preemption/abort/stale-output 字段和 frontend ignore/propagate 路径为 `SOURCE_IMPLEMENTED`。它们在四类 vendor 真实 device completion、进程丢失和 network partition 下的正确性为 `UNKNOWN`。

## 本篇输出契约

KV 物理 fence 由 `03/32`、P/D transaction 由 `03/36`、future completion 由 `02/28` 展开；它们引用本篇的恢复不变式。
