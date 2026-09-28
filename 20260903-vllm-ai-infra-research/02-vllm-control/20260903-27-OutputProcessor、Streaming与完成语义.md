# OutputProcessor、Streaming 与完成语义

## 本篇唯一主问题

vLLM frontend 如何将 `EngineCoreOutput` delta 组合成有序的 token/text/logprob/finish 输出，并在 stop、abort、parallel sampling 和 streaming input 下定义完成？

## In scope

`OutputProcessor`、frontend `RequestState`、collector/queue、detokenizer、logprobs、finish reason、external/internal ID mapping、parallel-sampling parent/children、stream interval 和 streaming input continuation。

## Out of scope

不解释 HTTP/SSE socket backpressure，不解释 sampler 数学，不讨论 device completion。

**固定源码基线**：vLLM `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` 的 V1 output processor、frontend request state 与 detokenizer 路径。

## Frontend RequestState 的责任

`OutputProcessor` 在 `request_states` 中按 internal request ID 保存 frontend state，并另有：

- `parent_requests`：parallel sampling parent 到 children 的关系；
- `external_req_ids`：external ID 到多个 internal ID 的映射；
- `LoRARequestStates`：用于请求级 LoRA 统计/生命周期；
- tokenizer/detokenizer/logprob processor；
- async `RequestOutputCollector` 或 offline returned-output 模式；
- stream interval 和 tracing/stats 状态。

它不是 Scheduler `Request` 的共享引用。两份状态通过 ID 和 delta 消息同步，所以 frontend/core 部分失败是显式一致性问题。

## Output 处理顺序

`process_outputs()` 对每个 core output 的概念顺序：

1. 按 request ID 找 frontend state；已 abort 时忽略。
2. 用 core timestamp/events 更新 iteration/request stats。
3. 提取 token、pooling、finish、stop、KV/EC transfer 和其他 metadata。
4. 首次离开 prefill 时记录 cached/cache-creation token stats。
5. generation request 执行 detokenization 和 stop-string 检测。
6. 更新 sample/prompt logprobs 和 sampling mask/routed experts/spec metrics。
7. `make_request_output()` 根据 output kind/stream interval 构造对外 object。
8. async 模式放入 per-request queue，offline 模式放入返回 list。
9. finish 时处理 streaming continuation 或移除 state/ID mapping/parent child。
10. 若 stop string 只在 frontend 被发现，返回 `reqs_to_abort` 请求 core 停止。

`SOURCE_IMPLEMENTED`。

## FinishReason 是 core/frontend 共享语义

`FINISH_REASON_STRINGS` 是 `stop/length/abort/error/repetition`，源码注释指出它们是 external API 的一部分。`FinishReason.ERROR` 被说明为 request-level 可重试内部错误，并转为 HTTP 500。`SOURCE_IMPLEMENTED`

审查需要分开：

- `stop`：EOS/token stop 或 frontend stop string；
- `length`：max tokens/model length/过长 prompt ignored；
- `abort`：显式取消；
- `error`：内部请求错误，并非所有 executor crash 都能安全降级为此；
- `repetition`：重复模式检测。

同一对外 finish string 可对应多个内部 status，所以 metrics 如果只标 finish reason，会丢失内部原因。

## Stop string 使 frontend 拥有一个终止决策

detokenizer 使用新 token IDs 更新 text，并可返回 stop string。此时 frontend 把 finish reason 改为 STOP、stop reason 设为字符串。如果 core 未 finished，后续 abort core。

这一切分导致两个完成点：

1. frontend 决定不再公布新 token；
2. core/device 完成退休并回收资源。

两者之间需 abort/stale-output 协议，不能假定是原子操作。

## Streaming output 与 stream interval

`make_request_output()` 可根据 `RequestOutputKind` 和 `stream_interval` 选择 cumulative/delta/final-only 输出时机。所以：

- EngineCore 每次产生 delta 不保证 frontend 每次都向 client emit；
- stream interval 可改变客户观测 ITL，即使 device decode cadence 不变；
- cumulative 输出重复携带已有 text/token，delta 输出对序列丢失/重复更敏感；
- final-only 不应用 streaming ITL 评价。

SSE/socket 发送是更外层的 backpressure 边界，由 production/80 专篇讨论。

## Parallel sampling

一个外部 request 可对应 parent 与多个 child request。每个 child 拥有 internal ID、sampling params 和 RequestState；parent 负责收集/排序 child outputs。

完成语义必须回答：

- 某 child finished 是否立即可 emit；
- parent 何时 finished；
- abort external/parent ID 时是否所有 children 收敛；
- 某 child 错误是失败全部 request 还是返回部分；
- usage/logprob/metrics 按 parent 还是 child 计算。

vLLM 有 parent/child mapping 和 recursive abort 源码，但业务 API 对部分 child failure 的契约需按 endpoint/test 确认。

## Streaming input / resumable session

`OutputProcessor.add_request()` 如果发现同 internal request ID 已存在，进入 `_update_streaming_request_state()`：

- 新 request 若不 resumable，表示 final marker，可结束已停的 session 或标记最后 input chunk；
- resumable update 包含 prompt/token IDs/arrival time；
- 上一 input 已完成时可立即应用，否则放入 `input_chunk_queue`；
- core finish 时若还有 input chunk，应用下一个；否则进入等待更多 input。

这使 request/session/chunk 成为三层身份。普通 one-shot completion 规则不能直接外推 resumable session。

## 完成的四个层次

| 层 | 完成含义 |
|---|---|
| model step | 本轮 tensor/token 已产生 |
| core request | `EngineCoreOutput.finish_reason != None` 或 core 请求状态 finished |
| frontend request | state 移除，最终 `RequestOutput` 已放入 collector/返回 |
| client-visible | 最终 protocol chunk 已写出/客户收到，取决于定义 |

指标和重试必须选定层次。用 core finish time 当 client E2E latency 会忽略 detokenization、queue 和 network backpressure。

## 失败边界

- core output 找不到 frontend state：忽略，应与 abort metric 区分以便发现 ID drift。
- detokenization/logprob 处理失败：此时 core 可已计算，需 abort/collector error cleanup。
- collector consumer 太慢：如果 queue 无界，可耗尽 host memory；如果有界，需 backpressure/cancel policy。
- final output 入队后 API task 失败：engine/frontend completion 不等于 client receipt。
- streaming input 中间断开：session timeout 和 core waiting resource retention 需有 owner。

## 架构决策点

1. 对外 first/final token 线性化点在 collector 还是 socket write？
2. delta/cumulative/final-only 在重试和丢包下如何保证序列？
3. stop string 的 frontend decision 与 core abort 窗口是否有指标？
4. parallel-sampling child failure 和 usage 如何聚合？
5. resumable session 的 idle timeout、KV retention 和 tenant quota 由谁管理？
6. collector/stream backpressure 是否会反向降低 engine scheduling？

## 证据结论

OutputProcessor state/mapping、处理顺序、stop abort 和 streaming-input queue 为 `SOURCE_IMPLEMENTED`。客户收到的线性化、socket backpressure、部分 child failure 和 session timeout 需 endpoint/deployment/runtime 证据。

## 本篇输出契约

production/80 只需将 frontend collector 映射到 HTTP/SSE；不再重写 core/frontend output state。
