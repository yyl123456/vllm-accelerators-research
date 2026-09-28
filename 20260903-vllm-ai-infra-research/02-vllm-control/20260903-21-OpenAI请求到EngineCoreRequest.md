# OpenAI 请求到 EngineCoreRequest

## 本篇唯一主问题

一个 OpenAI-compatible 请求经过哪些协议、renderer 和 engine 边界才成为 `EngineCoreRequest`，哪些原始 API 语义在进入 core 前已被消费？

## In scope

API request、serving handler、renderer/chat template、tokenization/多模态处理、`AsyncLLM.generate/add_request`、`InputProcessor`、`EngineCoreRequest` 的边界与字段 provenance。

## Out of scope

不解释 `InputProcessor` 的内部校验算法，不解释 OutputProcessor/SSE，不作全部 OpenAI API 字段清单。

**固定源码基线**：vLLM `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` 的 `vllm/entrypoints/openai/`、`vllm/v1/engine/async_llm.py`、`vllm/v1/engine/input_processor.py`。

## “请求进入系统”有多个时刻

| 时刻 | 已发生的事 | 还未发生的事 |
|---|---|---|
| HTTP 开始 | server 开始接收 | body/鉴权/校验未必完成 |
| protocol parsed | request schema 合法 | template/tokenization/model limit 未必完成 |
| renderer output | messages/media 转为 model input | core request 未必建立 |
| `InputProcessor.process_inputs` return | 核心字段已正规化 | scheduler 还未 admission |
| `engine_core.add_request` | 请求已穿过 core client | 未必被当前 iteration schedule |
| `QUEUED/SCHEDULED` event | core 已记录对应状态 | device work 未必完成 |

因此 TTFT 和 admission 计时必须指定起点。vLLM 源码中 `arrival_time` 可来自 prompt/explicit value/当前 wall clock，core event 则使用 monotonic clock 且注释禁止跨进程直接比较。`SOURCE_IMPLEMENTED`

## 输入链路

```text
HTTP OpenAI-compatible schema
  -> serving handler validates endpoint-level options
  -> Renderer parses completion/chat content and template
  -> tokenizer / multimodal processor produces EngineInput
  -> AsyncLLM.add_request()/generate()
  -> InputProcessor.process_inputs()
  -> EngineCoreRequest
  -> InputProcessor.assign_request_id()
  -> OutputProcessor registers frontend RequestState/collector
  -> EngineCoreClient.add_request_async()
```

具体 endpoint 有 completions、chat、embeddings/pooling、transcription 等不同 handler；这里描述共同 engine 边界，不声称所有 endpoint 具有相同字段。

## API 字段不会原样进入 core

### 在 protocol/renderer 层消费

- messages、roles、content parts 和 chat template；
- tool/reasoning 呈现的部分协议细节；
- raw text 到 token IDs/prompt embeds；
- media URL/bytes 的 fetch/decode/processor 结果；
- endpoint-specific response formatting 和 HTTP error representation。

### 以正规化形式进入 core

`EngineCoreRequest` 包含：

- internal `request_id` 和 `external_req_id`；
- `prompt_token_ids`、`prompt_embeds`、`prompt_is_token_ids`；
- `mm_features`；
- 二选一的 `sampling_params`/`pooling_params`；
- `arrival_time`、`priority`、`trace_headers`；
- `lora_request`、`cache_salt`；
- DP `data_parallel_rank/client_index/current_wave`；
- resumable/session/reasoning 状态；
- 为 P/D rejection cleanup 使用的 `abort_immediately`。

它不保留 HTTP connection/socket 对象，因此客户端断连需要 frontend 通过单独 abort 控制消息传播。

## Request ID 是一个显式正确性边界

`InputProcessor.assign_request_id()` 把用户提供的 ID 复制到 `external_req_id`，默认向 internal `request_id` 追加 8 个随机字符。源码对禁用随机化发出警告：重复外部 ID 可导致失败或隐蔽正确性问题。`SOURCE_IMPLEMENTED`

这一设计分开：

- 对外返回/用户 abort 所使用的外部身份；
- core/scheduler/runner 必须唯一的内部身份；
- parallel sampling 还可从 parent request 产生 child ID。

架构审查不应把 HTTP `id`、external request ID、internal ID、client index、batch slot 和 trace ID 合并为一个字段。

## SamplingParams 是已加工的语义对象

`InputProcessor` 会 clone params，根据 max model length 补齐 `max_tokens`，合并 generation config/tokenizer EOS，并可对 trace replay 重写 stop/EOS/min/max 语义。

因此 core 看到的 `SamplingParams` 不是 API JSON 的原样镜像。调试“用户传了什么”和“runner 实际使用什么”时，需要分别记录 desired 和 effective params，同时避免记录敏感 prompt/token。

## Multimodal 使边界更复杂

media 可在 frontend 变成 `MultiModalFeatureSpec`，processor cache 可使用 hash/address 代替重复数据。`EngineCoreOutput.mm_cache_miss_hashes` 还支持当 P0/P1 缓存漂移时由 frontend 丢弃 sender cache 并重发数据。`SOURCE_IMPLEMENTED`

这意味着“请求已经进入 core”不保证所有大型输入都随消息携带；它可以依赖跨进程 cache 协议。缓存 miss 重试的 idempotency、deadline 和计费边界必须单独验证。

## 什么时候算“accepted”

vLLM 的多阶段处理提供了多个候选点，但源码不会为用户的业务 SLO 自动选择定义。`RECOMMENDATION`：

- protocol validation 成功只能叫“输入合法”；
- `EngineCoreRequest` 建立只能叫“前端正规化完成”；
- 需要对外承诺处理时，accepted 应与唯一有界 queue/admission owner 接管对齐；
- 如果 engine 还可因 hard capacity/profile/feature 拒绝，gateway 不应在没有保留或 rollback 的情况下宣称已接受。

## 失败边界

- renderer/tokenizer 失败：尚无 core request，不应泄漏 core/KV 资源。
- params/platform validation 失败：应是可解释的请求错误，不是 worker 失效。
- frontend `RequestState` 已建立但 core add 失败：collector 必须完成/清理，否则 stream 永久等待。
- core 已 add 但 frontend task 取消：需要 abort 消息，不能只删 collector。
- DP 路由到旧 wave：`start_wave` 机制表明存在竞态协调，需按对应 DP 测试验证。

## 架构决策点

1. SLO 起点是 HTTP 边界还是 `arrival_time`，两者差额是否可观测？
2. request identity 在 gateway/frontend/core/child sampling 之间如何映射？
3. desired 与 effective sampling params 是否均可审计？
4. multimodal cache 重发是否有 retry budget 和 deadline？
5. 哪个点开始对外承诺容量，失败时谁撤销？
6. frontend add/core add 部分成功的清理是否有 fault test？

## 证据结论

`EngineCoreRequest` 字段、ID 随机化、params 正规化、MM cache miss 重发为 `SOURCE_IMPLEMENTED`。HTTP 层端到端 accepted/TTFT 线性化是部署/API 契约，不能由 core struct 自动得出。

## 本篇输出契约

`22` 从 `InputProcessor` 内部继续；`23`–`28` 可假定已有正规化 `EngineCoreRequest`，不再重写 API/renderer 边界。
