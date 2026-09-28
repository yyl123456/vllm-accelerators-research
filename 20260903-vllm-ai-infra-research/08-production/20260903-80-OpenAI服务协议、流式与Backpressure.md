# OpenAI 服务协议、流式与 Backpressure

## 文档契约

- **唯一问题**：内部request/output语义如何成为HTTP/SSE客户端可观察协议，并在慢客户端下保持有界。
- **In scope**：endpoint validation、SSE lifecycle、disconnect/cancel、queue/backpressure、client-visible timing。
- **Out of scope**：core OutputProcessor（27）、admission policy（13）、安全认证（85）。
- **证据基线**：vLLM `bb363db9...` 的 entrypoints/openai、async engine/output processor。

API acceptance、EngineCore admission、first core token、first SSE write、client receipt、final core output、final SSE和socket close是不同线性化点。TTFT必须说明测到哪一层；HTTP 200后内部失败需要协议化error，不能改回status code。

流式delta要维护choice/index、role/content/tool fields、finish reason和usage。stop string跨token时的裁剪由27定义，HTTP层只序列化最终delta。断连应触发幂等abort，但网络半开检测有延迟，期间资源仍在消耗。

慢客户端会填满per-request/output queues。策略可限buffer、暂停读取core、取消请求或断开客户端；若阻塞共享output loop，会把一个租户的backpressure传播给全部请求。queue bytes比item count更可靠。

验证覆盖客户端不读/慢读/重连、disconnect race、重复cancel、超大logprobs/tool output、proxy buffering和timeout；追踪各线性化时间与buffer峰值。兼容OpenAI schema不等于传输错误和backpressure语义完全一致。

## 状态与错误映射

| 内部阶段 | 客户端可见 | 透明重试 |
|---|---|---|
| validation拒绝 | 4xx/结构化错误 | 修正请求后 |
| 已接受未输出 | 可能无 delta | 需 attempt/幂等策略 |
| 已输出 token | SSE delta 已见 | 默认不可 |
| engine失败 | error/断连 | 新 attempt并去重 |
| client断连 | 无后续输出 | server异步 abort |

固定入口在 `vllm/entrypoints/openai/`、`vllm/v1/engine/async_llm.py` 和 output processor。`SOURCE_IMPLEMENTED`：HTTP/SSE/abort路径；`RECOMMENDATION`：queue按 bytes+age 双限；`UNKNOWN`：具体 proxy 的断连与 buffering。
