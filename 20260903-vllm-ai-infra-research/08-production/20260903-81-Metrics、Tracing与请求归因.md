# Metrics、Tracing 与请求归因

## 文档契约

- **唯一问题**：如何把跨frontend/core/worker/device的事件归因到同一请求、step和release tuple。
- **In scope**：IDs、timestamps、metrics cardinality、traces、logs、clock domains、critical path。
- **Out of scope**：benchmark方法（82）、failure recovery（84）、compile专属metrics（55）。
- **证据基线**：vLLM `bb363db9...` observability/stat logger/tracing/event structs。

关联键至少含external/internal request ID、parent/child、engine/replica、step/scheduler sequence和process epoch。request ID不应作为Prometheus label导致无界cardinality；应进入trace/log，聚合metrics使用model/backend/status等有界维度。

各进程monotonic clock不可直接比较绝对值；跨进程critical path需trace context与clock校准，或只比较同域durations。wall clock用于人类关联但会受NTP跳变。

核心分段：frontend preprocess/queue、scheduler waiting、KV/grammar wait、execute queue、H2D/forward/collective/sample、output queue/SSE、connector transfer。还要记录cache hit、preempt、fallback、compile、selected kernel/backend和effective batch。

采样策略不能丢掉罕见错误/长尾；错误trace全采，正常请求按率或tail采样。日志脱敏prompt、tokens、headers和remote URLs。验证用synthetic request贯穿所有进程，检查parent-child/P-D/DP重试归因、clock不误减和metrics cardinality。

## 最小事件 schema

```text
clock_domain, monotonic_ts, process_epoch, engine/replica/rank/device
request_id, attempt, parent, step, batch_row
model/backend/artifact/effective_config, event, duration, bytes/tokens, status
```

`vllm/observability/`、`vllm/v1/metrics/` 与 stat logger/tracing hooks 提供入口；vendor runtime需补充 correlation。Prometheus 保存有界聚合，trace/log 保存高基数 identity。

`SOURCE_IMPLEMENTED`：metrics/tracing入口；`RECOMMENDATION`：定义跨进程 propagation 与 clock label；`UNKNOWN`：device/host timestamp 校准。
