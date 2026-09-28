# Structured Output Grammar 状态机

## 文档契约

- **唯一问题/Owner**：grammar 如何编译、阻塞请求、逐 token 转移并产生 allowed-token mask。
- **依赖输入**：24 的 request 状态；46 的 sampling pipeline；22 的请求正规化。
- **唯一输出**：grammar identity、readiness、mask application、finish/error contract。
- **相邻篇不得重述**：LoRA residency 归 49；组合能力归 49B；HTTP schema 归 80。
- **证据基线**：vLLM `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` 的 structured-output manager、grammar backend 与 scheduler grammar fields。

## 1. 状态机

请求 grammar 状态可建模为 `RECEIVED → COMPILING → READY(state_0) → ADVANCING(state_n) → ACCEPTING/FAILED/CANCELLED`。编译可能异步，因此 scheduler 有等待 grammar readiness 的状态；READY 前不得无约束采样。

每次提交 accepted token 后 grammar state 恰好推进一次，再为下一 position 生成 allowed mask。speculative verification 若一次接受多个 token，必须按顺序推进或批量证明等价；rejected draft 不能推进最终 grammar。

## 2. Identity 与 Cache

key 绑定规范化 grammar/schema content、backend/compiler 版本、tokenizer vocabulary/revision 和相关 options。相同 JSON 文本在不同 tokenizer 下 allowed token 集合不同。compiled automaton 可只读共享，但每请求 cursor/state 独立。

## 3. Sampling 接口

mask 应在明确 pipeline 位置作用于 logits；全禁用集合、NaN 或 tokenizer mismatch 要 fail closed。返回 logprobs 需说明是 mask 前还是 mask 后 distribution。stop/EOS 与 grammar accepting 冲突时，优先级必须由 API 合同定义。

## 4. 资源与故障

复杂 grammar 可消耗 CPU/time/memory，是 pre-core admission 面；需 size/time limits、bounded pool 和 cancel。编译异常不能让请求永久 waiting。worker/frontend 重启后 cursor 是否可恢复应明确；默认请求级失败比猜测状态安全。

## 5. 验证与证据

覆盖无效/空/大 grammar、tokenizer revision、Unicode、all-masked、compile cancel/timeout、spec partial accept、preempt/resume，并逐步和 reference automaton 比较。

- `SOURCE_IMPLEMENTED`：固定 revision 存在 grammar 等待状态和 structured-output 路径。
- `INFERENCE`：上述完整状态机是从调用面抽象出的审计模型。
- `RECOMMENDATION`：compiled artifact 与 per-request cursor 分离 owner。
- `UNKNOWN`：第三方 device sampling 能否等价应用动态 mask。

