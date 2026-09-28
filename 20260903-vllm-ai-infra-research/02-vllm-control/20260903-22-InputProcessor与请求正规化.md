# InputProcessor 与请求正规化

## 本篇唯一主问题

`InputProcessor` 在 request 进入 EngineCore 之前确立了哪些模型、长度、params、LoRA、多模态和 platform 不变式？

## In scope

`vllm/v1/engine/input_processor.py` 的校验/正规化责任、sync/async preprocessing、generation config 合并、prompt 表示和 `EngineCoreRequest` 构造。

## Out of scope

不解释 endpoint schema/chat template 语义，不展开 multimodal encoder runtime，不解释 Scheduler 的 admission。

**固定源码基线**：vLLM `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` 的 `vllm/v1/engine/input_processor.py::InputProcessor` 与 `EngineCoreRequest` 构造路径。

## 类的上下文

`InputProcessor.__init__()` 保留以下 config：

- model/cache/LoRA/scheduler/speculative/structured-output/observability；
- renderer/tokenizer；
- generation config fields；
- multimodal capability 和 encoder cache budget；
- platform request validation 入口。

这表明 input normalization 不只是 tokenization，而是第一个将请求与当前 model/backend effective config 相交的地方。`SOURCE_IMPLEMENTED`

## 同步与异步边界

raw prompt parsing、tokenization 和 multimodal processing 可阻塞。类在初始化时通过 `make_async(self.process_inputs, executor=self.renderer._executor)` 创建 `process_inputs_async`，目的是避免 async caller 的 event loop 被阻塞。`SOURCE_IMPLEMENTED`

这不保证 preprocessing 没有竞争：线程池大小、tokenizer/processor 线程安全、MM cache 并发、CPU/NUMA 和 cancellation 仍需要 runtime 测试。

## Params 校验

### Generation

`SamplingParams.verify()` 接收 model、speculative、structured-output 和 tokenizer 上下文。另外：

- model 必须有 generation task；
- sampling distribution replay 要求 `temperature > 0` 且 `top_k > 0`，后者用于限制 mask 体积和避免 OOM；
- thinking token budget 要求 reasoning config/parser；
- trace replay token IDs 要求 model config 开启 trace replay。

### Pooling

model 必须包含对应 pooling task；未指定 task 时会按 `token_embed`、`token_classify`、`plugin` 顺序选可用值，然后执行 `PoolingParams.verify()`。

因此“模型能加载”不等于它支持所有 task/params。

## Platform validation 的位置

`process_inputs()` 在 renderer 产生 `EngineInput` 后调用 `current_platform.validate_request(engine_input, params)`。这允许 platform 在进入 core 前拒绝特定输入/参数组合。`SOURCE_IMPLEMENTED`

限制：platform hook 能观察的只是当前请求和 static config，不自动拥有当前 KV pressure、compile cache、device health 或全局 queue 状态。它是 request compatibility validation，不是完整 runtime admission controller。

## Prompt 的三种表示

1. **token IDs**：`prompt_token_ids` 有值，`prompt_embeds=None`。
2. **embeddings**：`prompt_embeds` 有值，token ID 可缺失；core `Request` 会为 token history 创建占位值。
3. **mixed mode**：token IDs 与 embeddings 都存在，`prompt_is_token_ids` 按位置指明真 token 与 embed 位置。

mixed mode 使长度、position、hash、embedding gather 和多模态覆盖变成一个联合契约。后端不能只消费 token ID list 而忽略 mask。

## Generation config 和 tokenizer 合并

`SamplingParams` 被 clone 后：

1. 若 `max_tokens` 未设置，用 `max_model_len - prompt_len`；
2. `update_from_generation_config()` 合并 generation config 和 renderer EOS；
3. 若有 tokenizer，`update_from_tokenizer()` 补充 tokenizer-dependent 语义；
4. trace replay 可截断 token trace，重写 max/min token、ignore EOS、stop strings/IDs。

该顺序是语义的一部分；架构测试应对比最终 effective params，不只测单个 helper。

## LoRA 校验

- request 带 LoRA 但 engine 未启用 LoRA 时拒绝；
- 当前代码警告 per-LoRA tokenizer 支持已 deprecated，默认使用 base model tokenizer；
- 开启 tower connector LoRA 时，multimodal identifier 将 LoRA name 加入 MM hash，防止不同 LoRA 的 encoder result 错误命中。

最后一点是 cache key 需要包含会改变计算结果的 adapter identity 的直接源码证据。

## Multimodal cache 注入

`inject_into_mm_cache()` 允许外部已处理 tensor 注入 renderer MM cache，使用 hash 和空 prompt update list，并更新 cache stats。异常只记 warning。`SOURCE_IMPLEMENTED`

该 fail-open 行为的语义是“注入失败不终止请求”，不表示后续一定可从其他路径获得必需 data。P0/P1 cache drift 另有 output miss/retry 协议；正确性需要 MM 集成和故障测试。

## 请求正规化后的不变式

1. generation 与 pooling params 恰有一种。
2. request task 属于 model supported tasks。
3. prompt token/embed/mixed representation 长度合法。
4. `max_tokens` 已有有效值，与 max model length 协调。
5. platform-specific request validation 已通过。
6. LoRA 使用与 engine config 协调。
7. DP rank 若指定，在当前 local/global engine 范围内。
8. MM feature/hash 与 tower LoRA identity 协调。

这些不变式不包括“有足够 KV/worker/profile 可执行”，后者在 scheduler/runner 阶段决定。

## 失败与观测性

| 失败 | 期望边界 | 应观测的信息 |
|---|---|---|
| unsupported task/params | request validation | 类型化原因，不记 prompt |
| prompt too long | pre-core reject | effective max length、prompt length |
| invalid DP rank | pre-core reject | desired rank、valid range |
| platform reject | pre-core reject | backend、feature/mode，去敏 |
| tokenizer/renderer fault | frontend fault | model/tokenizer revision |
| MM processor/cache fault | frontend/cache protocol | modality/hash class，不记原始 media |

## 架构决策点

1. preprocessing thread pool 是否有界且受 cancellation/backpressure 控制？
2. desired/effective params 与拒绝原因能否去敏审计？
3. platform validation 只做 static compatibility，还是意外依赖 runtime mutable state？
4. mixed embeds/token 是否被目标 runner/attention backend 真正支持？
5. MM/LoRA cache key 是否包含所有改变 encoder result 的因素？
6. 一个 request validation 失败是否会留下 collector/cache reservation？

## 证据结论

本篇所述校验、正规化和 struct 构造为 `SOURCE_IMPLEMENTED`。并发 MM cache、线程池 backpressure 和所有 platform 组合的运行时正确性为 `UNKNOWN`。

## 本篇输出契约

`23`–`28` 从已通过正规化的 `EngineCoreRequest` 开始；模型/LoRA/MM 数据面语义由 `04-vllm-execution` 展开。
