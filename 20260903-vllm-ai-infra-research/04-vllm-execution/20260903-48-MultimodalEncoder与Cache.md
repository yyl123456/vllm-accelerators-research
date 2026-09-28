# Multimodal Encoder 与 Cache

## 文档契约

- **唯一问题**：非文本输入如何被预处理、编码、缓存并在正确 token positions 注入 language model。
- **In scope**：media identity、processor outputs、encoder scheduling/cache、placeholder merge、memory/failure。
- **Out of scope**：API schema（21-22）、ViT attention kernel细节（44）、通用多租户安全（84）。
- **依赖输入**：22 的规范化输入；42 的 batch mapping；13 的 encoder admission。
- **唯一输出/Owner**：media identity、encoder scheduling/cache 与 placeholder merge 合同。
- **相邻篇不得重述**：22 停在 EngineCoreRequest；49B 只记录组合交叉点。
- **证据基线**：vLLM `bb363db9...` 的 `multimodal/`、`v1/worker/mm_encoder_model_runner.py`、GPU `mm/`、encoder cache manager。

## 1. 两段 pipeline

media loader/processor 将 image/audio/video 与 prompt 转为 model-specific tensors、hash/UUID、placeholder/token ranges；encoder runner产生 embeddings/features；language model 在声明 positions 合并它们。三者的 shape/order contract 必须一致。

## 2. identity 与 cache

cache key 不能只用 URL 或文件名；至少要绑定 content、processor/model revision、resize/crop/frame sampling、dtype 与 adapter。客户端提供 UUID 只能在信任其内容不变时作为 identity。错误命中会产生跨请求语义污染。

encoder cache 与 decoder KV cache 是不同资源：前者按媒体 feature bytes，后者按 token state。scheduler 的 encoder budget决定何时算/复用 input；释放顺序需覆盖 request cancel 与共享 media refcount。

## 3. ragged 与 placeholder

图片数量、resolution、视频帧、音频长度使 encoder shape 高度动态；AoT/graph backend 需要 bucketing/padding。placeholder token count 与 feature rows 必须相等或满足模型明确定义；截断 prompt 不应留下孤立 features。

## 4. 隔离与资源风险

解码 media 可能消耗大量 CPU/host memory，必须在 EngineCore admission 之前也有界；恶意压缩炸弹、超长视频、远程 URL 属于输入安全边界。MM cache 应按 tenant/权限隔离，避免从 timing 或复用结果泄露内容。

## 5. 固定源码数据链

`vllm/multimodal/` 负责 processor/registry/cache，`vllm/v1/worker/gpu_model_runner.py` 将 encoder outputs 与 placeholder ranges 合并，model-specific multimodal implementations消费 embeddings。cache identity 需覆盖原始内容或稳定 hash、processor config/version、model/encoder revision、resize/crop/frame sampling 与 dtype/layout。

encoder token budget 与 text token budget 是相关但不同的资源；CPU preprocessing、device encoder、cache residency 和 decoder merge 各有 queue/memory。cache hit 不能跳过 placeholder 数量/range 验证。

`SOURCE_IMPLEMENTED`：multimodal registry/cache 与 runner merge；`INFERENCE`：processor identity 是数值 ABI；`RECOMMENDATION`：按 modality 记录 preprocessing/queue/encode/cache time；`UNKNOWN`：vendor MM×graph/LoRA/spec 的组合。

## 6. 验收

同 media 不同 processor 参数必须 miss；并发共享与 cancel refcount；多 image/order/截断；encoder OOM/timeout；cache eviction 与 LM batching；reference embeddings/logits；记录 CPU preprocessing、encoder queue、device encode、merge 分段延迟。

**结论**：multimodal 支持不是“模型能接 image tensor”，而是额外的 admission、cache 和动态 shape 子系统。
