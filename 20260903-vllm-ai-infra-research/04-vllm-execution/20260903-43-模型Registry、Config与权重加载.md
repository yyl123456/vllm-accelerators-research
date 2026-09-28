# 模型 Registry、Config 与权重加载

## 文档契约

- **唯一问题**：一个模型标识如何解析为受支持 architecture，并把 checkpoint tensors 安全映射到并行模型。
- **In scope**：registry、HF config、task/interface inspection、loader、name mapping、sharding/quant weights。
- **Out of scope**：模型 forward kernel（44-45）、artifact compile（53）、LoRA（49）。
- **依赖输入**：00 的证据规则；71 的平台能力；60-64 的并行坐标。
- **唯一输出/Owner**：model/config/checkpoint 到已验证 sharded parameters 的兼容合同。
- **相邻篇不得重述**：53 只把本篇 identity 纳入 artifact key；49 只处理增量 adapter。
- **证据基线**：vLLM `bb363db9...` 的 `model_executor/models/registry.py`、`config/model.py`、`model_loader/`、models utils。

## 1. 支持声明的分层

“模型支持”至少分为：architecture 可解析；config 可验证；class 实现所需 task/interface；checkpoint 可完整加载；目标 backend ops 可执行；输出质量经验证。registry entry 只证明第一层。

动态 import/lazy registry 可避免主进程污染 device context，但错误会延迟到 inspection/load。remote code 还扩大安全边界，必须由部署策略显式允许。

## 2. config 是语义源

config 决定 architecture、hidden/head geometry、rope、dtype、max length、quantization、MM/encoder、MoE、tie weights 等。CLI override 形成新的 effective config；文档和 artifact key 应记录解析后的值，而非只记录原始模型名。

## 3. loader 合同

loader 选择 checkpoint format/source，枚举 tensors，做 name mapping、stacked parameters、TP/EP shard selection、dtype/quant packing，然后验证 missing/unexpected/duplicate weights。`AutoWeightsLoader` 等 helper 不能替代模型特定 mapping 的审计。

一个安全 loader 要拒绝：shape 静默 reshape、未知 tensor 被无条件忽略、同一 parameter 多次覆盖、TP shard 非整除、quant metadata 缺失、tie alias 被复制成不一致参数。

## 4. 分布式与 IO

每 rank 只读取所需 shard 可减少 host memory/IO，但要求 checkpoint layout 与 parallel placement 已知。否则先全量读取再切分会在大模型启动时耗尽 host memory。远程下载、cache、streaming load 与权重 reload 各自有版本/原子性风险。

## 5. release tuple

模型兼容记录应固定：model revision/config/tokenizer、vLLM SHA、plugin/runtime/driver、loader/quant format、TP/PP/EP、task、关键 flags。相同 model family 不代表不同 revision 权重命名或 config 相同。

## 6. 固定源码调用链与失败面

`vllm/model_executor/models/registry.py::ModelRegistry` 解析 architecture；`vllm/model_executor/model_loader/` 依据 `LoadConfig` 选择 loader；各模型 `load_weights()` 将 checkpoint names 映射到 parameter shards。`ModelConfig`、`QuantizationConfig`、`ParallelConfig` 与 platform config 共同决定最终 class/loader/layout。

```text
HF/config artifact -> architecture registry -> model class
 -> construct empty parameters -> loader name mapping/sharding
 -> quant/post-load processing -> per-rank completeness check
```

必须拒绝 missing/unexpected/shape-incompatible shards，且 tied/stacked/fused parameters 需有明确映射；lazy/streaming loader 在 worker ready 前必须证明必要 weights resident。`SOURCE_IMPLEMENTED`：registry/loader/model `load_weights` 协议；`RECOMMENDATION`：发布记录 config/weight digest 与每 rank shard manifest；`UNKNOWN`：vendor 转换格式的数值等价性。

## 7. 验证

静态检查 loaded parameter coverage 与 bytes；小输入对 reference logits；多 rank 重构或 checksum shard；tie/quant/MoE/PP 专项；cold start 峰值和时间。dummy weights 仅验证 topology/compile，不是模型正确性证据。
