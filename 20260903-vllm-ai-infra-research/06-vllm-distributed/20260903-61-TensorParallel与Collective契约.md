# Tensor Parallel 与 Collective 契约

## 文档契约

- **唯一问题**：单层 tensor 如何分片，以及哪些 collective 恢复等价模型语义。
- **In scope**：column/row parallel、vocab/logits、attention heads、all-reduce/gather/reduce-scatter、latency model。
- **Out of scope**：process groups（60）、PP（62）、EP（64）。
- **依赖输入**：60 的 TP group；43 的 parameter geometry；44 的 attention backend。
- **唯一输出/Owner**：TP parameter/activation/logits shard 与 collective 恢复语义。
- **相邻篇不得重述**：93 只评价互联；64 只拥有 expert all-to-all。
- **证据基线**：vLLM `bb363db9...` tensor-parallel layers与communication ops。

column-parallel线性层切输出维，后续若可继续分片则不立即gather；row-parallel切输入维并对partial outputs归约。embedding/vocab和LM head还需处理padding vocabulary与distributed sampling。attention要求query/KV heads可分或定义replication，GQA在TP degree超过KV heads时尤其敏感。

collective不是可替换的“同步”：all-reduce、all-gather、reduce-scatter的bytes、buffer形状和后续ownership不同。自定义all-reduce只有在拓扑、peer access、alignment和world size满足时才合法；fallback到标准backend需可观测。

性能近似由计算缩短与 collective `latency + bytes/BW` 竞争。decode小矩阵时latency主导，扩大TP可能变慢；prefill大矩阵更易摊薄通信。跨节点TP通常比单机高带宽互联风险更高。

正确性要求所有ranks同序调用、分片边界覆盖且不重叠、归约dtype/顺序满足tolerance、async collective在consumer前完成。验证需和单rank reference、覆盖非整除/quant/LoRA/spec、记录真实algorithm/link和p99，而非只看world size。

## 固定源码落点

`vllm/distributed/communication_op.py` 的 `tensor_model_parallel_all_reduce/all_gather/reduce_scatter` 消费 TP group；`vllm/model_executor/layers/linear.py` 的 column/row-parallel layers 决定 parameter shard 与 collective placement；`vocab_parallel_embedding.py` 处理 padded vocabulary、mask/reduction。attention head 的分配还由 model/attention config 与 backend metadata 决定。

## Shape 与 ownership 表

| 路径 | 本地输入 | collective 后语义 | consumer |
|---|---|---|---|
| column parallel | replicated/适配后的 `[..., K]` | `[..., N/tp]` 或 gather 为 `N` | 下一分片层/完整 consumer |
| row parallel | `[..., K/tp]` | partial `[..., N]` all-reduce | replicated activation |
| vocab parallel | local vocab logits | distributed top-k/sampling 或 gather | sampler/logprob |
| GQA attention | Q/KV head shards或KV复制 | attention partial/full output | output projection |

通信 buffer 的 owner 在 collective 完成前是 communicator/op；consumer 只有在同步 completion 后才能覆盖。`async_op` 或 custom all-reduce 返回的 handle 必须进入依赖链，不能以 Python return 代替 device event。

## 固定 revision 决策/guard

- TP size 必须与 layer/head/quant packing 的 divisibility contract 匹配；具体模型可能采用 padding或KV replication。
- custom all-reduce 是否可用由 platform、world size、topology/peer access等共同决定；fallback 需要记录 effective backend。
- distributed sampling/logprob 需要保持 vocab padding mask，不能让 padded IDs 被选中。
- TP 与 LoRA/quant/custom op 的组合必须验证对应 shard loader 和 kernel，不能由 dense FP16 path 推断。

## 证据分类

- `SOURCE_IMPLEMENTED`：communication ops 与 parallel layers 的调用面。
- `INFERENCE`：decode 扩 TP 可能被 collective latency 主导。
- `RECOMMENDATION`：每个关键 collective 输出 bytes、algorithm、link domain、wait time。
- `UNKNOWN`：各 vendor backend 的真实归约顺序、数值误差和 p99。
