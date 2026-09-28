# Expert Parallel、AllToAll 与 EPLB

## 文档契约

- **唯一问题**：MoE tokens 如何路由到分布式 experts，并在负载倾斜下保持正确性和容量。
- **In scope**：top-k route、dispatch/combine、all-to-all、expert placement、EPLB/elastic state。
- **Out of scope**：TP基础（61）、硬件互联比较（93）、model loader通论（43）。
- **依赖输入**：60 的 EP group；43 的 expert weights；42 的 token mapping。
- **唯一输出/Owner**：route→dispatch→expert→combine 与 placement/rebalance transaction。
- **相邻篇不得重述**：61 不拥有 expert route；93 只评价 all-to-all物理代价。
- **证据基线**：vLLM `bb363db9...` fused MoE、distributed all2all、`eplb/`、`elastic_ep/`。

router为每token选top-k experts及权重；dispatch按destination打包，通过all-to-all送到expert ranks，执行MLP，再反向combine恢复原token顺序并加权。任何drop、重复、order或weight错误都会改变模型。

通信bytes由tokens×top-k×hidden×dtype决定，但分布极不均。每rank buffer需按capacity规划；溢出策略（padding、drop、reroute、fallback）必须符合模型语义。平均负载不能覆盖hot expert p99。

EPLB观察expert load并复制/迁移/重排placement。它改变weights residency、route table和collective mapping，需barrier/epoch原子切换；in-flight batch不能一半使用旧placement。迁移成本和额外memory需进入收益判断。

EP与TP/DP组合会改变group坐标；quant/LoRA expert weights又扩大feature矩阵。验收以synthetic极端route、真实trace、不同all2all backend、rank failure和rebalance交错测试，报告per-expert load、bytes、tail、drop=0证据与输出reference。

## 固定源码落点

`vllm/model_executor/layers/fused_moe/` 持有 route、dispatch、kernel 与 backend 选择；`vllm/distributed/parallel_state.py` 建立 EP group；`vllm/config/parallel.py::ParallelConfig` 持有 `enable_expert_parallel`、`all2all_backend`、`enable_eplb`；`vllm/distributed/eplb/` 与 `elastic_ep/` 管理统计、placement 和重配置。

## All-to-all backend 不是同义实现

固定 config 默认 `allgather_reducescatter`，还可能选择平台/依赖支持的其他 backend。源码会对部分不适用选择回退或改写；有效 backend 必须进入启动证据。不同实现对 token packing、counts exchange、buffer capacity、dtype 与 completion 的要求不同，性能结果不能只写“EP size”。

## Placement 事务

```text
collect load statistics at epoch e
 -> compute new logical expert→physical slot mapping
 -> ensure weights resident / buffers ready
 -> barrier all participants
 -> atomically publish epoch e+1 route table
 -> drain e batches, reclaim old replicas
```

若先发布 route 再完成 weight residency，会把 token 发到空 slot；若旧 batch 与新 mapping 共享 slot，则 combine 可读错 expert output。elastic state 还会改变 effective DP/EP size，process group 与 model metadata 必须一起换 epoch。

## 当前 guards

- elastic EP 需要 `enable_eplb=True`。
- 固定 config 限制 elastic EP 与 PP>1 的组合。
- backend/platform 可能不支持请求的 all2all 实现并进行 fallback。
- EPLB 的冗余 expert 数占用额外 weights/working memory，应进入 capacity profile。

## 证据分类

- `SOURCE_IMPLEMENTED`：fused MoE、all2all config、EPLB/elastic modules。
- `INFERENCE`：placement publish 是 route correctness 的线性化点。
- `RECOMMENDATION`：记录每 expert tokens、drop、queue、mapping epoch、migration bytes。
- `UNKNOWN`：vendor all2all 的 fault recovery、尾延迟与重配置数值一致性。
