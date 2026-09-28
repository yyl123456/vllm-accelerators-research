# TPU Runner、Model、KV 与 Sampling

## 文档契约

- owner：request batch 到 JAX arrays、模型、KV 与 token commit 的数据面。
- 依赖输入：`42`、`46`、`132`。
- 唯一输出：TPU runner 的状态 owner 和执行事务。
- 不讨论：compile cache 或全局并行拓扑；由 `134`、`135` 持有。

## 核心对象

- `runner/tpu_runner.py::TPUModelRunner`：组合 KV connector 与 LoRA mixin，拥有模型执行主路径。
- `runner/input_batch.py`、`persistent_batch_manager.py`：将动态请求映射为持久 batch rows。
- `runner/block_table.py`、`kv_cache.py`、`kv_cache_manager.py`：逻辑 blocks 到 JAX-sharded physical arrays。
- `runner/decode_loop.py::TpuSamplingState`：decode/sampling 跨步状态。
- `structured_decoding_manager.py`、`speculative_decoding_manager.py`、`multimodal_manager.py`：各自 request-scoped 状态机。

## 执行链

```text
SchedulerOutput
 -> persistent batch row update
 -> token/position/block-table arrays
 -> sharding placement
 -> compiled prefill/decode function
 -> logits / draft / processed logprobs
 -> sampling + structured/spec constraints
 -> ModelRunnerOutput
 -> EngineCore commits request state
```

row、request ID、KV blocks 和 sampling state 必须拥有同一 epoch。preemption/abort 后的旧 executable completion 不能写入已复用 row。JAX array dispatch 可能异步，host Python 函数返回不应未经 runtime fence 就作为 physical buffer 可复用证明。

## Model implementation 双路径

`models/common/model_loader.py::resolve_model_impl_type` 选择 JAX-native 或 vLLM/TorchAX 实现。两者最终都进入 JAX/XLA，但权重树、normalization、mutation、custom op 和 multimodal preprocessing contract 可不同。功能矩阵必须把 model implementation type 纳入维度。

## KV contract

固定源码用 `NamedSharding`/`PartitionSpec` 创建 KV arrays，并有独立 KV cache manager、continuous block pool、hybrid/Mamba 支持及多种 connector/offload 实现。platform 会根据 Pallas attention 的 page constraints 改写 block size，并显式拒绝 TPU 未实现的 partial prefix-hit state copy。

因此“支持 prefix cache”不能独立于 attention type、Mamba mode、speculative decode、DCP/prefix match unit 和 connector 声称。

## Sampling contract

sampling 不只是 argmax：需要 top-k/top-p/temperature、logprobs、grammar mask、speculative accept/reject 与 RNG progression 一致。固定 platform 明确拒绝 per-request seed；这是一项 API 能力差异。structured/spec managers 和 tests 的存在只证明源码覆盖，设备数值仍需单独证据。

## 验证矩阵

| 维度 | 最小覆盖 |
|---|---|
| batch state | insert/remove/preempt/abort/row reuse |
| KV | full/hybrid、prefix hit/miss、long context、offload |
| sampling | greedy/random、logprobs、grammar、seed rejection |
| spec | zero/full/partial acceptance、rollback |
| model path | JAX-native 与 TorchAX 分别验证 |

## 结论分类

- `SOURCE_IMPLEMENTED`：上述 runner、KV、sampling 与 feature manager objects。
- `INFERENCE`：row/KV/RNG epoch 是最小事务边界。
- `RECOMMENDATION`：故障日志输出 request-row-block-executable epoch 映射。
- `UNKNOWN`：固定 SHA 在 TPU 上的长程数值、stale completion 和 feature-combination 结果。

