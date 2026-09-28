# Pipeline Parallel 与 Microbatch

## 文档契约

- **唯一问题**：模型 layers 分 stage 后，请求 state、intermediate tensors 和调度如何跨 stage推进。
- **In scope**：layer partition、stage I/O、microbatch、bubbles、finish/error传播。
- **Out of scope**：TP（61）、DP routing（63）、launcher（66）。
- **依赖输入**：60 的 PP group；43 的 layer placement；42 的 batch identity。
- **唯一输出/Owner**：stage I/O、microbatch lifecycle 与跨 stage commit/abort。
- **相邻篇不得重述**：66 只决定进程放置；23 只消费 PP completion。
- **证据基线**：vLLM `bb363db9...` PP model helpers、executor capability与distributed code。

PP按layer范围分配weights/KV；首stage接收tokens/embeddings，中间stage传 hidden/intermediate tensors，末stage计算logits/sampling。每stage对同一request的位置、batch order和step sequence必须一致。

inference microbatch用来流水化多个token groups；收益受stage balance和传输影响，bubble约由最慢stage与可填充microbatches决定。decode依赖前一步token，跨step流水化受限；不能直接套训练1F1B模型。

layer partition不能只按参数量：attention/MLP/MoE、KV bytes、activation和device差异会不均。stage boundary还决定跨节点activation bytes与failure domain。

故障时某stage partial成功不能提交整体step。abort/preempt需清理各stage KV/runner rows；late activation要带epoch/sequence。验收覆盖不均partition、PP×TP、空/尾microbatch、stage crash/hang、不同shape，并分解bubble/transfer/compute。

## 固定源码落点与调用序

模型层通过 `vllm/model_executor/models/utils.py` 的 PP helpers 判断 first/last rank、missing layers 与 intermediate tensor；模型实现按 `get_pp_group()` 的 rank 装载 layer range。V1 executor/worker 在 PP>1 时选择相应多 worker 路径，device communicator 的 `send_tensor_dict/recv_tensor_dict` 承担 stage payload；最后 stage 才形成可提交输出。

```text
stage 0 tokens/embeddings
 -> local layer range
 -> IntermediateTensors + step/request epoch
 -> send to stage i+1
 -> last stage logits/sample
 -> executor returns whole-step result
```

## 当前配置与未支持组合

| 维度 | 当前需要核查的 guard |
|---|---|
| PP×executor | platform 是否允许 multiproc/Ray/自定义 executor |
| PP×PCP/DCP | `ParallelConfig` world 构造和 backend path 是否覆盖 |
| PP×KV transfer | connector 可能单独拒绝 PP>1，不能只看 PP core support |
| PP×pooling/MM/spec | model runner 与 output owner 是否只在 last rank成立 |
| elastic EP | 固定 config 明确限制 PP>1 |

## Stage transaction

每个 microbatch descriptor 至少含 step epoch、batch row mapping、tensor schema 和 sender/receiver stage。stage i 失败后，其他 stage 的 partial KV/activation 不得提交；timeout owner 必须能唤醒等待 recv 的 peers。重试若不重建所有 stage 的一致状态，只会把旧 intermediate 注入新 step。

## 证据分类

- `SOURCE_IMPLEMENTED`：PP group、model layer partition、intermediate send/recv 与 executor branches。
- `INFERENCE`：whole-step commit 和 late-message epoch 是正确性所需。
- `RECOMMENDATION`：按 stage 记录 compute/transfer/queue/bubble 与 request epoch。
- `UNKNOWN`：固定 revision 对每个 vendor 的 PP×高级功能真实覆盖。
