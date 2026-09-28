# ModelRunner 与 InputBatch 数据契约

## 文档契约

- **唯一问题**：SchedulerOutput 如何被增量 materialize 成 device batch，并映射回 request outputs。
- **In scope**：persistent batch state、request rows、positions/slots、prepare→forward→sample→output。
- **Out of scope**：调度算法（24-25）、attention kernel（44）、sampling 语义（46）。
- **依赖输入**：23 的 SchedulerOutput；32 的 block table；40 的 execution call。
- **唯一输出/Owner**：request↔row↔token↔slot↔output 的 batch 映射与更新事务。
- **相邻篇不得重述**：44-49 只说明各自 metadata 如何挂到该映射。
- **证据基线**：vLLM `bb363db9...` 的 GPU `model_runner.py`、`input_batch.py`、`gpu_model_runner.py` 与 `outputs.py`。

## 1. runner 不是无状态 `model(inputs)`

在线 decode 每 step 只增量更新 batch：新增/移除/恢复 requests，更新 token history、positions、block tables、sampling metadata、LoRA mapping、MM inputs 和 spec state。`InputBatch` 保存 slot→request 与 CPU/device buffers；runner 将 scheduler 的 request-centric plan 转为 dense/ragged tensors。

## 2. 核心映射

一个 step 至少保持：request ID ↔ batch row；logical token position ↔ flattened input index；token ↔ KV slot；sample position ↔ output row；request ↔ RNG/sampling state。batch compaction 或 swap 若只更新其中一张表，会把一个请求的 token/logprob/KV 写给另一个请求。

## 3. 数据阶段

1. apply scheduler delta，释放 finished/preempted runner state；
2. add/update request rows 和 block tables；
3. 生成 input IDs/embeddings、positions、slot mapping、attention metadata；
4. 选择 padded/graph shape 并传 H2D；
5. forward，可能含 PP intermediate tensors；
6. gather last-token hidden/logits；
7. sample/verify；
8. 形成带 request IDs、tokens、logprobs、connector data 的 `ModelRunnerOutput`。

## 4. 不变量

- scheduler token counters 与 runner materialized tokens 守恒；
- padding 不写入有效 KV，也不参与 sample；
- stale rows 在 block reuse 前清零/覆盖所有有语义字段；
- graph padding shape 不改变有效 sequence 的 position/mask；
- PP/TP ranks 对 batch order 达成一致；
- exception 时，部分更新不能成为下一 step 的合法基线。

## 5. backend 接入含义

自定义 runner可以使用不同 tensor/API，但必须兑现相同 request-level语义。AoT backend 常要求 bucketed shapes；它需把真实 ragged batch 映射到 profile，并定义 padding、max shape 和 output unpadding。device runtime 支持 dynamic shape 不代表 KV slot、sampling state 或 MM ragged metadata 自动兼容。

## 6. 固定源码状态表

主路径位于 `vllm/v1/worker/gpu_model_runner.py::GPUModelRunner`；batch bookkeeping 位于 `vllm/v1/worker/gpu/input_batch.py::InputBatch`，执行入口为 runner 的 `execute_model()`。`SchedulerOutput` 不是 device-ready tensor：runner 先更新 request rows、token/position/block table、LoRA/MM/spec/grammar metadata，再选择 execution shape。

| identity | owner | 复用条件 |
|---|---|---|
| request ID→batch row | InputBatch/runner | 旧 step completion 已退休 |
| row→token/position | runner input buffers | 本 step copy/dispatch 已完成 |
| row→KV blocks | scheduler logical + runner physical | block lease 与 device fence 均结束 |
| row→RNG/spec/grammar | feature managers | token commit/rollback 原子完成 |

`SOURCE_IMPLEMENTED`：固定 SHA 的 GPUModelRunner/InputBatch；`INFERENCE`：上述 identities 需共享 step epoch；`RECOMMENDATION`：调试 dump 输出 row mapping diff；`UNKNOWN`：vendor async runner 的 physical completion。

## 7. 验证

随机执行 add/remove/preempt/abort/compact 序列，以 reference state machine 比较；在 page、bucket、batch边界放 canary；为每个 output 检查 request/step sequence；故障注入 H2D/forward/sample 任一阶段并验证 quarantine。吞吐基准不能替代该状态一致性测试。
