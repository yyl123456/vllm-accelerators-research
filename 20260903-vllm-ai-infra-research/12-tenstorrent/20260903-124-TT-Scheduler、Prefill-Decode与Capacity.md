# TT Scheduler、Prefill/Decode 与 Capacity

## 文档契约

- owner：TT step 形成、prefill/decode 隔离、lane scheduling 与 token/KV capacity。
- 依赖输入：`23`–`26` 的 vLLM scheduler、`123` 的注入路径。
- 唯一输出：TT 调度不变式和容量模型。
- 不讨论：trace/RNG 或 release manifest；由 `125`、`126` 持有。

## 实现对象

- `scheduler.py::TTScheduler(AsyncScheduler)`：TT 普通调度策略。
- `lane_scheduler.py::TTLaneCoordinator(SchedulerInterface)`：单次 shared-device execute 下的多 lane 协调。
- `lane_scheduler.py::TTStepPlan`、`_LaneStepState`：将各 lane 的请求状态汇总为一个设备 step。
- `model_input.py::TTModelInput`、`TTDecodeReloadPlan`：host scheduler output 到 runner 的 TT 执行描述。

## 核心不变式

一个 TT model step 必须是 prefill-only 或 decode-only。混合 batch 若模型 generator 没有相同语义，会破坏 shape/trace/KV update contract。普通模型的 prefill 通常不切 token chunk；固定 plugin 文档指出 Gemma 4 有专门 token-chunked prefill 路径，未完成 prompt 的 chunk 不产生 token。

lane 模式不是传统多进程 DP。多个 vLLM logical lanes 由 `TTLaneCoordinator` 收集，在一个 engine/mesh 上形成一次 shared execute，再把结果归属回 lane。其正确性依赖 request→lane→row 的稳定映射、空 lane padding、abort 清理和每 lane backpressure。

## Capacity 不是显存 profile

`TTWorker.determine_available_memory()` 不执行通用 device-memory profiling，而调用 `get_num_available_blocks_tt()`。后者从 model class 的 per-device `max_tokens_all_users` 预算、device count、block size、hybrid/sliding-window padding 与 block-output canvas headroom 推导 block count，再向 vLLM engine-side KV planner 发布 override/等价 byte budget。

因此：

- `max_tokens_all_users` 是全部并发请求共享的物理 token pool；
- `max_model_len` 是单请求上限；
- `max_model_len=-1` 可自动贴合“一条请求能装下”的最大长度，但可能把并发降到约 1；
- HF-derived max length 超出 pool 时应 startup fail，而不是运行后 OOM。

## 调度与容量状态表

| 状态 | 可执行动作 | 必须保留的状态 |
|---|---|---|
| new prefill | 分配 KV、选择支持 bucket | request/block ownership |
| chunked prefill | 追加 chunk，不发 token | processed-token count |
| transition | 固化 KV 与 row/lane mapping | reload plan/version |
| decode | 每步更新 token/KV | sampled token、position |
| preempt/abort | 释放 scheduler 与 model state | completion/fence 后释放 |

## 高风险失败

- block 数按 nominal DRAM 而非 model budget：启动或长上下文 OOM。
- scheduler 释放 block 时设备仍异步读写：use-after-free。
- lane 复用 row 但旧 KV/reload state 未清：跨请求污染。
- 自动关闭 prefix cache 后仍按命中容量估算：SLO/并发预测失真。
- block-output 一次产生多 token，却按单 token step 提交：逻辑长度与物理 canvas 分裂。

## 结论分类

- `SOURCE_IMPLEMENTED`：上述 scheduler、lane、capacity 符号与 README capacity contract。
- `INFERENCE`：lane 模式把故障域从 rank 扩大到 shared execute。
- `RECOMMENDATION`：容量报告必须同时输出 pool、单请求上限、预计并发和 headroom 构成。
- `UNKNOWN`：不同模型/SKU 下保守预算与真实可用容量差距。

