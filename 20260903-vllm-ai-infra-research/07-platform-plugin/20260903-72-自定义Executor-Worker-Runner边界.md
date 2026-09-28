# 自定义 Executor、Worker、Runner 边界

## 文档契约

- **唯一问题**：第三方硬件应该在哪一层复用 vLLM，在哪一层替换执行实现。
- **In scope**：复用scheduler/core、custom Executor/Worker/Runner、private engine/fork的选择与责任。
- **Out of scope**：通用Executor细节（40）、Platform API（71）、具体vendor路径。
- **证据基线**：vLLM `bb363db9...` 抽象类与四个插件固定revisions。

层次选择：只需不同kernels时复用runner并注册ops/backends；runtime/device init不同则自定义Worker；进程/collective不同则自定义Executor；input batching/KV ownership/AoT profiles不同则自定义Runner；若连request/scheduler协议都替换，则实质成为private engine，不应仍声称继承vLLM全部语义。

每下沉一层，复用更多上游功能但受动态tensor/contracts约束；每上移替换，硬件自由度增加但必须自己承担state machine、abort、streaming、metrics、P/D、LoRA/spec/MM等语义。

架构评审要画“owner delta”：哪些状态仍由vLLM core拥有，哪些进入vendor runtime；每个共享状态的同步/epoch/completion；unsupported features在哪里拒绝。最危险的是core认为拥有paged KV，而runtime内部另建cache且二者没有一致性协议。

选择标准包括execution regime、dynamic shape、distributed runtime、kernel coverage、发布节奏和故障模型。验收不以继承class数量，而以端到端contract和negative paths。private scheduler若只兼容OpenAI endpoint，应准确称API兼容层，而非vLLM backend等价。

## 替换层决策表

| 差异来源 | 最窄建议层 | 新 owner 责任 |
|---|---|---|
| op/kernel/layout | AttentionBackend/CustomOp | schema、dispatch、numeric |
| device init/memory | Worker | context、profile、completion |
| process/collective | Executor | placement、RPC、failure |
| batch/KV/artifact | Runner | row/block/shape/submit |
| admission/step model | Scheduler | request state、fairness、abort |

固定接口从 `v1/executor/abstract.py::Executor`、`v1/worker/worker_base.py::WorkerBase`、runner 与 scheduler interface 对照。每替换一层都要列父层仍调用的方法、返回、异常和 shutdown。

最低等价证明覆盖 submit→completion、abort、preempt、stream output、KV reuse、feature validation 与 metrics。`INFERENCE`：越靠近 core 状态空间越大；`RECOMMENDATION`：维护 owner-delta/unsupported matrix。
