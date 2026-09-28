# Executor 抽象与 Worker RPC

## 文档契约

- **唯一问题**：EngineCore 如何把一个逻辑 step 委派给单设备或分布式 workers。
- **In scope**：Executor class selection、collective RPC、control/data-plane boundary、failure/completion contract。
- **Out of scope**：worker 启动（41）、runner tensor contract（42）、进程拓扑实现（66）。
- **依赖输入**：23 的 EngineCore step；28 的 completion 分级；60 的 rank group。
- **唯一输出/Owner**：worker 集合 RPC 投递、聚合、异常与 epoch 合同；不拥有 device kernel completion。
- **相邻篇不得重述**：41 只写单 worker readiness；66 只写 launcher/placement。
- **证据基线**：vLLM `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` 的 `vllm/v1/executor/abstract.py::{Executor.get_class,collective_rpc,execute_model,determine_available_memory}`、uniproc/multiproc/ray executors、`worker_base.py`。

## 1. 抽象职责

`Executor` 将 core 的单一调用面映射到一个或多个 `WorkerBase`：初始化 device、加载模型、收集 KV spec/available memory、初始化 cache、compile/warmup、执行 model、采样、profile、sleep/wake/reconfigure。它不是 kernel abstraction，而是跨进程/跨 rank orchestration boundary。

固定 revision 依据 `distributed_executor_backend` 选择 `uni`、`mp`、Ray、external launcher 或可解析的自定义 class。`supports_pp`、`uses_ray` 是能力信号，不等于完整 feature matrix。

## 2. RPC 的语义

`collective_rpc(method, args, kwargs, non_block)` 向所有 workers 下发控制调用或执行描述；源码注释建议该通道用于 control messages，而 bulk tensor/KV payload 应使用独立 data plane。`SchedulerOutput` 本身是启动 data-plane work 的控制 descriptor，并非大 tensor payload。同步调用返回 per-worker list；非阻塞调用返回 Future。`execute_model` 最终取聚合结果的首项作为 driver output，这要求其余 ranks 的副作用和错误已被正确纳入 executor completion。

必须定义四项：目标 worker 集、调用顺序、超时/失败聚合、completion level。否则“Future 已完成”无法证明所有 rank device writes 可见。

## 3. 分布式原子性

一个 batch 对 TP/PP ranks 是联合事务。以下都不能被误当成功：仅 driver 有 output；一个 worker timeout 但其他 worker继续；collective 已 launch 但未完成；旧 epoch 的迟到 reply 被新 executor 接收。安全策略通常是整 core fail-stop 或明确的全 rank recovery，不能请求级重试一个 collective 的局部 rank。

RPC 应携带/隐含 executor epoch 与 step sequence。若 transport 只靠 FIFO，process restart、queue reconnect、async run-ahead 会使 stale reply 难以检测。

## 4. 自定义 backend 的最小实现面

第三方不能只覆盖 `execute_model`。至少要兑现：device/rank binding、distributed init、model load、memory profiling、KV spec/layout negotiation、cache initialization、warmup/artifact preparation、execute/sample、shutdown/failure callback，以及 sleep/profile 等被启用功能的明确支持或拒绝。

如果 backend 以 AoT engine 执行，Executor 仍需把动态 `SchedulerOutput` 映射到有限 profiles；无法表示的 shape 应在 admission/validation 阶段拒绝，而不是 worker 内随机失败。

## 5. 架构验收

- 单 rank 与多 rank RPC 结果/异常矩阵；
- worker crash、hang、迟到 reply、driver crash；
- non-block Future 的 host/device completion fence；
- reconfigure/shutdown 与 in-flight step 的 epoch 隔离；
- 大 metadata 不误走低效 control channel；
- 自定义 executor 的 capabilities 可机器读取。

**结论**：Executor 是 execution transaction coordinator。插件“能 import Executor”只证明发现，不证明 collective、failure、completion 或 feature semantics。

## 6. 证据分类

- `SOURCE_IMPLEMENTED`：上述 class selection、RPC overload、`execute_model` 首结果返回和 worker methods 均存在于固定 SHA。
- `INFERENCE`：跨 rank batch 应作为联合事务，是从 collective 数据依赖得出的架构要求；抽象接口本身没有完整声明两阶段提交。
- `RECOMMENDATION`：reply 携带 process epoch/step sequence，失败后 quarantine 整个 rank group 的当步副作用。
- `UNKNOWN`：各 Executor 实现的 Future 在 host result、device event、all-rank completion 中具体对应哪一级，需逐实现和运行证据确认。
