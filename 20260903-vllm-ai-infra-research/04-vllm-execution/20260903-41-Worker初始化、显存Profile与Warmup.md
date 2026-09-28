# Worker 初始化、显存 Profile 与 Warmup

## 文档契约

- **唯一问题**：一个 worker 从进程启动到可安全接收真实请求，要经过哪些有序阶段。
- **In scope**：device/rank init、model load、profiling、KV allocation、compile/warmup、readiness。
- **Out of scope**：权重格式（43）、compile 机制（50-55）、生产 readiness probe（83）。
- **依赖输入**：40 的 Worker RPC；35 的 KV 容量变量；43 的模型加载合同。
- **唯一输出/Owner**：单 worker 从创建到 SERVING 的阶段、资源账本与 readiness 证据。
- **相邻篇不得重述**：83 只消费 readiness/capacity；52 只拥有 graph capture 机制。
- **证据基线**：vLLM `bb363db9...` 的 `worker_base.py`、`gpu_worker.py`、CPU/XPU workers。

## 1. 阶段机

建议把 worker 明确建模为：`CREATED → DEVICE_READY → DISTRIBUTED_READY → MODEL_LOADED → PROFILED → CACHE_READY → ARTIFACT_READY → WARM → SERVING`。每一步输入和 side effect 不同；跳过阶段必须是 backend 的显式合同。

GPU 路径通常先绑定 device/环境和 process groups，再构造 runner/load weights；随后用代表性 dummy/profile run 测非 KV 峰值，engine 统一 KV config，worker 才分配真实 KV tensors，最后 compile/capture/warmup。

## 2. 为什么顺序重要

- profile 前未加载全部 weights 会高估 KV 容量；
- graph capture 后再改变 KV address 可能使 replay 引用旧地址；
- distributed groups 未就绪便 load sharded weights 会得到错误 shard；
- warmup shape 不覆盖真实 profile，会把首次编译延迟泄露给用户；
- backend runtime lazy allocation 若 profile 未触发，会在 serving 时 OOM。

## 3. memory profile 的可审计输入

需记录 device total/free、weights、runtime baseline、peak activation/workspace、graph/artifact memory、non-framework allocations、configured utilization、reserve 与最终 available KV bytes。多 worker 返回值应逐 rank保留，由 core 选择共同可用 config。

profile workload 必须覆盖最大 token/batch、multimodal encoder、MoE routing、speculative lookahead 等启用组合。只跑 decode=1 不能证明 max prefill 安全。

## 4. warmup 的多重目的

warmup 可能触发 kernel JIT、allocator pools、通信建链、collective autotune、graph/trace capture、weight packing 和 runtime executable load。它既影响 memory 也影响 readiness latency。第三方插件若把 compile 和 warmup 合并，应分别暴露 time/失败原因，否则难区分 build defect 与 device execution defect。

## 5. 固定源码与阶段证据

`vllm/v1/worker/worker_base.py::WorkerBase` 定义 worker contract；CUDA 主路径落在 `gpu_worker.py::Worker` 的 `init_device/load_model/determine_available_memory/initialize_from_config/compile_or_warm_up_model`，runner 在 `gpu_model_runner.py`。第三方 backend 替换 class 后仍必须保持这些返回值和调用顺序语义。

| 阶段 | 创建的持久状态 | 失败后必须撤销 |
|---|---|---|
| init device/group | context、communicator、streams | group/context、IPC epoch |
| load model | parameter shards、loader metadata | incomplete weights/buffers |
| profile | peak/reserve 估计 | profiling temporaries |
| initialize KV | physical pages/addresses | allocator leases |
| warmup/capture | kernels、graphs/artifacts | partial capture/cache entry |

`SOURCE_IMPLEMENTED`：上述 WorkerBase/GPU Worker 调用面；`INFERENCE`：readiness 是所有阶段原子成功；`RECOMMENDATION`：逐阶段记录 duration、memory delta、artifact key；`UNKNOWN`：vendor runtime 的真实 cleanup/completion。

## 6. readiness、失败与验收

只有所有 ranks 到达相同 epoch 的 WARM 状态，service 才 ready。任一 rank profile/compile 失败时，不应让 gateway继续发请求。重启需丢弃旧 cache/block table/artifact handles；sleep/wake 后也要声明哪些对象保留。

验收覆盖 cold/warm restart、低 memory、rank mismatch、首次真实 shape、并发启动、artifact cache corruption、device reset。**UNKNOWN**：无目标设备运行时，只能证明调用面和阶段存在，不能证明容量或 warmup 完整。
