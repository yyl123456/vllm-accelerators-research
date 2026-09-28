# ParallelState、Rank 与 ProcessGroup

## 文档契约

- **唯一问题**：多维并行 ranks 如何形成互不混淆的通信 groups 和全局坐标系。
- **In scope**：world/rank、TP/PP/DP/EP/DCP/PCP groups、初始化/销毁、group ownership。
- **Out of scope**：各并行算法（61-65）、launcher实现（66）。
- **依赖输入**：73 的 physical device mapping；40 的 worker集合。
- **唯一输出/Owner**：全局/局部/各并行维 rank 坐标与 process-group epoch。
- **相邻篇不得重述**：61-65 仅消费 group，不重定义 rank namespace。
- **证据基线**：vLLM `bb363db9...` 的 `distributed/parallel_state.py`、utils与communicators。

一个 worker 不是只有 global rank；它在多个正交/重叠 group 中有 local rank。初始化必须从同一 parallel config和rank mapping构造 groups，并保证所有进程以兼容顺序创建，否则 collective hang。world-size乘积/布局关系需显式校验，不能依赖默认整数除法。

group对象持有 communicator、device group、CPU control group、streams/buffers等资源。它们属于 process epoch；elastic reconfigure或restart后旧 handle不能复用。destroy需要与 in-flight collective隔离。

核心不变量：同一 collective的participants、sequence、dtype/count一致；每个 model shard只使用正确group；global↔local rank可逆；driver身份在各维不冲突；device ID与rank不是同义词。

验证应输出完整 rank table：host/process/device、global rank、各维坐标、group members、transport/topology。做 all-reduce/broadcast/all-to-all smoke、错误world size、重复device、restart epoch和不同launcher对照。**结论**：ParallelState 是分布式执行的地址空间，不只是若干全局变量。

## 固定源码调用链

`vllm/config/parallel.py::ParallelConfig` 先持有 TP、PP、DP、PCP、DCP、EP/EPLB 等尺寸与 guard；`vllm/distributed/parallel_state.py::init_distributed_environment()` 建立默认 world，随后 `initialize_model_parallel()` 按 rank tensor 生成各维 group，`GroupCoordinator` 包装 device/CPU group 与 communicator。`ensure_model_parallel_initialized()` 是重复初始化的一致性门禁，而非新建另一套 namespace。

```text
ParallelConfig validation
 -> init_distributed_environment(world/rank/local_rank/backend)
 -> initialize_model_parallel(config)
 -> rank tensor reshape/transpose
 -> GroupCoordinator per axis
 -> layer/runner obtains axis-specific group
```

## 当前 revision 配置/guard 表

| 约束 | 固定源码位置 | 失败行为 |
|---|---|---|
| local DP size 不得大于 global DP | `ParallelConfig.__post_init__` | `ValueError` |
| external LB 需要 DP>1 | 同上 | `ValueError` |
| PCP/DP、PCP/DCP 有组合约束 | `parallel.py` PCP/DCP guards | `ValueError` |
| elastic EP 需要 EPLB，且限制 PP | `parallel.py` elastic guards | `ValueError` |
| communicator backend 受 platform/all2all 约束 | config + device communicator | 改写、fallback 或错误 |

## 生命周期与失败收敛

group 创建是 collective protocol：不同 rank 若以不同顺序或不同 members 创建，会在初始化或首个 collective hang。destroy 前必须停止新请求、drain/cancel execution、等待 in-flight collectives，再销毁 model groups/default group。worker 重启若 global membership 不变但 process epoch 已变，旧 communicator、CUDA/accelerator stream、shared buffer 均不可复用。

## 证据分类

- `SOURCE_IMPLEMENTED`：上述类/函数与 guard 均来自 `bb363db9a5ec2edc7b39e99b00af363a89d1fb81`。
- `INFERENCE`：将 group epoch 绑定 worker/engine epoch 是故障安全要求。
- `RECOMMENDATION`：启动 artifact 输出每轴完整 rank table 和 group UUID。
- `UNKNOWN`：第三方 communicator 在 rank fault 后能否可靠 unblock，需设备/多进程故障测试。
