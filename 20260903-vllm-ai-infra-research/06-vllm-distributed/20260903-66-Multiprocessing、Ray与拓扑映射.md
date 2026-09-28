# Multiprocessing、Ray 与拓扑映射

## 文档契约

- **唯一问题**：逻辑 ranks 如何由本机进程或集群 actors 启动、放置、通信和回收。
- **In scope**：uni/mp/Ray/external launcher、IPC、placement、env/device binding、lifecycle。
- **Out of scope**：Executor API（40）、rank groups（60）、平台device ID（73）。
- **依赖输入**：73 的 physical topology；40 的 Executor selection；60 的 rank table。
- **唯一输出/Owner**：uni/mp/Ray/external launcher 的进程创建、placement 与 cleanup合同。
- **相邻篇不得重述**：20 只画运行时拓扑；73 只定义device mapping。
- **证据基线**：vLLM `bb363db9...` V1 executors、multiproc/ray utils。

uniproc把worker同进程化，调试简单但隔离弱；multiproc在单机或显式环境中spawn workers，依赖IPC/queues/shared memory；Ray以actors和placement groups跨节点管理资源；external launcher让外部系统提供process world。

launcher必须传递同一有效config、master address/ports、rank/world、local device、network interfaces和plugin环境。fork/spawn差异会影响已初始化device runtime；通常应避免在父进程初始化accelerator后fork。

placement需认识NUMA、PCIe root、GPU/NPU互联、NIC affinity，而非只满足device count。TP/EP高通信group应优先放高速域；PP/DP可跨较慢link，仍需按实测决定。

IPC control path应有backpressure、heartbeats、epoch和message size限制；worker exit必须唤醒等待Future。shutdown顺序是停admission、drain/cancel、停止collectives、释放device、关闭IPC/actors。

验收包含rank table与topology probe、单/多节点启动、端口冲突、worker迟到/崩溃、head node失败、环境漂移和cleanup无孤儿进程。Ray可调度成功不等于硬件topology最优。

## 固定源码执行器

- `vllm/v1/executor/uniproc_executor.py`：同一进程内 worker。
- `vllm/v1/executor/multiproc_executor.py::MultiprocExecutor`：local worker processes、IPC 与 worker monitor。
- `vllm/v1/executor/ray_executor.py::RayDistributedExecutor`：Ray actors、placement 与 remote calls。
- platform 可在 `check_and_update_config()` 替换 executor class；因此 launcher capability 是 backend tuple 的一部分。

## 启动状态机

```text
resolve effective world/topology
 -> reserve placement/resources
 -> spawn actors/processes with immutable epoch
 -> bind local rank/device/env
 -> init runtime and process groups
 -> load/profile/cache/warmup
 -> all workers ready -> engine ready
```

任何子步骤失败都应撤销该 epoch 的 reservation、IPC endpoints、actors/processes、communicators 和 device handles。只杀主进程会留下共享内存、端口或 device context，造成下一次启动的非确定错误。

## 配置与拓扑检查

| 检查 | 原因 |
|---|---|
| `TP×PP×PCP×local DP` 对应 local workers | 防止 rank 空洞或重复 |
| Ray resource/placement bundle 对应 physical device | 资源数量不代表互联邻接 |
| env/config digest 各 worker 相同 | plugin/runtime 差异可导致不同 control flow |
| master/control/data endpoints 唯一 | 端口冲突或串接旧 epoch |
| process start method 在 device init 前确定 | fork 已初始化 runtime 风险 |

## Failure propagation

worker monitor/actor future 必须把 exit、exception 和 timeout传播到 executor，再由 EngineCore 停止提交并完成/失败 pending futures。collective hang 可能没有 Python exception，需要 watchdog/health signal；错误恢复的最小安全范围通常是整个相关 process group，而不是单 worker。

## 证据分类

- `SOURCE_IMPLEMENTED`：三个执行器与 platform replacement contract。
- `INFERENCE`：startup/cleanup 必须按 epoch 原子化。
- `RECOMMENDATION`：保存 placement plan、env digest、rank table和每阶段 readiness。
- `UNKNOWN`：Ray/MP 在各 vendor runtime 的真实孤儿清理和 collective hang 收敛。
