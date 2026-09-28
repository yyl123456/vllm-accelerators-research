# 容量规划、过载与 Autoscaling

## 文档契约

- **唯一问题**：如何从workload与SLO推导replica/accelerator数量，并在动态负载下维持安全余量。
- **In scope**：capacity envelope、queue signal、headroom、scale triggers、cold start、heterogeneous pools。
- **Out of scope**：scheduler局部budget（25）、benchmark采集（82）、routing具体决策（144）。
- **证据基线**：01组模型、vLLM runtime metrics及vendor待测系数。

capacity不是单个max concurrency，而是prompt/output/arrival/feature分布下满足TTFT/ITL/error目标的区域。内存约束和compute/communication约束共同决定；prefix hit、spec acceptance、MoE skew使每请求成本非平稳。

规划从trace分桶，按backend建立service-time与KV bytes模型，再用load test校准。安全余量覆盖p99长度、runtime/graph峰值、device差异、故障N+1和发布期间双版本。只按平均tokens/s配置会在长prompt波峰崩溃。

autoscaling signal应组合arrival/work backlog、predicted token work、queue delay、KV headroom和SLO burn。GPU/NPU utilization可能在memory-bound decode低算力占用时误导。scale-out要计模型下载/load/compile/warmup与cache冷启动；在新replica ready前不能算capacity。

过载时按明确policy reject/degrade，而不是无界queue。heterogeneous pool需按capability和实测service rate分配，不以device count等权。验证使用阶跃/尖峰/长尾、单replica故障、冷启动失败和rollout，观察oscillation、reject、公平性与goodput。

## 控制回路

```text
observe work/KV/SLO burn -> predict deficit -> scale/reject/degrade
 -> wait load/compile readiness -> route canary -> measure with cooldown
```

scale target 应是 token-work/service-time envelope，而非请求数。replica 只有达到 device/model/compile readiness 才计入 capacity。

`INFERENCE`：autoscaler 是延迟反馈系统，短窗口会振荡；`RECOMMENDATION`：分设 memory emergency、queue/SLO、forecast triggers；`UNKNOWN`：backend 冷启动/服务率需 82 校准。
