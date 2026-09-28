# Data Parallel、Replica 与路由

## 文档契约

- **唯一问题**：完整模型 replicas 如何接收请求、协调负载并维持会话/cache affinity。
- **In scope**：DP coordinator、replica capacity、global/local routing、wave、failure/rebalance。
- **Out of scope**：单engine admission（13/25）、EP（64）、P/D routing（36）。
- **依赖输入**：13 的全局 admission；60 的 DP group；83 的 capacity envelope。
- **唯一输出/Owner**：replica identity、routing/failover state 与 global-local queue boundary。
- **相邻篇不得重述**：144 只使用本篇能力/负载信号做异构选择。
- **证据基线**：vLLM `bb363db9...` DP coordinator/parallel state与serving配置。

inference DP复制完整逻辑模型，每请求通常由一个replica服务。它提升aggregate throughput和failure isolation，但不降低单请求model memory。replica内部仍可TP/PP/EP。

route key不能只看queue length：需估算prompt/decode work、KV headroom、adapter residency、prefix locality、MM/feature capability、P/D role和SLO。多轮会话粘性提高cache命中，却可能造成热点；迁移则需重算或转移KV。

global admission决定是否接受流量，replica scheduler决定本地step；两级queue必须有明确owner与期限。否则gateway认为已分配capacity，而replica无限等待。

replica failure需要停止routing、处理in-flight结果和幂等retry。DP reconfigure应使用epoch，不能把旧wave output归入新membership。验证以skew workload、单replica降速/崩溃、adapter locality和cache affinity评估goodput/p99，而不是只测均匀round-robin。

## 固定源码落点

`ParallelConfig` 持有 `data_parallel_size/local/rank/start_rank`、external LB 与 hybrid LB 设置；`vllm/v1/worker/dp_utils.py` 负责 DP worker coordination；EngineCore/serving 侧的 DP coordinator 汇总 load/health。`parallel_state.initialize_model_parallel()` 生成 DP groups；MoE 下 DP 还可能参与 EP world。

## 路由状态协议

| 状态 | owner | 必须携带 |
|---|---|---|
| advertised capacity | replica/health reporter | epoch、queue、KV headroom、capabilities |
| route decision | gateway/coordinator | request ID、deadline、target replica epoch |
| accepted | replica scheduler | local queue lease |
| executing | worker/runner | row/KV/device epoch |
| finished/failed | output/health plane | finish reason、retry safety |

gateway 仅在 replica 明确接受后才能把 route 视为生效；否则 coordinator crash 会造成请求丢失或双发。生成请求通常不是天然幂等：随机 sampling、streamed prefix 和外部 side effects 使 failover 需要新 request attempt/epoch 与去重策略。

## 当前 guards 与模式

- `data_parallel_size_local <= data_parallel_size`。
- external LB 只有 DP>1 合法；internal/hybrid 模式决定谁拥有 global queue。
- offline/online launcher 对 local rank/start rank 的来源不同。
- attention DP、MoE EP 与普通 replica DP 可能复用 `data_parallel_size`，语义必须从 effective sharding/config 判定。

## 证据分类

- `SOURCE_IMPLEMENTED`：DP config、group、worker coordination 与 serving coordinator 入口。
- `INFERENCE`：route accept 是 global/local queue 的线性化点。
- `RECOMMENDATION`：capacity advertisement 带 TTL 与 replica epoch，路由记录 desired/effective target。
- `UNKNOWN`：不同 deployment launcher 的真实 failure detection/retry 时延。
