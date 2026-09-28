# Compile Failure、Fallback 与可观测性

## 文档契约

- **唯一问题**：编译/加载/replay失败如何分类、对用户可见并安全降级。
- **In scope**：failure taxonomy、fallback policy、metrics/logs/artifacts、retry/circuit breaker。
- **Out of scope**：通用生产监控（81）、op fallback（45）、artifact identity（53）。
- **依赖输入**：50-54 的各阶段输出；84 的 failure owner模型。
- **唯一输出/Owner**：compile-specific failure taxonomy、fallback policy 与诊断字段。
- **相邻篇不得重述**：81 只聚合这些指标；45 只处理单 op runtime fallback。
- **证据基线**：vLLM `bb363db9...` compile monitor/counter/debug docs与 compiler interfaces。

## 1. 失败阶段

区分 capture/graph-break explosion、pass错误、codegen/toolchain、binary link/load、guard miss/recompile、runtime launch、async device error、numeric divergence。把它们统一记录为“compile failed”无法选择恢复策略。

## 2. fallback policy

可选 fail startup、request reject、eager fallback、较小profile、disable某 pass、restart worker。策略必须声明 correctness 等价、SLO成本、适用scope和持续时间。numeric mismatch不得自动 fallback 后静默继续；它应隔离 artifact并触发高等级告警。

同一 shape反复编译失败需 negative cache/circuit breaker，避免每请求重试放大过载。fallback capacity可能远低于 compiled path，gateway admission要同步降额。

## 3. 可观测字段

记录 semantic/binary key、model/backend/device tuple、stage、graph/partition、shape/guards、pass matches、compiler stderr摘要、耗时、cache hit、fallback target和request impact。源码/IR dump可能含路径或模型信息，应受权限和retention控制。

## 4. 异步错误

launch返回后出现的device error必须归因到step/artifact，而非下一次随机synchronize。worker应停止复用可能被部分写入的KV/output buffers；executor决定整rank group fail-stop。

## 5. 固定源码错误落点与状态机

`vllm/compilation/compiler_interface.py` 在 lower/compile/cache lookup/load wrapper 各阶段产生错误；`vllm/config/vllm.py` 在初始化时依据平台/features 改写 compilation/CUDAGraph mode；`vllm/compilation/cuda_graph.py` 持有 capture/replay entries 与 stats。错误必须带 stage 与 key，不能压成 generic “compile failed”。

```text
requested mode -> config guard/effective mode
 -> cache lookup -> lower -> compile -> load/capture -> execute
 -> success publish；任一步失败 -> invalidate partial -> policy decision
```

fallback policy需要声明是否语义等价、是否改变采样/RNG、是否允许在 production、是否有 retry budget。异步编译 future 异常必须在 readiness 或请求边界被消费。

`SOURCE_IMPLEMENTED`：compiler/config/graph error surfaces；`RECOMMENDATION`：指标按 stage/key/fallback reason 聚合；`UNKNOWN`：vendor compiler 的 crash/timeout cleanup。

## 6. 验收

注入每阶段失败；验证只执行允许fallback、metrics准确、无无限retry、资源被quarantine；比较 fallback质量与capacity；重启后坏artifact不再加载。**结论**：可恢复性是compile架构的一部分，不是日志补丁。
