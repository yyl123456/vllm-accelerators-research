# XLA Shape、Bucket、Compile Cache 与 Fallback

## 文档契约

- owner：JAX/XLA specialization key、预编译、persistent cache 与 miss 行为。
- 依赖输入：`50`–`55`、`131`、`133`。
- 唯一输出：TPU serving 的 compile-latency/correctness contract。
- 不讨论：runner request lifecycle 或 SPMD topology；由 `133`、`135` 持有。

## 为什么需要 bucket

在线请求的 batch、token count、context、KV pages、multimodal item 数和 speculative window 动态变化，而 XLA executable 对 shape、dtype、static args、sharding 和 compiler options specialization。无限精确 shape 会造成编译风暴；过度 padding 则浪费 MXU/HBM 和 goodput。

固定源码 `runner/compilation_manager.py::CompilationManager` 预编译可能输入组合；`capture_model()` 覆盖 backbone、vision、embedding merge、sampling、logprobs、structured/spec/continue-decode 等子图。`_run_compilation()` 可 lower/compile，也会在无法 AOT lower 时记录并回到 warmup-triggered inline compile。

## Cache key 的最低闭包

```text
function/subgraph identity
+ model/weights/quant + implementation type
+ input shapes/dtypes/static args
+ mesh + NamedSharding/PartitionSpec
+ KV layout/page size + feature mode
+ compiler options + jax/XLA/libtpu/target generation
```

只按 model name 或 batch bucket 缓存会错用 executable。persistent cache 还需要版本、目标和 artifact integrity；进程内 callable cache 则要处理 model unload 与 topology change。

## Miss/fallback 分类

| 情形 | 可接受行为 | 风险 |
|---|---|---|
| 已知 bucket | cache hit/replay | guard 缺维度导致错图 |
| 合法未预热 bucket | inline compile 或排队 compile | TTFT 尖峰/惊群 |
| 超出 profile | reject 或受控 padding | HBM OOM/极端浪费 |
| AOT lower 不支持 | warmup compile（源码已有分支） | readiness 假阳性 |
| op 无 lowering | fail closed 或明确的同语义 fallback | 静默 CPU/数值差异 |

`simple_compile_backend='eager'` 只绕过 `torch.compile`；不能把它解释为无 XLA compile fallback。JAX `jit` 仍产生 executable。

## Readiness 与编译并发

`CompilationManager` 可用 thread pool 提交多个 compile future，并在 `_flush_compilations()`/finalize 等待。readiness 必须定义是“关键 buckets 已编译”还是“允许首请求触发编译”；否则健康探针和用户 TTFT 目标相互矛盾。多 worker 同时 cold start 还会放大 compiler/host memory 压力。

## 观测指标

- compile key/hash 与 miss reason；
- lower、compile、load、first execute 分段时间；
- executable/cache bytes 与 eviction；
- request padding ratio；
- bucket hit ratio与 cold-start affected requests；
- fallback/reject count，按 feature 与 model path 分组。

## 结论分类

- `SOURCE_IMPLEMENTED`：预编译 manager、persistent JAX cache、future flush 和 warmup fallback。
- `INFERENCE`：bucket 策略是编译成本与有效计算的多目标优化。
- `RECOMMENDATION`：发布物携带 finite bucket manifest，并在 admission 前判定覆盖。
- `UNKNOWN`：固定 tuple 的 compile time、cache portability 和 cold-start SLO。

