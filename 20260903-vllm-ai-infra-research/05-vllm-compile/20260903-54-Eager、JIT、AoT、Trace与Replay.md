# Eager、JIT、AoT、Trace 与 Replay

## 文档契约

- **唯一问题**：不同 backend execution regime 如何映射 vLLM 的动态请求状态。
- **In scope**：模式定义、shape/profile、state ownership、startup/runtime tradeoff、统一适配要求。
- **Out of scope**：CUDA Graph实现（52）、artifact key（53）、各 vendor具体实现。
- **依赖输入**：42 的动态 batch；50-53 的 compile/capture/artifact contracts。
- **唯一输出/Owner**：跨 backend execution regime 的统一分类和选择维度。
- **相邻篇不得重述**：vendor 文档只把自身实现映射到这些分类。
- **证据基线**：vLLM `bb363db9...` compile interfaces；第三方插件固定 revisions 后续专题。

## 1. 不同模式

- **Eager**：逐 op即时 dispatch，动态性高、诊断直接，launch/框架开销大。
- **JIT**：运行中按首次见到的 graph/shape编译；warmup后快，但首请求和recompile产生tail。
- **AoT**：部署前生成目标 executable/profiles；启动可预测，模型/shape/feature集合必须预先封闭。
- **Trace/replay**：录制具体运行序列和buffers后重复；launch低，但地址、control flow和shape更受限。

这些不是线性“更高级”。一个 backend常混合：AoT transformer graph + eager scheduler/sampling + runtime DMA + trace decode loop。

## 2. vLLM 动态性接口

必须表达 active sequences、tokens per step、block tables/slot mapping、positions、adapter/grammar、MM inputs、spec length、parallel collectives。Eager可直接 materialize；静态模式要通过 profiles、padding、indirect tables或host loop承载。

## 3. state ownership

如果 runtime内部管理 KV和batch，vLLM core的block ownership可能失真；若 vLLM外部管理，executable需接受动态tables。架构必须选择一个真源并定义同步，不能双方都暗中allocate。

## 4. 决策维度

比较 startup time、steady throughput、ITL tail、shape coverage、memory/artifact footprint、fallback、debuggability、model cadence和release coupling。AoT在固定 appliance可能合适，在长尾模型/LoRA多租户环境可能产生artifact爆炸。

## 5. 模式语义对照

| 模式 | specialization 时机 | state/address 约束 | miss 行为 |
|---|---|---|---|
| eager | 每次 runtime dispatch | 最弱，仍有 async buffer lease | 直接执行/unsupported op error |
| JIT | 首次 shape/guard | executable 与 guards | compile或reject |
| AoT | 发布/部署前 | target/profile/artifact manifest | reject/显式 fallback |
| trace/replay | warmup/capture | command、shape、地址、runtime state | recapture/eager/reject |

固定 vLLM 的 Inductor/CUDAGraph 是其中两种具体实现；QAIC QPC、TT trace、TPU JAX/XLA 是不同 backend 映射，不能把 CUDA completion/地址假设移植过去。

`SOURCE_IMPLEMENTED`：vLLM compilation config/compiler/CUDAGraph modes；`INFERENCE`：四模式统一于 specialization key+state lease；`RECOMMENDATION`：mode fallback 进入 API/metrics；`UNKNOWN`：不同 backend fallback 的数值和延迟等价性。

## 6. 验收

同一 workload分别测 cold/warm；列出支持profiles和fallback；超profile输入明确行为；验证 state迁移、abort/preempt、P/D、LoRA/spec组合。不得把 trace replay的固定 demo称为通用 serving support。
