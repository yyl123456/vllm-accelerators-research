# CUDA Graph、Piecewise Capture 与 Shape

## 文档契约

- **唯一问题**：如何将重复 execution 段 capture/replay，并安全容纳动态 batch shape。
- **In scope**：full/piecewise graph、capture buckets、static buffers、addresses、replay selection、memory。
- **Out of scope**：Dynamo partitions（50）、跨厂商 trace/AoT（54）、artifact disk cache（53）。
- **依赖输入**：50 的 partitions；42 的 buffers；28/94 的 completion。
- **唯一输出/Owner**：CUDA Graph capture/replay 的 address、bucket 与 invalidation 合同。
- **相邻篇不得重述**：54 只持有跨 execution-regime 分类；53 只持有持久 artifact identity。
- **证据基线**：vLLM `bb363db9...` 的 `compilation/cuda_graph.py`、`breakable_cudagraph.py`、piecewise backend 与 worker warmup。

## 1. graph 的收益与约束

CUDA Graph 将一串 launches 录制后 replay，降低 CPU launch overhead并稳定执行。代价是许多 buffer address、control flow 和 launch topology 必须稳定，且 capture 本身占 memory/time。

在线 ragged batch 通常映射到预捕获 size buckets，以 padding 填满静态 buffers；有效 token/sequence metadata 控制语义。bucket过密增加 capture/artifact memory，过疏增加 padding work。

## 2. piecewise capture

不适合 capture 的 custom op、collective或动态段形成 graph breaks；可分别 capture稳定 partitions并在 eager边界串联。此处的 piecewise 是 runtime capture粒度，不等于 50 中 compiler partition 的所有概念，二者要有明确映射。

## 3. 地址与生命周期

input/output/KV/workspace 若被 graph记住，cache reallocation、sleep/wake、model reload、layout change后必须 invalidate并 recapture。仅内容相同而地址变化也不可安全 replay。并发 replay 若共享 static buffer，需要序列化或多实例 buffer pool。

## 4. shape selection

runtime shape 应选择可容纳且语义兼容的最小 bucket；超过最大 shape应 eager/recompile/reject，策略需明确。PP/TP ranks必须选同一 collective-compatible bucket。spec `k`、MM encoder、LoRA等会扩大 key。

## 5. 固定源码 capture 对象

`vllm/config/compilation.py::CUDAGraphMode` 表达 NONE/PIECEWISE/FULL 等模式；`vllm/compilation/cuda_graph.py::CUDAGraphWrapper` 以 `BatchDescriptor` 管理 concrete entries；`VllmConfig` 会依据 connector、cascade attention 和 feature constraints 改写 mode。wrapper 注释明确其本身不拥有 persistent buffers，地址 owner 仍在 runner。

capture key 除 padded batch/num tokens 外还受 uniform decode、attention/feature mode 与平台实现影响。KV connector 可要求 piecewise graph，说明 graph 模式不是局部性能开关。

`SOURCE_IMPLEMENTED`：CUDAGraphMode/Wrapper/BatchDescriptor 与 config guards；`INFERENCE`：buffer lease 必须覆盖 replay completion；`RECOMMENDATION`：记录 capture/miss/invalidation reason；`UNKNOWN`：vendor graph API 的 address/completion 等价性。

## 6. 验收

比较 eager/capture输出；覆盖 bucket边界和 padding；检查 recapture/invalidation；测 capture memory、replay hit rate和 fallback；故障注入 device reset/late graph error。**结论**：graph replay 是地址和形状 specialization，不是透明加速开关。
