# CustomOp、Kernel Dispatch 与 Fallback

## 文档契约

- **唯一问题**：模型算子如何在多个实现中选择目标 kernel，并在缺失时保持可解释的正确性边界。
- **In scope**：CustomOp registration、native/platform implementation、dispatch、fallback、capability gaps。
- **Out of scope**：attention backend（44）、compiler fusion（51）、单一 vendor kernel 清单。
- **依赖输入**：71 的 platform dispatch；43 的 model layer；54 的 execution mode。
- **唯一输出/Owner**：runtime custom-op ABI、dispatch key、binary availability 与语义 fallback。
- **相邻篇不得重述**：51 仅解释 FX/Inductor 如何识别/重写该 op，不拥有 runtime dispatch。
- **证据基线**：vLLM `bb363db9...` 的 `model_executor/custom_op.py`、`layers/`、`_custom_ops.py`、platform dispatch。

## 1. 三层视图

模型 layer 表达语义 operation；CustomOp 选择 native 或 platform-specialized implementation；底层 extension/Triton/runtime 实现 kernel。把三层混合会让模型代码到处判断 device，也让插件升级时难以确定缺口。

## 2. dispatch key

正确 key 往往包含 platform、dtype、shape/alignment、quant scheme、architecture capability、compile mode 和 feature flags。仅按 `device.type` 选择会让 unsupported shape 落入 kernel。dispatch 应返回明确的 selected implementation 和拒绝理由。

## 3. fallback 的两种意义

语义 fallback 是较慢但数值等价、测试过的实现；应急 workaround 可能只覆盖窄 shape/dtype。两者不可同称“fallback”。没有 native implementation 时应 fail closed；不能返回近似或 CPU copy 而不记录。

fallback 还可能破坏 graph capture、distributed sharding、quant format 或 memory budget，即便单算子数值正确。因此 capability 必须是组合 predicate。

## 4. extension 边界

C++/CUDA/other runtime op 要约束 tensor device、dtype、contiguity/stride、alignment、lifetime、stream 与 error reporting。异步 kernel launch 成功不等于执行成功；backend 要在统一 completion fence 前传播 device error。

## 5. 验证矩阵

逐 op 以 reference 覆盖 edge shapes/dtypes、非连续 tensor、空/尾块、quant extremes；验证 dispatch 正/负路径；在 eager/compile/graph 三模式比较；采集真正 kernel 名与 fallback rate。E2E “能生成文本”会掩盖大量未走到的 kernels。

## 6. 固定源码 ABI 与可达性

`vllm/custom_op.py::CustomOp` 持有注册、forward/native 实现与平台选择；`vllm/_custom_ops.py` 暴露编译 extension ops；`vllm/platforms/interface.py` 提供平台 custom-op/Inductor hooks。本篇只持有 runtime op schema、dispatch key、binary availability 与 eager fallback；FX rewrite 唯一属于 51。

| 层 | 需要匹配 | 错误类别 |
|---|---|---|
| Python schema | args、dtype、shape、alias/mutation | tracing/runtime contract |
| dispatch | CPU/CUDA/PrivateUse/XLA key | 错设备或 missing kernel |
| binary | arch、ABI、runtime symbols | import/load/launch failure |
| fallback | 数值、side effect、async semantics | 静默差异 |

`SOURCE_IMPLEMENTED`：CustomOp/_custom_ops/platform hooks；`RECOMMENDATION`：negative test 证明 unsupported device fail closed；`UNKNOWN`：第三方二进制组合的 ABI 覆盖。

## 7. 架构决策

插件应优先实现稳定 op contract 或注册扩展，而不是 fork 每个模型。若硬件图编译器负责 fusion，则单 op fallback 可能切断图，应在 54/55 的 artifact/failure 合同中明确。**UNKNOWN**：源码含某实现不等于 binary 已构建或设备支持。
