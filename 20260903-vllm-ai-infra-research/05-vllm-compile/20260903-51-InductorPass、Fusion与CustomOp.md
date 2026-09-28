# Inductor Pass、Fusion 与 CustomOp

## 文档契约

- **唯一问题**：vLLM 如何在 captured graph 上应用可验证的重写与 fusion。
- **In scope**：pass manager、pattern/replacement、pass UUID、range applicability、collective/quant/RoPE/RMS fusions。
- **Out of scope**：op runtime fallback（45）、图捕获（50）、artifact key总论（53）。
- **依赖输入**：50 的 FX partitions；45 的 runtime op ABI。
- **唯一输出/Owner**：Inductor rewrite、pass identity/order 与 fusion 等价性。
- **相邻篇不得重述**：45 唯一拥有 dispatch/binary/fallback；本篇不重新定义 op runtime。
- **证据基线**：vLLM `bb363db9...` 的 `compilation/passes/`。

## 1. pass 是语义变换

vLLM 在 Inductor pre/post-grad pipeline 插入定制 passes，包括 cleanup、functionalization 和 pattern fusions。典型 fusion 将 QK norm/RoPE/KV write、activation quant、RMSNorm、collective 等序列替换为专用 op。收益来自减少中间 memory、launch 与 collective，但 replacement 必须对完整 predicate 语义等价。

## 2. matcher 风险

pattern 需约束 op overload、dtype、shape、stride、constants、alias/mutation、backend capability。只匹配拓扑可能在不同 RoPE variant、quant scale 或 inplace语义下误替换。no match 是性能缺失；false-positive match 是 correctness bug，风险等级不同。

## 3. pass identity

pass source/config/order改变生成代码，必须进入 compile hash。`InductorPass.uuid`/source hash 等机制用于区分；自定义插件 pass 也需稳定、内容寻址的 identity。只以 pass class name 作为 key 会复用旧 artifact。

pass order也有语义：早期 canonicalization决定后续 matcher 是否命中；两个 fusion可能争用相同 nodes。应输出每 pass 前后 graph digest、match count、耗时和拒绝原因。

## 4. collective fusion

把 collective 与计算 fusion 会改变 stream、buffer lifetime 和 rank同步点。数值 reference 之外还要验证所有 ranks 参与同一 collective sequence；条件分支导致某 rank跳过会 hang。

## 5. 固定源码 pass pipeline

`vllm/compilation/passes/inductor_pass.py::InductorPass` 要求可形成 unique identifier；`vllm_inductor_pass.py::VllmInductorPass` 与 `passes/pass_manager.py` 组织 post-grad passes，并将 pass state 纳入 Inductor cache。这里唯一持有 FX pattern/rewrite/fusion identity，不定义 runtime op dispatch（见 45）。

pass 验证应比较 rewrite 前后 graph semantics、alias/mutation、dtype/shape、collective ordering，并用 near-miss pattern 证明 matcher 不误触发。`SOURCE_IMPLEMENTED`：上述 pass classes；`INFERENCE`：pass source/config hash 是 artifact key 的组成；`RECOMMENDATION`：每次命中记录 pattern/pass UUID；`UNKNOWN`：不同 torch/Inductor revision 下 matcher 稳定性。

## 6. 验收

每个 pattern 有 positive/near-miss/negative graph；replacement 与 unfused输出/side effects对比；compile modes和多 rank覆盖；记录 kernel/graph变化与性能。禁用 pass 后可作为诊断 fallback，但生产 silently disable 应进入 metrics。
