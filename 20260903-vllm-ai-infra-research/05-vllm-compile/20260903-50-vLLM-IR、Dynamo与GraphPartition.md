# vLLM IR、Dynamo 与 Graph Partition

## 文档契约

- **唯一问题**：动态 Python model forward 如何被捕获为可优化子图，同时保留 vLLM stateful 边界。
- **In scope**：Dynamo/FX capture、dynamic dimensions、graph breaks、partition rules、piecewise execution。
- **Out of scope**：fusion（51）、CUDA Graph（52）、artifact cache（53）。
- **依赖输入**：43-45 的 model/op ABI；42 的 representative inputs。
- **唯一输出/Owner**：FX graph、guards、graph breaks 与 compiler partition 边界。
- **相邻篇不得重述**：51 只消费 graph 做 rewrite；52 只 capture 已形成的执行片段。
- **证据基线**：vLLM `bb363db9...` 的 `compilation/decorators.py`、`backends.py`、`piecewise_backend.py`、`partition_rules.py` 与 compile design docs。

## 1. 捕获边界

模型 class 通过 compile-support decorator 将 forward 接入 Torch Dynamo。Dynamo 观察 Python bytecode 和 example inputs，产生 FX graph、guards 与 graph breaks；vLLM backend 再将图按不可合并/stateful ops 分割，编译可优化 partitions，并在运行时编排 eager/custom-op 片段。

不能把 FX graph 当完整 Engine graph：scheduler、block allocation、sampling、connector、通信或 attention custom op 可能位于图外。图边界必须保存 tensor alias、mutation、KV side effects 与 stream order。

## 2. dynamic shape

在线 batch 的 token 数、sequence 数、MM shape、PP shard 都会变化。标记 dynamic dimension 可减少重编译，但扩大 guard/kernel 泛化成本；bucket specialization 增加 artifacts 和 warmup。架构目标是稳定形状变量集合，而非盲目“全部 dynamic”。

每个 guard miss 必须可观测：新编译、fallback 或 reject 是三种不同结果。静态 backend 不能假装支持动态 guard，只能用 profile/bucket admission。

## 3. partition 正确性

split point 应围绕有 side effect、opaque custom op、unsupported control flow 或 capture boundary。跨 partition tensors 的 lifetime、stride、dtype和 device 必须显式；分区数过多会增加 launch/Python 开销，过少则 compile failure/fusion受限。

KV write、collective 等 side effect 不能被 compiler 重排越过依赖。若 functionalization 复制或消除 alias，需要验证 cache state仍写到真实 backing allocation。

## 4. 固定源码捕获链

`vllm/compilation/decorators.py` 与 config 决定模型 compile wrapper；`vllm/compilation/backends.py`、`compiler_interface.py` 连接 Dynamo/FX 与 compiler adapter；`vllm/ir/op.py` 为 vLLM IR/custom op 提供图内语义与 cache identity。graph breaks 和 split points 必须保持 KV mutation、collective 与 request state 的顺序。

```text
model forward -> Dynamo guards/FX graph
 -> vLLM graph partition/split ops
 -> compiler adapter per graph/shape
 -> eager regions + compiled regions按原序组合
```

`SOURCE_IMPLEMENTED`：上述 compilation/IR modules；`INFERENCE`：partition boundary 是 side-effect ordering ABI；`RECOMMENDATION`：保存 FX/guard/split dump 与 eager island 原因；`UNKNOWN`：第三方前端的等价捕获范围。

## 5. 验收

保存 capture graph/guards/partition map；覆盖 representative dynamic shapes；逐 partition 与 eager reference 比较；检查 graph-break count、recompile count 和 eager islands；用 mutation/alias tests 验证 KV/custom op。**UNKNOWN**：一个模型标注支持 compile 不等于所有功能组合均被捕获。
