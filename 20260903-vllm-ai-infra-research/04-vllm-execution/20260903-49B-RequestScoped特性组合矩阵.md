# Request-Scoped 特性组合矩阵

## 文档契约

- **唯一问题/Owner**：多个已分别支持的 request-scoped features 组合时，谁判定可用并如何证明。
- **依赖输入**：47 spec、48 MM、49 LoRA、49A grammar、15 quant、33 prefix cache。
- **唯一输出**：capability predicate 与组合验证方法；不拥有任何单项机制。
- **相邻篇不得重述**：只记录交叉约束和证据状态，单项原理链接到其 owner。
- **证据基线**：vLLM `bb363db9...` 配置 validation/runner 调用面及各 vendor 固定 revision 矩阵。

## 1. 为什么不能乘法推导

`supports(A) ∧ supports(B)` 不推出 `supports(A×B)`：二者可能争用 graph key、KV layout、runner metadata、sampling 位置、device memory 或 static profiles。组合能力应为 predicate：`C(model, backend, runtime, parallel, features, shapes)`。

## 2. 高风险交叉点

| 组合 | 共享状态/冲突 | 需要的联合证据 |
|---|---|---|
| LoRA×prefix/P-D | KV identity | 不同 adapter 必 miss，transfer descriptor 匹配 |
| grammar×spec | token commit/cursor | partial accept/reject 逐步等价 |
| MM×graph/AoT | dynamic encoder shapes | profile 覆盖和 fallback |
| quant×LoRA | packed weights/kernel | fused/非 fused 数值与 dispatch |
| spec×KV quant | temporary KV/rollback | rejected state 无污染 |
| EP×LoRA | expert adapter placement | all ranks/layers 覆盖 |
| P/D×不同 TP | KV shard mapping | gather/scatter 数值与 lease |

## 3. 判定层

静态不兼容由 config/platform validation 拒绝；依赖 model/shape 的由 startup/profile gate；依赖当前 resident 资源的由 admission；运行时意外只作为 failure，不应是正常 capability discovery。拒绝必须返回具体 constraint，禁止静默关闭 feature。

## 4. 覆盖与证据

先做所有声明 pairwise，再按共享状态选三元/四元高风险组合；生产 trace 补真实频率。矩阵单元记录事实来源类型与验证成熟度的二元组（字典见 02）和完整 tuple；未支持单独写 `UNSUPPORTED`，不得把它当证据等级。

- `SOURCE_IMPLEMENTED`：固定 revision 存在多项 request-scoped 字段和 validation 路径。
- `INFERENCE`：组合 predicate 是架构归纳，不代表当前有统一 registry。
- `RECOMMENDATION`：vendor release 以机器可读矩阵 fail closed。
- `UNKNOWN`：未显式测试的单元不能由邻格外推。
