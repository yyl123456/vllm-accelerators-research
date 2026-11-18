# Qwen3.5 Attention：MAX 算子 lowering、融合与硬件分派

## 1. 调研边界与版本

- 调研日期：2026-11-18。
- 仓库：`modular/third_party/modular`，上游 `https://github.com/modular/modular.git`。
- 更新前：`da0db31d422e19ad5952d8d106b6da1d6899cf0a`（2026-10-23）。
- 更新后：`bbc13a719b6c139ede763f1434a4342ebfd843ac`（2026-11-17），对应 Mojo `1.3.0.dev2026111705`、MAX `26.8.0.dev2026111705`。执行 fetch 后 fast-forward，共前进 2,044 commits；再次用 `ls-remote` 确认上游 main 相同，源码工作树干净。这是源码更新，未安装或升级系统 MAX/Mojo 二进制。
- 目标：文本 decoder 的 full attention 与 Gated DeltaNet；具体 checkpoint、设备、batch/sequence geometry 为 `UNKNOWN`。不展开 vision encoder、MoE、量化 GEMM 内部或投机解码完整协议。
- 权威对应编号：[50：IR 与图分区](../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-50-vLLM-IR、Dynamo与GraphPartition.md)、[51：Fusion 与 CustomOp](../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-51-InductorPass、Fusion与CustomOp.md)、[54：执行模式分类](../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-54-Eager、JIT、AoT、Trace与Replay.md)。105 篇中没有 Modular 专属 owner；本篇为固定 SHA 的补充源码审查记录，不扩展 Manifest，也不改变 vLLM 固定基线。
- 证据等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`；commit message 与设计文档用于解释意图，未进行模型编译、设备数值或性能实测。
- 已读取上游根、`GraphCompiler/`、`max/`、`max/kernels/` 的 AGENTS.md；源码只读。

本文下述源码路径均相对该 Modular 仓库，行号固定于更新后的 SHA。

## 2. 核心结论与路线侧重

MAX 的当前路径是：**模型显式表达 attention 大算子，图编译器进行通用 lowering/结构化融合，Mojo kernel 根据目标和运行时几何分派**。不能把这一过程统称为“编译器把整个 self-attention 自动融合成一个 kernel”。

- **图编译 → MEF → runtime 为 Serving 主线，侧重 ★★★★★**。`Qwen3_5Model.load_model` 进入 `compile_and_init_graphs`，后者 trace Graph/GraphModule，调用 `session.compile_reusing_mefs`，再 `session.init_all`。kernel 编译管线仍有 `JitCompileKernels`，所以“图编译路线”不意味着所有 kernel 必须在模型部署前生成。
- **设备图 capture/replay 是该主线的补充，侧重 ★★★★☆**。Qwen3.5 注册 `supports_device_graph_capture=True`；不能将其视为与图编译互斥的第二套模型实现。
- **独立逐算子 eager 为辅助，侧重 ★★☆☆☆**。不从 custom op 的存在推断模型走 eager；此处 custom op 在模型图内部编译。

以上星级表达代码路径侧重，不是实测性能或生产成熟度评分。证据：`max/python/max/pipelines/architectures/qwen3_5/model.py:L485-L539`、`max/python/max/pipelines/lib/graph_specs.py:L214-L296`、`max/python/max/pipelines/architectures/qwen3_5/arch.py:L56-L63`、`GraphCompiler/lib/GraphCompiler.cpp:L613-L625`。

### 最新版本与旧版的重要差异

提交 [`d8931e10e5f9f3816b91279ddd5262f6a8b22784`](https://github.com/modular/modular/commit/d8931e10e5f9f3816b91279ddd5262f6a8b22784)（2026-11-05 合入）删除了 `MultiHeadAttentionPattern` 及其选项。旧版曾匹配 BMM → scale/mask → softmax → BMM；新版将相关模型改为直接发出 attention op。

Qwen3.5 的 paged attention 本身已经显式调用 `flash_attention_ragged`，因此该删除不代表 Qwen3.5 失去 FlashAttention，也不能将这个提交中无缓存 attention wrapper 的 head-dim 限制套用到 Qwen3.5 paged 路径。更新后的实际符号是 `attention.mha.ragged.paged`，旧名 `mo.mha.ragged.paged` 已不适合作为当前检索入口。

## 3. 先区分两种 attention

`Qwen3_5Config._get_layer_types` 优先读取 checkpoint 的 `layer_types`；缺省才按 `full_attention_interval=4` 生成每四层一层 full attention、其余 linear attention。不能将“48/64 层”泛化为全部 Qwen3.5 尺寸。

证据：`max/python/max/pipelines/architectures/qwen3_5/model_config.py:L569-L585`；`max/python/max/pipelines/architectures/qwen3_5/qwen3_5.py:L446-L479` 按 layer type 实例化两种 block。

| 层类型 | 主计算 | 主要持久状态 | 优化重点 |
|---|---|---|---|
| Full attention | causal GQA / FlashAttention | paged K/V | 合并投影、partial RoPE 与缓存写入、分块 softmax、prefill/decode kernel 选择 |
| Gated DeltaNet | causal conv + gated delta recurrence | conv pool + recurrent pool | 按 slot 原地更新、寄存器状态、满足条件时 chunk-parallel prefill |

## 4. Full attention：模型主动选择了什么

```text
x
 → StackedLinear(Q + gate, K, V)
 → 按 head 拆 Q / gate；Q/K RMSNorm
 → partial RoPE 的维度重排
 → BF16: store K + store V → fused Q/K RoPE
   FP8 KV: fused RoPE + split + store
 → flash_attention_ragged（causal，paged KV）
 → reshape → output * sigmoid(gate)
 → o_proj
```

1. **合并 Q/K/V 投影在模型层表达。** `qkv_proj` 用 `StackedLinear` 连接 checkpoint 的三份权重；`StackedLinear.__call__` 对拼接权重调用一次 `linear`，其中 Q 投影包含 gate。不是等待编译器从三个 GEMM 发现此优化。证据：`max/python/max/pipelines/architectures/qwen3_5/layers/attention.py:L131-L158,L378-L414`；`max/python/max/nn/legacy/stacked_linear.py:L175-L225,L405-L422`。
2. **Q/K normalization 必须保留 Qwen 数值语义。** `weight_offset=1.0`，`multiply_before_cast=False`；Q/gate 是每 head 交错布局，不能按“所有 Q 后面所有 gate”拆分。证据：`max/python/max/pipelines/architectures/qwen3_5/layers/attention.py:L114-L129,L391-L426`。
3. **Partial RoPE 按实际 kernel 布局转换。** Q/K 从 `[RoPE | NoPE]` 调整成 `[NoPE | interleaved RoPE]`，旋转部分用 reshape → transpose → reshape 交错成对。源码注释对比 slice+concat 为每个 Q/K 5 vs 9 个 graph ops；这是构图操作数，不能等同 GPU launch 数。证据：同文件 `L433-L477`。
4. **BF16 与 FP8 KV 的拓扑不同。** BF16 分别 store K/V，再由 `fused_qk_ragged_rope` 旋转 Q 和 cache 中 K。FP8 路径 `rope_split_store_ragged` 在寄存器完成 RoPE 后再 cast/store，避免先写 FP8 再旋转带来的两次舍入。`_full_width_freqs` 给 NoPE 维补 `(cos,sin)=(1,0)`。证据：同文件 `L283-L316,L478-L528`。
5. **Attention 是显式的大算子。** `_attend` 调 `flash_attention_ragged`；wrapper 以 `ops.inplace_custom("attention.mha.ragged.paged", ...)` 传入 Q、offsets、KV pages、lookup table、cache lengths、scale、dispatch metadata 与 mixed-step split。证据：同文件 `L318-L351`；`max/python/max/nn/_graph_ops/__init__.py:L6298-L6341`。
6. **输出 gate 在图中位于 attention 后。** `sigmoid(gate)`、乘法、`o_proj` 分开表达。证据：`max/python/max/pipelines/architectures/qwen3_5/layers/attention.py:L530-L546`。

## 5. 算子 lowering：编译器具体做什么

### 5.0 Python 到 C++ 编译管线的交接形式（2026-11-18 补充）

**进入第 574 行所组织的编译管线时，输入已经是内存中的 MLIR `ModuleOp`，其中包含 Qwen3.5 的 `mo.graph`、RMO/MO 算子、类型与依赖。** Python Graph API 在构图时就通过原生绑定创建 MLIR 对象；“Python 阶段 / C++ pipeline 阶段”是职责边界，不表示此前从未执行 C++ 绑定。

```text
Qwen3_5 Python Module + 输入类型 / 权重注册
 → GraphSpec.trace → Qwen3_5.trace
 → 使用 Graph 输入的符号 Value 调用 forward / __call__，创建 RMO/MO 图
 → module.mlir_module._CAPIPtr（原生 ModuleOp 句柄）
 → C++ MOFContext.compileToBinary：取回并 clone ModuleOp
 → runGCPipeline：组装 passes → pm.run(moduleOp)
 → MGP / 编译 kernel → MEF
```

构图行为可直接核查：

- `max/python/max/pipelines/lib/graph_specs.py:L144-L163` 调模型的 trace；`max/python/max/pipelines/architectures/qwen3_5/qwen3_5.py:L769-L818` 把平坦的符号输入重组为 tokens、attention cache、recurrent state 等，再进入模型调用。
- `max/python/max/graph/graph.py:L912-L937` 创建 `builtin.ModuleOp`、`mo.GraphOp` 与输入参数；`L979-L993` 用 `self.inputs` 调 `forward` 并登记 outputs。这里图输入是符号 Value，实际请求的 tokens/cache 内容在执行时绑定。
- `max/python/max/graph/ops/matmul.py:L103-L111` 把 matmul 创建成 `rmo.MatmulOp` 并设置结果 dtype；`max/python/max/graph/ops/custom.py:L173-L208` 创建 `mo.custom`，写入 symbol、parameters、device，以及更新有副作用操作的 chain。

因此，一个 attention 层在此时已有“投影、norm、重排、custom RoPE、custom paged attention、gate、输出投影”的**张量级依赖结构**。每个值可携带 shape（含符号维）、dtype 和 device；KV/SSM 是可变 buffer，读写顺序用 chain 表达。大算子保留注册符号；尚未全部展开为 GPU 指令。

以下仅示意图结构，不是本轮编译得到的 IR，也不是可直接运行的 MLIR：

```text
mo.graph @qwen3_5(tokens, row_offsets, kv_pages, recurrent_state, ...) {
  qkv       = rmo.matmul(x, stacked_weight_transposed)
  q, k, v   = split / reshape / norm / partial-RoPE-layout(qkv)
  q, chain  = mo.custom[symbol="kv_cache.fused_qk_rope.ragged.paged"](...)
  a, chain  = mo.custom[symbol="attention.mha.ragged.paged"](...)
  gated     = multiply(reshape(a), sigmoid(gate))
  out       = rmo.matmul(gated, output_weight_transposed)
}
```

交接处没有要求先序列化成文件：`max/python/max/engine/api.py:L1344-L1351` 直接传 `_CAPIPtr`；`max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L654-L676` 将 capsule 中指针写入 `ModelSource`；`max/lib/max_license/API/c/CompiledModels.cpp:L100-L119` 调 `compileToBinary`；`GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L378-L386` 以 `ModuleOp::getFromOpaquePointer(source).clone()` 恢复并复制模块。

| C++ 管线阶段 | 对图做的事 | 输出的抽象层次 |
|---|---|---|
| MO pipeline | 注册/校验 kernel，RMO 合法化，shape、常量、代数与 composite pattern 优化 | 规范化后的张量算子图 |
| MO → MAP，MAP fusion | 将算子表示为 elementwise / rowwise / opaque kernel，合并允许的计算与 view | 选定接口及融合边界的 kernel 图 |
| MAP → MOGG | 下降融合区域，整理 kernel 调用、参数与代码生成例程 | 面向代码生成的图 |
| MOGG → MGP，MGP pipeline | 显式 buffer、内存复用/规划、init 与 execute 的不变计算移动、依赖与设备图处理 | 面向 runtime 的程序 |
| EmitMojo / JitCompileKernels / 导出 | 生成并编译 Mojo/KGEN kernel/glue，导出 MEF | 可加载的执行产物 |

阶段证据：`GraphCompiler/lib/GraphCompiler.cpp:L221-L262,L326-L397,L574-L625`；`GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L356-L375`。具体选择受 options、kernel 注册和目标设备约束，缓存命中可以跳过编译。

**第 574 行是组装 pass pipeline 的位置。** `buildMoPipeline` 等函数把 pass 加到 `PassManager`；`runGCPipeline` 的 `pm.run(moduleOp)` 才实际执行改写。证据：`GraphCompiler/lib/GraphCompiler.cpp:L640-L656`。图编译前 attention 大算子已由 Python 选择；C++ 将其连接到已注册 kernel 并处理可允许的结构融合，kernel 的数值实现仍在 Mojo 源码或对应库中。

#### 交接粒度：完整 Module，内部可以包含层子图

常规 Qwen3.5 Serving 编译传入的是**该编译组的完整 MLIR Module**：有完整 decoder 顶层图，以及它调用的层子图定义。`Module` 是交接/编译容器；`Graph` 和层 subgraph 是容器内部的组织层次，二者不矛盾。

- `use_subgraphs` 默认 True（`max/python/max/pipelines/lib/config/model_config.py:L742-L749`），并传到 Qwen3.5 配置（`max/python/max/pipelines/architectures/qwen3_5/model_config.py:L975-L983`）。模型将 full-attention 层和 linear-attention 层分成两组，分别命名 `qwen3_5_full_attention_block`、`qwen3_5_linear_attention_block`（`max/python/max/pipelines/architectures/qwen3_5/qwen3_5.py:L681-L716`）。这些是 transformer block 子图，包含 attention、MLP 与 norm，不仅是 attention 算子（同文件 `L117-L160`）。
- `forward_sequential_layers` 为每组的第一层构建一个子图，后续层在顶层图中通过 `ops.call` 复用，传各层输入和权重 prefix。关闭子图时，直接逐层调用构图；编译交接仍是完整 Module。证据：`max/python/max/nn/legacy/transformer/transformer.py:L145-L190`；权重 placeholder/prefix 约定见 `max/python/max/nn/legacy/layer/layer.py:L315-L356`。
- 子图通过 `Graph.current.add_subgraph` 创建，并指定 `module=self.module`，所以定义与调用者属于同一个 Module。证据：`max/python/max/graph/graph.py:L1084-L1099`。这不是 Python 每层单独调用一次 C++ compile。
- 文本 checkpoint 的组只有 decoder；有 vision encoder 时，`qwen3_5_vision` 与 decoder 使用相同 `_COMPILE_GROUP`，常规 Serving 一起 trace 到一个 GraphModule 再 compile。证据：`max/python/max/pipelines/architectures/qwen3_5/model.py:L91-L92,L239-L253,L661-L697`；`max/python/max/pipelines/lib/graph_specs.py:L247-L286`。
- `max export` 的 `compile_roles_separately` 可以忽略上述组，将 decoder/vision 各自编译；这仍按模型 role 分开，不是把每层分别传入。证据：`max/python/max/pipelines/lib/graph_specs.py:L190-L211`。

默认文本路径的组织示意（非实际 IR dump）：

```text
Module                    ← 整个交给 C++
 ├─ graph @qwen3_5         ← embedding → 按层 call → final norm / logits
 ├─ subgraph @qwen3_5_full_attention_block
 └─ subgraph @qwen3_5_linear_attention_block
```

这里“完整 decoder 图”指一次 forward 的全部层与数据依赖，不等于把多轮生成的调度流程全部收进图；也不能由此推断编译器必定做跨层融合。融合还受 subgraph/call、kernel 接口和具体 pass 约束。

实际管线见 `GraphCompiler/lib/GraphCompiler.cpp:L574-L625`：

```text
MO 图优化（shape、constant folding、CSE、symbolic、保留的 composite patterns）
 → composite 分解 / stable 标记
 → MOToMAP
 → MAP fusion / view 与 reduction 优化
 → MAPToMOGG
 → MOGGToMGP / EmitMojo
 → JitCompileKernels：LIT → KGEN → elaboration → LLVM archive
 → MEF / GraphRT
```

### 5.1 在 MAP 中暴露可融合的结构

`lowerThroughKernel` 按注册 kernel 的种类下降：reduction → `map.iter.rowwise`，elementwise → `map.iter.elementwise`，其余 → `map.iter.opaque`。slice、transpose、broadcast、reshape、concat 有专门 view lowering 入口。证据：`GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:L815-L869`。

`FuseKernels` 依次尝试 elementwise、views、prologue、reductions、epilogue、reshape epilogue、horizontal fusion，重复直到 kernel 数不再减少。适用于 normalization 周围的算术、gate、cast 等可表达结构；具体模型最终融合成几个 kernel 仍须看编译 IR。证据：`GraphCompiler/lib/MAPDialect/Transforms/FuseKernels.cpp:L36-L88`。

### 5.2 消除拷贝、重复计算与循环内无用开销

`buildMapPipeline` 还执行 Reduction LICM/CSE、累积 buffer 分配、连续 view 还原与融合、LoopInvariantViewMotion、InferMaxSimdWidth。`FuseContiguousViews` 将符合条件的 reshape/连续 slice 作为 kernel 内 stride view 使用，无需每个 view 单独发起 copy kernel；不能推广为所有 transpose/concat 都零拷贝。证据：`GraphCompiler/lib/GraphCompiler.cpp:L326-L365`；`GraphCompiler/lib/MAPDialect/Transforms/FuseContiguousViews.cpp:L14-L18,L78-L100`。

RMSNorm 是 composite：若库中没有实现它的 kernel，`TmpDecomposeComposites` 才分解为 cast、square/reduce.add、rsqrt、weight-offset 与乘法，并保留 multiply-before-cast 的分支。不是所有构建都必定分解。证据：`GraphCompiler/lib/MODialect/Transforms/TmpDecomposeComposites.cpp:L137-L193`。

仓库还包含 Qwen3.5 专项回归输入：要求静态 RoPE `freqs_cis` 的 H2D 移入 model init，避免每步 execute 重传。证据：`GraphCompiler/test/mo-opt/MGPDialect/Transforms/qwen3_5-rope-freqs-hoisting.mlir:L1-L22`。这里只审阅测试目的，未执行该测试；动态 M-RoPE 位置计算不因此全部消失。

### 5.3 不能越过的融合接口边界

`MOToMAP.lowerToOpaque` 仅在 kernel 参数 `isFusedTensorType()` 为真时构造 load prologue / store epilogue。trait 参数必须明确带 `Fused`。当前 paged MHA 注册的 Q 为 `Some[Input & Tensor]`，输出为 `Some[Output & Tensor]`，不带该标记。因此不能声称 QK norm、partial RoPE、sigmoid gate 经通用 prologue/epilogue 自动并入这个 FlashAttention kernel；连续 view 的消除是另一个机制。

证据：`GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:L142-L162,L219-L253`；`GraphCompiler/include/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h:L375-L386`；`GraphCompiler/lib/MOGGDialect/Analysis/KernelDeclAdaptor.cpp:L887-L905`；`max/kernels/src/graph_compiler/builtin_kernels/attention.mojo:L4228-L4265`。

另有 `MatmulFusedPartialRMSNormPattern`，但它要求 `multiply_before_cast=True`、M=1、两路 split；Qwen3.5 当前 Q/K norm 明确为 False。存在一个同名或相似优化 pass，不等于本模型可命中它。`ReduceRMSNormRoPEPattern` 匹配显式 half-rotate 算术；此处模型使用 custom cache RoPE，也不能直接套用。证据：`GraphCompiler/lib/MODialect/Patterns/Fusion.cpp:L2058-L2074,L3510-L3555`。

### 5.4 专门 kernel 的编译与缓存

`JitCompileKernels` 为 generator 查询缓存；miss 后运行 elaboration，再通过 `ObjectCompiler.emitArchive` 下降到 LLVM 并保存产物。其 GPU tile、head geometry、dtype 等编译时参数来自 kernel 的泛型/`comptime` 实现；不是重新从任意 Python 算术搜索 FlashAttention 算法。证据：`GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L199-L220,L409-L480`。

## 6. Full attention 的硬件优化落点

`attention.mha.ragged.paged` 注册进入 `_execute_mha_ragged_paged`，重建 paged cache 后走 `generic_flash_attention_kv_cache_ragged` 与 GPU dispatch。页表间接寻址使 kernel 直接读取 paged K/V；无需先将历史 KV 拼成连续大张量。证据入口：`max/kernels/src/graph_compiler/builtin_kernels/attention.mojo:L4228-L4440`，`max/kernels/src/nn/kv_cache_ragged.mojo` 的 `generic_flash_attention_kv_cache_ragged`。

- 官方 FlashAttention 设计文档说明分块 QK、online softmax 和 PV 避免物化整个 S×S attention matrix。证据：`max/docs/design-docs/multi-head-flash-attention.md:L64-L145`。这是 kernel 算法机制，不是 graph CSE 的结果。
- 当 SM100、dtype、mask、head-dim 满足 dispatch 条件时，depth=256/512 且 `not is_token_generation`、`max_prompt_len * group > 32` 进入 pair-CTA prefill；短 prefill/decode 走后续 FA4 dispatch。没有指定设备时，不能断定最终就是这条分支。证据：`max/kernels/src/nn/attention/gpu/mha.mojo:L1094-L1195`。
- 该 pair-CTA kernel 使用相邻两个 SM 协作，专门安排 softmax、correction、MMA、TMA load warp；depth=256 的几何为 MMA_M=256、BM=128、BN=128。证据：`max/kernels/src/nn/attention/gpu/nvidia/sm100/mha_depth512/kernel.mojo:L13-L28`。这类布局和流水是手写 Mojo 实现的显式选择，不能全归因于图编译器自动优化。
- 新的 paged MHA wrapper 对满足条件的 NVIDIA mixed step 可依据 split record，将 decode rows 与 prefill rows 作为两个 view 分派并写入同一输出；无有效 split 或存在不满足条件的 operand 时保留单路。证据：`max/kernels/src/graph_compiler/builtin_kernels/attention.mojo:L4230-L4236,L4365-L4435`。

## 7. Gated DeltaNet：另一条必须审查的路径

```text
合并输入投影（QKV / z / beta / decay projections）
 → causal conv1d（slot-indexed pool 原地更新）→ SiLU
 → decode：gated_delta_recurrence_fwd
   prefill：条件满足时 kda_chunk(use_computebound=True)，否则顺序 recurrence
 → RMSNorm → SiLU(z) gate → output projection
```

模型入口：`max/python/max/pipelines/architectures/qwen3_5/layers/gated_deltanet.py:L523-L617,L657-L701,L735-L841`。

- 开源 recurrence kernel 每个线程持有一个 KD 元素的 fp32 状态列，序列循环中在寄存器更新；完成后写回 slot。代码将 decay、memory dot、delta update、output dot 放在同一个 per-token step。证据：`max/kernels/src/state_space/gated_delta.mojo:L145-L239,L389-L442`。寄存器驻留是源码设计意图，实际是否 spill 仍需编译/设备 profile。
- 非投机路径用运行时 `total_tokens > batch_size` 区分包含多 token 的 batch，`ops.cond` 在 chunk prefill 与顺序 decode 之间选择。所有序列长度为 1 时走 decode；不是将所有 attention 层统一调用 FlashAttention。
- **当前 Qwen3.5 模型 gate 要求 CUDA + SM100+ + 支持的 `(KD,VD)` + 已注册 `kda_chunk`**。支持维度表为 `(32,32)` / `(128,128)`；不满足时 prefill 也走顺序 recurrence。底层通用 wrapper 还有 AMD 约束，不等于此 Qwen3.5 模型 gate 会启用 AMD chunk。证据：`gated_deltanet.py:L735-L753,L807-L818`；`max/python/max/nn/_graph_ops/state_space/kimi_delta.py:L83-L132`。
- **开闭源边界：chunk kernel 随闭源 kernel 包交付**，纯源码构建缺少注册；可审阅 Python wrapper 的 chunk-parallel/compute-bound 契约，内部 kernel 融合和性能为 `UNKNOWN`。证据：`max/python/max/nn/_graph_ops/state_space/kimi_delta.py:L98-L109,L905-L939`。文件顶部“两遍处理 prefill/decode”的旧概述不覆盖当前新增的条件分支，应以可执行分支为准。
- chunk 输入会把 per-head gate 展开成 per-channel 实体数据供 TMA 读取，源码特意用乘 ones 而非可能保持 stride-0 的 broadcast view；这说明消除物化必须服从 kernel 内存契约。证据：`gated_deltanet.py:L622-L655`。

## 8. 后续验证入口与工程局限

本轮只完成源码更新与静态证据闭环；未下载模型、未生成该 checkpoint 的 IR/MEF、未测数值或性能。不能报告“全 attention 单 kernel”“固定 launch 数”“加速倍数”或“闭源 chunk 的内部流水”。

若需确认指定设备/模型的实际融合，可在准备好 MAX 构建环境后使用 `MODULAR_MAX_TEMPS_DIR=<输出目录>` 导出编译中间阶段，依次检查 `mo-pre-map`、`mo-to-map`、`map-pre-mogg`、`map-to-mogg`、`mogg-to-mgp`、`mgp`。配置解析与阶段入口：`GraphCompiler/lib/Driver/Common/Utils.cpp:L46-L56`；`GraphCompiler/lib/GraphCompiler.cpp:L570-L625`。缓存命中可能跳过编译，应记录 cold/warm 状态。

重点核对：Q/gate 的 per-head 布局、partial RoPE 数值顺序、FP8 KV 的舍入边界、gate 与 view 的最终归属、静态 freqs 是否 hoist、DeltaNet chunk gate 是否命中，再分别测 prefill 和 decode。未完成这些验证前，结论成熟度保持 `STATIC_REVIEWED`。
