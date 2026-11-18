# SGLang NVIDIA MoE：路由整理与专家 GEMM 准备重叠

- 调研日期：2026-11-18（Asia/Shanghai）。
- 固定源码：`sgl-project/sgl-nvidia-mega`，commit `ad940462486c3fbd0a42348ab215b183e5db5d29`。
- 范围：DeepSeek-V4.1 小批量 routed experts，重点核查 `rows <= 8` 的 `route.cu` 路径；不将结论推广到其他模型或宽批次。
- 文档角色：补充源码阅读记录，不增加 105 篇 Manifest 或改写 vLLM 基线 Claim。
- 权威主题对应：[51：Inductor Pass、Fusion 与 CustomOp](../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-51-InductorPass、Fusion与CustomOp.md)、[52：CUDAGraph、PiecewiseCapture 与 Shape](../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-52-CUDAGraph、PiecewiseCapture与Shape.md)。这些是主题导航，不表示其原有 vLLM Claim 覆盖本插件。
- 证据等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`；无 GPU 实测。

## 1. 官方资料与路线边界

先检视仓库树：无顶层 `doc/`、`docs/`；优先阅读该 commit 的 [README](https://github.com/sgl-project/sgl-nvidia-mega/blob/ad940462486c3fbd0a42348ab215b183e5db5d29/README.md) 及 [SGLang 官方说明](https://www.lmsys.org/blog/2026-10-18-sglang-low-latency-mode/)。README 明示插件为 Experimental。

路线侧重是 SGLang 的小批量 Decode/CUDA Graph Replay 专用优化；未覆盖 step 继续使用通用 forward。本文研究成熟度评级为 **★★☆☆☆（实验性专用路径）**，依据是 README 的 Experimental 标记与模型、硬件、行数覆盖限制；不据此评级整个 SGLang，也不声称它是通用生产主力。AOT 全模型静态编译不在本次核查范围。

## 2. 最小事实与调用链

`MoeSlot.run -> RoutedExpertGemm.gate_topk_and_gemms -> gate_topk -> _route -> _gemms(GEMM1, GEMM2)`。

| 环节 | 固定源码位置 | 已核查事实 |
| --- | --- | --- |
| 入口 | `sgl_nvidia_mega/deepseek_v41/moe/routed.py:L124-L155` | 量化输入的 event join 放在 top-k 前；注释指出放在 route 与 GEMM1 之间会破坏 GEMM1 提前启动。 |
| 调用组织 | `sgl_nvidia_mega/deepseek_v41/sm103/moe/routed_expert_gemm.py:L435-L504` | gate top-k 输出 ids，再提交 route，随后提交 GEMM pair；沿用同一当前 stream。 |
| 启动配置 | `sgl_nvidia_mega/common/jit.py:L155-L195` | `launch` 设置 `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`，再调用 `cuLaunchKernelEx`。 |
| top-k 的提前释放 | `sgl_nvidia_mega/deepseek_v41/sm103/moe/src/gate_topk.cuh:L55-L75` | 函数开头触发 dependents，读取依赖数据前执行自身 wait。 |
| route 的提前释放 | `sgl_nvidia_mega/deepseek_v41/sm103/moe/src/route.cu:L23-L36` | 入口立即 `griddepcontrol.launch_dependents`；清理自身共享 scratch 后 wait，之后读 ids、构建路由记录。 |
| GEMM1 的独立准备 | `sgl_nvidia_mega/deepseek_v41/sm103/moe/src/gemm1.cu:L460-L610` | 先建立共享内存地址、初始化 mbarriers、分配 TMEM、同步线程并调整寄存器预算，再到 grid-dependency wait。 |
| 专家权重读取 | `sgl_nvidia_mega/deepseek_v41/sm103/moe/src/gemm1.cu:L1456-L1497` | 加载 warp 先 wait，再读路由表中的 expert id，随后用此 id 发起专家权重的 TMA 加载。 |

以上路径均相对本节固定的远程仓库，而非本工作区的 vLLM 子模块。

## 3. 重叠的究竟是什么

这里要区分两件事：top-k 选择专家；route 把已选的 token/expert 关系整理为 GEMM 需要的调度表。route 的数据处理依赖 top-k 的 ids，但其独立准备可以提前进行。GEMM1 同样可以先执行不读取 route 输出的准备工作。

```text
串行示意：route [构建调度表]  GEMM1 [初始化资源][读表、载入权重、矩阵乘]
PDL 示意：route [构建调度表────────]
                         GEMM1 [初始化资源][wait][读表、载入权重、矩阵乘]
```

时间线仅解释依赖关系，不表示真实持续时间或保证并发。`launch_dependents` 允许后继提前启动，不表示路由结果此时已经可读；后继的 `griddepcontrol.wait` 保留结果可见性的依赖边界。

因此本路径明确支持的是 **GEMM setup 与 routing 重叠**。不能据此声称专家选择尚未完成时，已经读取所选专家权重或开始该专家的矩阵乘法；已核查的权重加载 warp 在 wait 之后才读取 expert id。

## 4. 小批量 route 的简化

`route.cuh:L29-L34,L99-L141` 利用每个 token 的 top-k ids 不重复、总输入不超过八行，使每个被选专家可放进一个八行 tile。`present` 位图标记哪些专家出现，`rows_of[expert]` 标记哪些 token 选了该专家，通过 `__popc` 计算专家的 tile 编号与 token 在 tile 内的位置，生成 `CTA -> expert`、行界限、`padded row -> token`。

`routed_expert_gemm.py:L77-L81,L479-L480,L531-L560` 用输入行数、top-k 和专家数计算 CTA 数上界后提交 GEMM；运行时读取路由表决定实际工作。`gemm1.cu:L646-L650` 在 tile 超出实际数量时退出。因而提交 Kernel 不需要 CPU 先拿到每个专家的精确 token 数。

## 5. 性能与开闭源边界

- 官方 profiling 报告 MoE 区域中位数从 44.4 微秒降至 36.5 微秒，涉及路由简化、提前启动、跳过 padding 中的全零 tile 等组合；不能把全部收益归因于 PDL。
- 本次只读开源调用与 CUDA/PTX 指令；驱动的实际调度决策、不同资源占用下的真实重叠比例为 `UNKNOWN`，未推演闭源内部实现。
- 官方实验硬件与场景是 GB300、DeepSeek-V4.1-Flash/DSpark 小批量；本文固定源码是 2026-11-18 的 commit，不声称它就是官方文章测量时的精确 revision。
- `route_wide.cu` 与其他模型不在本次逐行核查范围；不声称存在按专家逐块发布后就立刻消费的细粒度路由流水线。

## 6. 精确远程源码入口

- [route.cu：提前释放与等待](https://github.com/sgl-project/sgl-nvidia-mega/blob/ad940462486c3fbd0a42348ab215b183e5db5d29/sgl_nvidia_mega/deepseek_v41/sm103/moe/src/route.cu#L23-L36)
- [GEMM1：资源初始化与首次等待](https://github.com/sgl-project/sgl-nvidia-mega/blob/ad940462486c3fbd0a42348ab215b183e5db5d29/sgl_nvidia_mega/deepseek_v41/sm103/moe/src/gemm1.cu#L513-L610)
- [GEMM1：等待之后读取专家 id 和权重](https://github.com/sgl-project/sgl-nvidia-mega/blob/ad940462486c3fbd0a42348ab215b183e5db5d29/sgl_nvidia_mega/deepseek_v41/sm103/moe/src/gemm1.cu#L1456-L1497)
- [route.cuh：位图与 popcount](https://github.com/sgl-project/sgl-nvidia-mega/blob/ad940462486c3fbd0a42348ab215b183e5db5d29/sgl_nvidia_mega/deepseek_v41/sm103/moe/src/route.cuh#L99-L141)
