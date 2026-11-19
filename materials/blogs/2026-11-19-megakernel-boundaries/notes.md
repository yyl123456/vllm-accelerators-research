# MegaKernel 的收益、成本与适用边界

- 调研日期：2026-11-19（Asia/Shanghai）。
- [材料] DeepPrompting《MegaKernel 不是万金油》，[原文](https://mp.weixin.qq.com/s/Lj8HkN-yy4jc8tWlEgCJJw)，2026-10-17；网页无页码，按章节定位。
- 材料路径：`materials/blogs/2026-11-19-megakernel-boundaries/`；[正文](article.md)、[可检索全文](extracted.txt)、[原始 HTML 无损压缩归档](original.html.gz)。
- 源码基线：vLLM `bb363db9a5ec2edc7b39e99b00af363a89d1fb81`，只读核对 CUDA Graph 路径，未审查 MegaKernel 项目实现。
- 权威对应编号：[52：CUDA Graph、Piecewise Capture 与 Shape](../../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-52-CUDAGraph、PiecewiseCapture与Shape.md)、[54：执行模式分类](../../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-54-Eager、JIT、AoT、Trace与Replay.md)。本笔记属于 secondary 资料解读，不进入 105 篇权威集。

## 核心观点与执行链路

文章讨论的是融合粒度的工程取舍：减少 launch、中间张量物化与同步，必须覆盖设备端调度和资源折中的新增成本。

| 机制 | 执行方式 | 主要优化对象 |
|---|---|---|
| 局部 Fusion | 合并一小段算子 | 部分中间读写与启动 |
| Persistent Kernel | 驻留线程循环处理任务 | 反复启动与任务分发 |
| MegaKernel（本文口径） | 跨算子或阶段，在设备内处理任务顺序与依赖 | 跨阶段启动、通信与计算重叠、部分物化 |
| CUDA Graph | 捕获多个操作及依赖，重复 replay | Host 逐操作提交成本 |

概念可组合：Persistent 不要求融合整模型，MegaKernel 也不要求采用动态任务队列。

普通执行链为 Host 提交 A → B → C，各 kernel 仍有独立的配置与依赖边界。Graph 路径预先捕获该链，之后 replay；MegaKernel 将其中一段下沉成设备执行体，由内部任务顺序、事件或队列维持依赖。

融合只可能让适当的中间结果保留在寄存器或共享内存中。跨 CTA 数据交换、权重和 KV Cache 访问仍需依据实际实现分析，不能声称整个模型数据都留在片上。

## 已核对的一手证据

1. **Ada-MK 混合执行，论文 v1 §4.3.1。** [作者论文](https://arxiv.org/html/2605.11581v1#S4.SS3.SSS1) 明确保留 TensorRT-LLM 的 Prefill 原生算子，Decode 才切换 MegaKernel；Embedding、KV 管理及业务能力复用已有 runtime。论文研究的是 L20 与指定 Qwen 小模型配置，不证明所有 Prefill 都不适合融合。
2. **Event Tensor，§4.5、Tables 2–3。** [作者投稿 PDF](https://openreview.net/pdf/4ce9ffc529dcb3b18619959ae62129551da5e049.pdf) 的搜索索引全文显示：Qwen3-32B、TP=4，Dynamic 相对 unfused megakernel 为 0.82–0.89；MoE 1024 tokens 时 Dynamic 为 1.08，Static 为 1.04。基线是单 launch、阶段间全局 barrier 的 persistent kernel，不能误写为 Eager 或 CUDA Graph。直接打开 PDF 触发 OpenReview 浏览器验证，本次依据搜索服务提供的论文索引段落核对，未保存 PDF。
3. **CUDA Graph 官方基线。** [NVIDIA 官方文章](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/) 报告 CUDA 12.6、RTX 3060 的已上传直线型 kernel-only graph 重复 CPU launch API 开销，约 2.5 μs + 1 ns/node（至少约 10 节点）。这是特定测量，不能等同于 GPU 全部执行时间。

## 固定源码对照：Graph 并不等于算子融合

先阅读官方设计入口 `third_party/vllm/docs/design/cuda_graphs.md`，再核对源码：

- `third_party/vllm/vllm/compilation/cuda_graph.py:L233-L263`：`CUDAGraphWrapper.__call__` 根据 forward context、模式与 batch descriptor 选择直接执行或缓存项。
- `third_party/vllm/vllm/compilation/cuda_graph.py:L283-L319`：创建 `torch.cuda.CUDAGraph`，在捕获上下文调用原 runnable。
- `third_party/vllm/vllm/compilation/cuda_graph.py:L346-L361`：可检查输入地址，随后执行 `entry.cudagraph.replay()` 并返回缓存输出。

上述证据只支持 capture/replay 执行契约；不能据此推断 runnable 内部融合程度，也不能证明 vLLM 已集成本文 MegaKernel。

## 如何理解适用边界

- **小批量 Decode**：每步工作较小，算子间隙和中间读写可能占较大比例；文章与 Ada-MK 案例支持针对这类负载评估。
- **Prefill/大矩阵**：专用 GEMM 和 Attention 效率更重要；减少启动的收益必须覆盖放弃专用配置的代价。
- **GPU 动态调度**：Queue、Atomic 和等待仍有成本。Event Tensor 只证明特定规则与不规则负载的差异，不能推广为所有动态调度都慢。
- **跨节点**：文章将细粒度 payload/signal 顺序与 fence 开销列为边界。本次未复核相关通信论文，具体 10×、2%、98% 数字保留为二手转述，不作为独立结论。
- **资源与运维**：文章列出寄存器、共享内存、驻留进度、取消、公平性与调试责任；本次未复现相关 issue，不给出跨硬件通用退化倍率。

## 开闭源边界与工程局限

本次完成网页阅读、一手文献对照及 vLLM Graph 路径的静态核对，没有性能实测、数值验证或硬件成熟度认证。Ada-MK 所述生产部署是作者报告；生产部署内部实现、线上流量及未公开细节为 UNKNOWN。

文章关于 RaMP、AutoMegaKernel、Perseus 与 Mirage/Triton/XLA issue 的各项数字未逐项核验。不得将其中的固定 batch、特定 GPU 或 weak-scaling 结果直接套到当前 vLLM 服务。

评估建议属于 RECOMMENDATION：在同精度、同模型、同工作负载下比较 Eager、Graph、局部融合加 Graph 和 MegaKernel，分别看 Prefill、Decode、端到端延迟及吞吐；不要只比较 kernel 数量。
