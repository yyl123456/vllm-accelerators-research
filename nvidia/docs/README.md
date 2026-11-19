# NVIDIA 软件栈研究导航

> 本目录保存 NVIDIA 专题的源码核查与官方契约解读，作为 105 篇权威集的厂商补充导航；不新增 Manifest 文档。第三方源码只读。

## TensorRT-RTX

- [整体架构与源码阅读](./tensorrt-rtx-architecture.md)：模型入口 → AOT Builder → 可分发 Engine → 终端 JIT/ExecutionContext → CUDA 执行。§4 说明图编译边界及对象职责；§5 说明三层 JIT 复用、Runtime Cache 与 CUDA Graph；§6 说明动态形状、主机参与和内存；§7 追踪 FLUX.1 多 Engine/Python 去噪循环及 FLUX.2 子模块编译。
- [此前的定位与开源边界核查](../../analysis-design/02-vendor-research/41-tensorrt-rtx-overview-and-source-boundary.md)。
- 权威主题对应：[54：执行模式分类](../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-54-Eager、JIT、AoT、Trace与Replay.md)、[52：图捕获与 Shape](../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-52-CUDAGraph、PiecewiseCapture与Shape.md)。主题对应不表示固定 vLLM Claim 已覆盖 TensorRT-RTX。

**路线侧重**：TensorRT-RTX 的官方主交付路线是 **AOT + 终端 JIT**。CUDA Graph 是编译后执行的可选加速机制；当前源码的原生 API 默认关闭，示例通过 `WHOLE_GRAPH_CAPTURE` 显式启用。不能将两者当成互斥产品路线。

**成熟度研究评级：★★★☆☆**。依据是已发布 SDK、C++/Python API、动态形状与真实模型示例；本工作区仅做静态核查，未安装 SDK、未在 GPU 上验证，因此不作生产就绪或性能承诺。

## 官方架构图索引

2026-11-19 检索固定 SHA 仓库的图片、Markdown 与两个 Notebook 的 Markdown 单元：仓库内只有 HelloWorld 的独立 ONNX 网络 PNG，未找到总体 runtime/compiler 架构图；Notebook 中未找到架构图的 Markdown 引用。总体架构图应阅读以下官方资料。

| 图 | 展示的范围 | 官方入口 |
| --- | --- | --- |
| 两阶段编译图（当前文档） | 离线/CPU AOT → 可分发 Engine → 终端 GPU JIT → 可执行 Engine | [Architecture Overview / Two-Phase Compilation](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/architecture/architecture-overview.html#two-phase-compilation)；[SVG 原图](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_images/trtrtx-two-phase-compilation.svg) |
| 两阶段编译图（2025 首发博文 Figure 4） | GPU 无关 AOT 与 GPU 专用 JIT 的部署分工 | [官方博文](https://developer.nvidia.com/blog/nvidia-tensorrt-for-rtx-introduces-an-optimized-inference-ai-library-on-windows/)；[PNG 原图](https://developer-blogs.nvidia.com/wp-content/uploads/2025/05/JIT-GPU.png) |
| Runtime 线程/对象图（当前文档） | 共享 ICudaEngine、各线程独立 ExecutionContext、CUDA stream 与应用 I/O buffer | [How TensorRT-RTX Works / Threading](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/architecture/how-trt-rtx-works.html#threading)；[SVG 原图](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_images/trtrtx-threading-model.svg) |
| Windows ML 生态图（2025 首发博文 Figure 1） | TensorRT-RTX 在 Windows ML 推理栈中的位置 | [PNG 原图](https://developer-blogs.nvidia.com/wp-content/uploads/2025/05/RTX-Windows.png) |
| HelloWorld 网络图（本地仓库） | Input → MatMul → ReLU → MatMul → Output，是示例网络拓扑 | [本地 PNG](../third_party/TensorRT-RTX/samples/helloWorld/helloWorldOnnx.png) |

上述图均来自项目或 NVIDIA 官方材料；当前文档与 2025 首发图代表不同时间点，不能将图示当成闭源内部 IR/pass/codegen 源码证明。

## 首次公开发布的时间

2026-11-19 核对 Git 历史、GitHub API 与官方发布博文：

- 官方首次宣布产品的博文标注 **2025-05-19**，说明通过 Windows ML 预览提供，并计划 6 月交付独立 SDK。[官方宣布](https://developer.nvidia.com/blog/nvidia-tensorrt-for-rtx-introduces-an-optimized-inference-ai-library-on-windows/)。
- GitHub 首个 Release `v1.0.0.21` 的 `published_at` 为 `2025-06-11T22:39:38Z`，即 **北京时间 2025-06-12 06:39:38**。[首个 Release](https://github.com/NVIDIA/TensorRT-RTX/releases/tag/v1.0.0.21)。本地根提交为 `15c2f47fa59230e8971e70073432061796d30cbf`，标题 `TensorRT-RTX 1.0 Release`。
- 官方 **2025-06-12** 发布博文明确宣布 SDK 可下载，同时提供 GitHub API samples/demos。[官方首发说明](https://developer.nvidia.com/blog/?p=101673)。这里的开源对象是配套示例和 Demo，不表示核心 SDK 全部开源。
- GitHub 仓库 `created_at` 为 `2025-06-05T19:55:09Z`；创建时间和 Git commit 时间均不能单独证明最早转为公开的时刻。已核查的 Release 与正式博文支持以 **2025 年 6 月中旬** 作为公开发布节点。

## 本地源码与复现

| 仓库 | 本地目录 | 本次固定 SHA | 交付边界 |
| --- | --- | --- | --- |
| NVIDIA/TensorRT-RTX | `nvidia/third_party/TensorRT-RTX/` | `cbffacf00c7d681bb31793e5413463e836af10d3` | samples、demo 等配套源码；核心 SDK 需另行安装 |

调研日期：2026-11-19（Asia/Shanghai）。该克隆与现有 NVIDIA 第三方目录一样受 `.gitignore` 保护，属于普通独立 Git checkout，不是主仓库 submodule；主仓库只提交研究文档。目标仓库没有独立 `AGENTS.md`，没有顶层 `doc/`、`docs/`，先读其 README 与官方在线架构文档，再追踪示例。

在工作区根目录复现：

```bash
git clone https://github.com/NVIDIA/TensorRT-RTX.git nvidia/third_party/TensorRT-RTX
git -C nvidia/third_party/TensorRT-RTX checkout --detach cbffacf00c7d681bb31793e5413463e836af10d3
```

已存在该路径时应先检查其 origin、HEAD 与工作树，勿覆盖本地内容。其他已存在的 NVIDIA 源码不在本专题核查范围。
