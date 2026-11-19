# NVIDIA 软件栈研究导航

> 本目录保存 NVIDIA 专题的源码核查与官方契约解读，作为 105 篇权威集的厂商补充导航；不新增 Manifest 文档。第三方源码只读。

## TensorRT-RTX

- [整体架构与源码阅读](./tensorrt-rtx-architecture.md)：模型入口 → AOT Builder → 可分发 Engine → 终端 JIT/ExecutionContext → CUDA 执行；区分 Engine、Runtime Cache 与 CUDA Graph。§4.2 解释网络图编译边界：可以是整模型或子模块，单 Engine 不等于单 Kernel。
- [此前的定位与开源边界核查](../../analysis-design/02-vendor-research/41-tensorrt-rtx-overview-and-source-boundary.md)。
- 权威主题对应：[54：执行模式分类](../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-54-Eager、JIT、AoT、Trace与Replay.md)、[52：图捕获与 Shape](../../20260903-vllm-ai-infra-research/05-vllm-compile/20260903-52-CUDAGraph、PiecewiseCapture与Shape.md)。主题对应不表示固定 vLLM Claim 已覆盖 TensorRT-RTX。

**路线侧重**：TensorRT-RTX 的官方主交付路线是 **AOT + 终端 JIT**。CUDA Graph 是编译后执行的可选加速机制；当前源码的原生 API 默认关闭，示例通过 `WHOLE_GRAPH_CAPTURE` 显式启用。不能将两者当成互斥产品路线。

**成熟度研究评级：★★★☆☆**。依据是已发布 SDK、C++/Python API、动态形状与真实模型示例；本工作区仅做静态核查，未安装 SDK、未在 GPU 上验证，因此不作生产就绪或性能承诺。

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
