# Google TPU / XLA 软件栈

> **资料角色声明**：
> 本目录归档早期集中保存的 Google 相关组件调研（从模型服务、框架前端、XLA/PJRT 到 TPU 后端），用于与 Ascend、QAIC、Tenstorrent 路线对比，深入调研 AOT 静态编译与 SPMD 软件栈架构。
> **权威文档关系**：TPU 路线的唯一权威架构与兼容性结论位于 [`../20260903-vllm-ai-infra-research/13-tpu/`](../20260903-vllm-ai-infra-research/13-tpu/)。
> **代码仓库位置说明**：本地代码统一存放于 `google/third_party/` 目录下（如 `google/third_party/tpu-inference/`、`google/third_party/torch_tpu/`），固定 Revision 详见 [`20260903-150`](../20260903-vllm-ai-infra-research/15-evidence-review/20260903-150-源码Revision与权威资料索引.md)。

## 端到端关系

```text
vLLM
  ├─→ tpu-inference / vllm-jax (google/third_party/tpu-inference)
  │     ├── JAX-Native 路线: flax.nnx 模型定义 (Llama, DeepSeek, Qwen 等)
  │     └── vLLM-PyTorch 路线: VllmModelWrapper → torchax (张量与算子桥接)
  └─→ vllm-torchtpu (独立仓库) → torch_tpu (google/third_party/torch_tpu)
        └── PyTorch 原生路线: PrivateUse1 后端 + DeferredOp（无 JAX runtime 依赖）
  → StableHLO / HLO IR 生成
  → OpenXLA Compiler (全图优化、SPMD 自动分片、Shape Bucketing)
  → PJRT C ABI (PJRT_Api / pjrt_c_api.h)
  → libtpu.so / TPU Runtime
  → Linux Kernel Driver / Firmware / TPU 硬件 (v4/v5e/v5p/v6e)

对照系统 (Google 原生生态):
JetStream / JetStream-PyTorch / MaxText / Saxml
  → JAX / Flax 或 PyTorch/XLA
  → OpenXLA / PJRT C ABI
  → TPU 硬件
```

`libtpu`、TPU Driver、Firmware 的核心实现未开源（标记为 `UNKNOWN`）；分析这部分时必须明确源码证据边界。

## 项目说明（位于 `google/third_party/`）

| 目录（位于 `google/third_party/`） | 主要功能 | 在研究链路中的位置 |
| --- | --- | --- |
| `tpu-inference/` | vLLM TPU 官方插件，包含 Platform、TPUModelRunner、模型执行（`flax_nnx` / `torchax`）、KV Cache、CompilationManager 预编译缓存、shape bucket 分桶及分布式适配 | vLLM 与 TPU 编译/Runtime 的直接连接层（固定 SHA: `fd338000`） |
| `torch_tpu/` | Google 官方开源 PyTorch 原生 TPU 后端：`PrivateUse1` 接入、ATen 算子 DeferredOp、StableHLO→XLA 编译（O1/O2）、三级编译缓存、bounded dynamism 与 `tpu_dist` | PyTorch 原生 Serving 路径的执行/编译底座（固定 SHA: `d7a3c1cc`） |
| `xla/` | OpenXLA 编译器和 PJRT 接口实现，负责 HLO 优化、设备 lowering、executable 构建与 `PJRT_Api` C ABI 抽象 | 编译器与 Runtime 接口核心（固定 SHA: `52a05962`） |
| `jax/` | JAX 前端及 jaxlib 接口，负责 tracing、JAXPR、lowering，并通过 XLA/PJRT 执行 | 动态 Python 前端与 XLA 入口（固定 SHA: `8a3235a3`） |
| `jetstream/` | 面向 TPU 的 LLM 推理服务框架，定义 Engine、请求处理、prefill/decode 和服务入口 | Google 原生 Serving 对照实现（固定 SHA: `acd4f5a`） |
| `jetstream-pytorch/` | 使用 PyTorch/XLA 模型接入 JetStream，包含分片、服务和 disaggregated inference 示例 | PyTorch/XLA Serving 路线（固定 SHA: `3734038`） |
| `maxtext/` | 基于 JAX/MaxText 的大模型训练、推理和服务实现，覆盖模型、分片、量化与性能配置 | JAX/TPU 大模型工作负载参考（固定 SHA: `c5544359`） |
| `paxml/` | 基于 Praxis 的大规模训练与实验编排框架 | 高层训练/实验系统参考（固定 SHA: `0ceed30`） |
| `praxis/` | JAX 神经网络层、模型组件、分片和训练基础库 | PaxML/SAX 的模型基础层（固定 SHA: `a6e2a49`） |
| `saxml/` | 面向 JAX/Pax 模型的在线推理服务系统，包含管理、客户端和服务端 | TPU Serving 架构对照实现（固定 SHA: `049730e`） |
| `t5x/` | 基于 JAX/Flax 和 XLA 的 T5/Transformer 训练与推理框架 | 经典 JAX/XLA 模型栈参考（固定 SHA: `2045b33`） |
| `gemma/` | Gemma 模型、推理、微调和研究实现（Google DeepMind 官方实现） | 模型层与端到端验证工作负载（固定 SHA: `7b78599`） |

## 调研重点

1. `google/third_party/tpu-inference/`：vLLM 如何把动态请求正规化为 TPU 可编译、可缓存的执行形态。
2. `xla/xla/pjrt/`：Client、Device、Buffer、LoadedExecutable、Compile/Execute 的稳定 Runtime 边界。
3. `xla/xla/tpu/`：TPU lowering、编译目标和拓扑相关实现；闭源调用处需标明边界。
4. `jax/`：动态 shape、分片、异步 dispatch 和 PJRT device/buffer 语义。
5. `jetstream*`、`maxtext/`、`saxml/`：prefill/decode、KV Cache、批处理、分片和生产服务策略。
6. `torch_tpu/`：PyTorch 原生路径的端到端机制（DeferredOp→物化/编译→PjRt），见 [`docs/aot/20261023-TorchTPU-开源架构与vLLM接入.md`](docs/aot/20261023-TorchTPU-开源架构与vLLM接入.md)。

## 来源快照

- `gemma`: `7b78599`
- `jax`: `8a3235a3b`
- `jetstream`: `acd4f5a`
- `jetstream-pytorch`: `3734038`
- `maxtext`: `c55443590`
- `paxml`: `0ceed30`
- `praxis`: `a6e2a49`
- `saxml`: `049730e`
- `t5x`: `2045b33`
- `xla`: `52a0596243`
- `tpu-inference`: `fd33800041510b957cd2da6199742cce2b5fd113`
- `torch_tpu`: `d7a3c1cc73a115376a058c7664986f5e34ca8362`
