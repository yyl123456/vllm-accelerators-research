# TT-Metal、TTNN 与 ModelGenerator 软件栈

## 文档契约

- owner：Tenstorrent 从 host API 到模型 generator 的软件分层。
- 依赖输入：`120`、`121`。
- 唯一输出：各层 ABI、状态 owner 与移植责任。
- 不讨论：vLLM plugin hook 和 scheduler；分别由 `123`、`124` 持有。

## 分层

```text
vLLM adapter / ModelGenerator
        |
      TTNN: tensor, layout, distributed mapping, high-level ops
        |
   TT-Metalium: device, buffer, program, kernel, CB, command queues
        |
 runtime/firmware/driver -> Tensix/NoC/DRAM/fabric
```

TT-Metalium 负责显式 device/program/buffer/kernel 生命周期；TTNN 提供 Python/C++ tensor 与 op 层并保留 layout、memory config、mesh mapping 等设备语义。模型目录中的 generator 将 Hugging Face config/weights 转换为 TT model args、KV state、prefill/decode entrypoints 和 sampling capability。

## ABI 契约

| 边界 | 输入 | 输出/状态 | 常见失配 |
|---|---|---|---|
| PyTorch/HF→generator | config、state dict、token ids | TT weights/model args | 名称、shape、量化格式 |
| generator→TTNN | tensor layout、mesh mapper、op config | device tensor/event | shard/layout 不一致 |
| TTNN→TT-Metal | op/program config | program、buffers、kernels | CB/SRAM/core-grid 不满足 |
| runtime→device | queues、trace、fabric config | completion/data | firmware/runtime tuple 漂移 |

## ModelGenerator 为何是接入核心

vLLM 只知道 request、token budget、KV blocks 和 sampling semantics；它不知道 TT 模型如何为不同 batch/context 捕获 trace、如何布置 KV、怎样读取 decode output。因此 generator 必须提供比普通 `nn.Module.forward` 更宽的协议：prefill/decode 分离、warmup/capture、KV page table 更新、可选 device sampling、异步 submit/read，以及能力声明。

固定 tt-metal revision 中可见 `models/tt_transformers/tt/generator.py` 的 decode/device-sampling 路径，以及 `models/common/llm_runtime/trace_compiler.py::TraceCapturePlan` 等 trace 组织。具体模型也可有自己的 `generator_vllm.py`。这说明“模型支持”是模型实现、plugin adapter 和 release config 的联合属性。

## 生命周期与资源 owner

- device/mesh：runtime 打开，worker 生命周期关闭。
- weights：model loader/generator 创建并常驻。
- KV：vLLM 管理逻辑 request/block，generator 持有实际 device tensor/layout。
- trace：模型/runtime 捕获，必须绑定 shape、mesh、地址与配置。
- sampling state：可能在 host vLLM，也可能在 model/device；不能双写 RNG 或 token state。

## 移植一个模型的最小问题集

1. 权重转换是否可重复并有数值 golden？
2. prefill/decode 支持哪些 shape bucket？
3. KV cache 的 physical layout 与 update API 是什么？
4. logits 在何处产生、何处采样、RNG 谁持有？
5. trace 捕获的地址与生命周期条件是什么？
6. mesh/firmware/tt-metal/plugin/vLLM 的匹配 tuple 是什么？

## 结论分类

- `SOURCE_IMPLEMENTED`：固定 tt-metal revision 的 TTNN、model generator 与 trace compiler 层存在。
- `INFERENCE`：generator 是 vLLM semantic contract 与 TT device contract 的主要阻抗匹配层。
- `RECOMMENDATION`：兼容表应以“model × mesh × mode × tuple”为主键，不能只有 model 名称。
- `UNKNOWN`：未在硬件上验证固定 tuple 的 ABI 和数值。

