# vllm_torchtpu && xla

> 来源：[飞书知识库原文](https://terapines.feishu.cn/wiki/Iv9xw6wLpiCuj1k74QAc4maLnuq)  
> 迁移版本：revision 9；飞书最后更新时间：2026-09-04T02:26:54Z  
> 本地同步日期：2026-09-08

```text
┌──────────────────────────────────────────────┐
│ 项目：vLLM                                   │
│ 仓库：https://github.com/vllm-project/vllm   │
│ 状态：公开开源                               │
│                                              │
│ EngineCore / Executor                        │
│ → 加载 vllm.platform_plugins                 │
│ → 执行标准 PyTorch vLLM 模型                 │
└──────────────────────┬───────────────────────┘
                       │ vLLM 平台插件接口
                       ▼
┌────────────────────────────────────────────────────┐
│ 项目：vllm-torchtpu                                │
│ 可见仓库：rushabh-46/torchtpu-vllm-old             │
│ 状态：个人公开旧仓库；Google 正式仓库仍是私有仓库  │
│                                                    │
│ register_tpu_platform()                            │
│ → TpuPlatform                                      │
│     dispatch_key = "PrivateUse1"                   │
│     simple_compile_backend = "tpu"                 │
│ → TPUModelRunner                                   │
│                                                    │
│ 两种编译入口：                                     │
│                                                    │
│ ① vLLM 分段编译                                    │
│    TpuCompilerAdaptor.compile(FX Graph, inputs)     │
│    → TpuBackend(graph, inputs)                     │
│                                                    │
│ ② 显式编译函数                                     │
│    @torch.compile(backend="tpu", fullgraph=True)    │
└──────────────────────┬─────────────────────────────┘
                       │ Python API
                       │ torch.compile(...)
                       │ torch.fx.GraphModule
                       ▼
┌────────────────────────────────────────────────────┐
│ 项目：PyTorch                                      │
│ 仓库：https://github.com/pytorch/pytorch            │
│ 状态：公开开源                                     │
│                                                    │
│ torch.compile                                      │
│ → TorchDynamo 捕获 Python/PyTorch 程序             │
│ → 生成 torch.fx.GraphModule                        │
│ → 根据 backend="tpu" 查找已注册的外部编译后端      │
│ → 回调 TorchTPU 的 TpuBackend(FX Graph, inputs)    │
│                                                    │
│ 注意：不经过 TorchInductor                         │
└──────────────────────┬─────────────────────────────┘
                       │ torch.compile backend API
                       │ FX Graph + example_inputs
                       ▼
┌────────────────────────────────────────────────────┐
│ 项目：TorchTPU / torch-tpu                         │
│ 官方仓库：尚未公开                                 │
│ 状态：Google 私有源码；以独立 Python 包提供         │
│                                                    │
│ 当前公开可见的符号：                               │
│ torch_tpu._internal.compile                        │
│ torch_tpu._internal.compile._backend.TpuBackend    │
│                                                    │
│ 【官方文档确认】                                   │
│ AOTAutograd / decomposition                        │
│ → PyTorch operators 映射为 StableHLO               │
│ → 调用 XLA 编译                                    │
│                                                    │
│ 【无法从源码确认】                                 │
│ · backend 注册函数                                 │
│ · TpuBackend.__call__ 的内部实现                   │
│ · FX→StableHLO 的具体函数和 Pass                   │
│ · 调用 XLA/PJRT 的具体函数                         │
│ · executable 的创建和执行函数                     │
└──────────────────────┬─────────────────────────────┘
                       │ StableHLO/编译请求
                       │ 具体内部 API 未开源
                       ▼
┌────────────────────────────────────────────────────┐
│ 项目：StableHLO                                    │
│ 仓库：https://github.com/openxla/stablehlo         │
│ 状态：公开开源                                     │
│                                                    │
│ XLA 的输入 IR / 算子语义                           │
│ 注意：StableHLO 是 IR，不是编译器                  │
└──────────────────────┬─────────────────────────────┘
                       │ StableHLO Module
                       ▼
┌────────────────────────────────────────────────────┐
│ 项目：OpenXLA / XLA                                │
│ 仓库：https://github.com/openxla/xla               │
│ 状态：公开开源                                     │
│                                                    │
│ XLA 编译器                                         │
│ → StableHLO/HLO 优化                               │
│ → TPU lowering                                     │
│ → 生成 TPU executable                             │
│                                                    │
│ 提供 PJRT C/C++ 接口：                             │
│ PJRT_Client_Compile                                │
│ PJRT_LoadedExecutable_Execute                      │
│                                                    │
│ 【无法确认】TorchTPU 是否直接调用这些具体入口      │
└──────────────────────┬─────────────────────────────┘
                       │ PJRT/runtime 接口
                       │ TorchTPU 的具体调用位置未开源
                       ▼
┌────────────────────────────────────────────────────┐
│ 项目/组件：libtpu                                  │
│ 分发：https://pypi.org/project/libtpu/              │
│ 状态：二进制发布；核心源码未公开                   │
│                                                    │
│ TPU 编译/runtime/设备通信实现                      │
│ → 加载 executable                                 │
│ → 提交 TPU 执行                                    │
└──────────────────────┬─────────────────────────────┘
                       │ 私有 Driver 接口
                       ▼
┌────────────────────────────────────────────────────┐
│ TPU Driver / Firmware / Hardware                   │
│ 状态：未开源                                       │
└────────────────────────────────────────────────────┘
```
