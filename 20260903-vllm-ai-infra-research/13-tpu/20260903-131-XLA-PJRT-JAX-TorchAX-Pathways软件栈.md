# XLA、PJRT、JAX、TorchAX 与 Pathways 软件栈

## 文档契约

- owner：从模型前端到 TPU runtime 的编译/执行分层。
- 依赖输入：`130` 的硬件对象。
- 唯一输出：每层的 IR、ABI、状态和失配点。
- 不讨论：vLLM 的 worker 或 bucket policy；由 `132`、`134` 持有。

## 两条前端、一条 lowering 主线

```text
JAX-native model -----------+
                            +-> JAX tracing/jit -> StableHLO/XLA -> PJRT/libtpu
PyTorch vLLM model -> TorchAX+
                                                -> TPU executable/buffers
Pathways proxy ---------------------------------> remote/global runtime path
```

固定 `tpu-inference` revision 同时包含 `models/jax` 与 `models/vllm`。后者通过 TorchAX 将 PyTorch op/parameter 视图转换为 JAX 可追踪对象，最终仍由 `jax.jit`/XLA 编译；`TpuPlatform.simple_compile_backend='eager'` 的注释明确表示绕过 `torch.compile`，并不表示设备逐 op eager 执行。

## 各层责任

| 层 | 责任 | 典型失配时机 |
|---|---|---|
| model frontend | state、op 语义、mutable→functional 转换 | tracing/unsupported op |
| TorchAX | torch tensor/op 与 JAX view/dispatch | view、mutation、dtype、dispatch |
| JAX | pytree、jit、sharding、compile cache | trace/static arg/shape |
| XLA | StableHLO/HLO 优化、layout、collective、codegen | lowering/compile |
| PJRT/libtpu | device/client/buffer/executable/execute | load/submit/completion |
| Pathways | proxy/global topology 与远程执行协调 | initialization/topology/runtime |

PJRT 是 framework 与 compiler/runtime 的硬件无关接口；compile 产生 loaded executable，execute 操作 device buffers。它的 buffer/executable lifecycle 与 vLLM request/KV lifecycle 不是同一层，必须由 plugin 明确桥接。

## Import 与初始化顺序也是 ABI

固定源码 `tpu_inference/__init__.py` 在其他子模块导入 JAX 前尝试预载 Raiden 的 XLA extension，避免两个 XLA runtime 的 static initializer 冲突；当 `JAX_PLATFORMS` 含 `proxy` 时先调用 `pathwaysutils.initialize()`，并提前解析 vLLM current platform。说明“import 顺序”属于运行时正确性契约，而非风格问题。

## Functionalization 难点

JAX JIT 偏好纯函数、静态可推导结构；vLLM/PyTorch 模型常有 in-place mutation、动态 Python container、view 与 request-local state。固定 repo 的 model patcher、vision JIT manager、Qwen/Gemma patches 正是在修补这些阻抗。补丁存在不代表所有 PyTorch 模型自动兼容。

## 结论分类

- `SOURCE_IMPLEMENTED`：JAX-native、TorchAX model paths、Pathways proxy 初始化和 engine-first XLA preload。
- `VENDOR_CLAIM × UNVALIDATED`：150 的 `WEB-XLA-01/02` 定义 OpenXLA PJRT compile/executable/buffer 分层。
- `INFERENCE`：front-end 统一不等于 model semantics 或 performance 自动统一。
- `UNKNOWN`：当前 tuple 在 Pathways/Raiden 环境的实际初始化、故障和吞吐。

## 权威入口

- https://openxla.org/xla/pjrt/cpp_api_overview
- https://openxla.org/xla/pjrt/pjrt_integration
