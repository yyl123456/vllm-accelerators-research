# QAIC SDK、Torch QAIC、QEfficient 软件栈

## 文档契约

- **唯一问题/Owner**：Cloud AI 从model到device执行的 compiler/runtime 层如何分工。
- **依赖输入**：110硬件；54 execution regimes；53 artifact identity。
- **唯一输出**：JIT/AoT、Apps/Platform SDK、QPC/QEfficient责任链。
- **Out of scope**：vLLM integration（112-116）。
- **权威来源**：[Cloud AI SDK](https://quic.github.io/cloud-ai-sdk-pages/1.19.10/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/index.html)、[qaic-runner](https://quic.github.io/cloud-ai-sdk-pages/1.21/Getting-Started/SDK-Tools/shared/qaic_runner.html)。

Platform SDK提供driver/runtime/device管理；Apps SDK提供compiler、tools、framework integrations与model workflow。官方文档区分JIT和AoT：JIT要求Apps+Platform SDK同机，compile/execution耦合；AoT可在build机产生compiled network，部署机只需Platform SDK。

通用workflow是model export/prepare→compile→QPC(Qaic Program Container)→activate/run。QEfficient/相关模型转换层负责将HF/PyTorch模型改写、量化并生成适合Cloud AI的图/配置；Torch QAIC/QAIC Python runtime为PyTorch式执行提供桥梁。具体package命名/版本需以plugin requirements为准。

AoT对batch/context/profile形成静态契约。官方SDK建议大模型编译主机至少数百GB RAM与大存储，说明artifact生成本身是独立infra工作负载。QPC不是只由model name标识：compiler/SDK、SKU、quant、profiles、cores/devices与模型revision都进key。

runtime activation将QPC映射到device/physical channels；`qaic-runner`能查询/执行precompiled network并输出host/submit/device stats，是device smoke而非vLLM语义证明。

completion需区分submit、device inference完成和output可读；timeout/SoC reset会使active handles失效。`VENDOR_CLAIM × UNVALIDATED` 支持上述 SDK/JIT/AoT/QPC 职责；plugin 具体调用为 `SOURCE_IMPLEMENTED × STATIC_REVIEWED`；本轮 device ABI 与性能为 `UNKNOWN`。标签定义见 02。

## 对象 ownership

| 对象 | 创建层 | 失效条件 |
|---|---|---|
| compiled QPC | Apps SDK/QEfficient | target/profile/toolchain变更 |
| device/session | Platform SDK/plugin | reset/driver/runtime fault |
| activation/exec object | session/runtime | completion/release/epoch变化 |
| host bindings | runner/session | shape/dtype/QPC schema变化 |

固定插件的 `model_loader/qaic_session_np.py::QAICInferenceSession` 与 `model_loader/qaic.py::QaicCausalLM` 消费这些对象。`RECOMMENDATION`：启动校验 artifact metadata 与 runtime tuple；`UNKNOWN`：SDK 对 cancel/reset 后 handle 的精确保证。
