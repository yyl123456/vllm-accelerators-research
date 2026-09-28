# vLLM Ascend Plugin 激活与 Platform

## 文档契约

- **唯一问题/Owner**：`vllm-ascend` 如何被 vLLM 发现并把全局 platform 策略切到 NPU。
- **依赖输入**：70/71 plugin与Platform contract；101 software stack。
- **唯一输出**：entry point→`NPUPlatform`→worker/backend/config hook链。
- **Out of scope**：runner/attention（103）、ACLGraph（104）、patch ledger（105）。
- **源码基线**：vllm-ascend `3546357838389aa7201d9b44e234f831e98b2fd4`。

## 1. 激活链

`setup.py` 注册 `vllm.platform_plugins: ascend = vllm_ascend:register`。vLLM加载entry point并调用register，最终选择 `vllm_ascend.platform.NPUPlatform`。验证必须逐级确认distribution metadata、entry point、factory、selected class和每进程初始化。

## 2. Platform responsibility

`NPUPlatform(Platform)`设置device/dispatch、HCCL distributed backend、supported dtype/quantization、attention/backend选择、worker class、compile/pass hooks、configuration validation与环境变量。A2/A3、310P、950等分支不能由单一“NPU”标签覆盖。

platform validation还会结合`AscendConfig`及model/features选择scheduler、runner v1/v2、graph和kernel路径。默认继承上游Platform方法需逐版本审计；未override不代表默认语义适合NPU。

## 3. Import side effects

plugin含大量patch/registration。由于platform/general plugins会在多个进程加载，初始化必须幂等并避免在frontend过早创建设备context。环境变量和patch应用顺序是effective behavior的一部分，应进入启动日志和tuple。

## 4. 选择不等于支持

成功得到`NPUPlatform`只证明policy入口。仍需分别验证NPUWorker、ModelRunner、attention/custom ops、HCCL、graph、model/quant/features。310P存在专门worker/runner/attention/sampler目录，说明不同产品路径不可外推。

## 5. 验收

无设备可做package/entry-point/static class与patch inventory；有设备需npu-smi、device init、single/multi-rank、config negative tests。冲突插件、错误CANN、unsupportedSoC/feature必须fail closed。

## 6. 证据标签

- `SOURCE_IMPLEMENTED`：entry point与`NPUPlatform`来自固定SHA。
- `INFERENCE`：多进程幂等/早期device-context是调用时序风险。
- `UNKNOWN`：该SHA与当前网页v0.23.0是否为同release，不作推断。

