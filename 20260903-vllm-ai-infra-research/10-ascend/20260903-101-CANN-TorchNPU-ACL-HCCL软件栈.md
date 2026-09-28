# CANN、TorchNPU、ACL 与 HCCL 软件栈

## 文档契约

- **唯一问题/Owner**：从 PyTorch op 到 Ascend device task/collective 的软件层分别负责什么。
- **依赖输入**：100 的硬件边界；54 的 execution regime；60 的process groups。
- **唯一输出**：runtime/compiler/communication责任链与版本tuple。
- **Out of scope**：vLLM plugin code（102-105）。
- **权威来源**：[vLLM Ascend 安装与分层栈](https://docs.vllm.ai/projects/ascend/en/main/getting_started/installation.html)、华为 CANN 文档。

## 1. 层次

PyTorch 提供frontend/tensor语义；`torch_npu` 注册NPU device、operators、streams/events和distributed适配；CANN提供operator libraries/compiler/runtime；ACL是host侧runtime/operator调用接口之一；HCCL承担collectives。ATB/自定义ops/Triton Ascend等处于更高性能library/codegen层。

vLLM Ascend不是绕过这些层直接控制AI Core：它通过TorchNPU tensors与CANN/ACL ops，部分路径以C++ extension直接链接 `torch_npu`、`ascendcl`、tiling/opapi等；通信可走torch.distributed HCCL或plugin的PyHccl wrapper。

## 2. compile/run路径

Eager op由dispatcher落到NPU implementation，可能在runtime完成tiling/task enqueue；Triton Ascend提供另一kernel生成路径；ACLGraph捕获/回放稳定task序列；模型级图或ATB路径又有不同artifact/profile。它们可以混合，不应统称“CANN编译”。

## 3. 版本耦合

driver/firmware、CANN、TorchNPU/PyTorch、Triton Ascend、plugin/vLLM需按发布矩阵成套。官方安装页明确警告不要任意混合不同release的版本，并按A2/A3/310P/950DT区分images/build。当前网页版本会漂移；本文源码分析固定plugin SHA，不把网页main版matrix冒充该SHA验证tuple。

## 4. completion/error

Torch host API可只enqueue；ACL stream/event定义device completion；HCCL还需collective peers完成。异步device error可能在后续sync暴露。vLLM Future必须映射这些层，不能将Python return作为KV reuse fence。

## 5. 可观测与验证

保留 `npu-smi`、driver/firmware、CANN、Torch/TorchNPU、plugin wheel device type、loaded shared libraries与SoC；分别做op、stream/event、collective、graph、model smoke。只import `torch_npu`不证明CANN custom ops或HCCL拓扑可用。

## 6. 证据结论

- `VENDOR_CLAIM × UNVALIDATED`：150 的 `WEB-ASCEND-01` 列出分层栈与 release 配套要求；具体 tag 矩阵才属于 `RELEASE_CONTRACT`。
- `SOURCE_IMPLEMENTED`：固定plugin链接并调用ACL/HCCL/TorchNPU路径。
- `UNKNOWN`：本轮无NPU，ABI/设备执行/性能不作已验证声称。
