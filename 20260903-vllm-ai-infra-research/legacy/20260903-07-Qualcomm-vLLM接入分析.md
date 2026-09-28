# Qualcomm Cloud AI vLLM 接入分析

## 范围

本章只讨论公开的 `qualcomm/vllm-qaic` 与 Cloud AI 100/80 数据中心加速卡。它不代表 Snapdragon、Hexagon/HTP、QNN 移动端或整个 Qualcomm AI 生态都可运行 vLLM。

## 核心架构

`vllm-qaic` 提供两种互斥执行模式：

| 模式 | 模型路径 | 执行 |
|---|---|---|
| Eager/PYT | 选定的 vLLM/PyTorch 模型 | `torch-qaic`、eager/compile |
| AoT | Hugging Face → QEfficient export/quantize/compile | QPC artifact + QAIC device session |

`SOURCE_IMPLEMENTED`：plugin 注册 `QaicPlatform` 和 QAIC KV connector；根据安装环境选择 `QaicWorkerPyt` 或 `QaicWorkerAoT`。两个 model runner 都继承 `GPUModelRunner`，但分别把执行映射到动态 PyTorch 或静态 QPC session。

## 硬件特性

Qualcomm 官方页面给出的 AI 100 Ultra 规格为 128GB LPDDR4x、548GB/s、576MB SRAM、PCIe Gen4 x16、150W，以及最高 870 TOPS INT8/288 TFLOPS FP16。[产品页](https://www.qualcomm.com/data-center/products/cloud-ai-100-ultra)

这些是 `VENDOR_CLAIM`。架构含义是：

- 大容量有利于单卡容纳较大模型和 KV；
- LPDDR 带宽明显不能只用 GPU HBM 经验解释，decode 需依赖量化、片上复用、并发和编译优化；
- 大 SRAM 可承载中间工作集，但需要 compiler/session 明确规划；
- PCIe attached 形态使 host/device 传输和多卡 collective 成为显式系统成本。

## AoT 的真实部署单元

AoT 路径不是“安装 plugin 后动态加载任意模型”。它需要 QEfficient 完成 export、量化和编译，生成 QPC，再由 session 加载。

artifact provenance 应至少包含：

- 模型和权重 revision；
- tokenizer/config；
- 量化方法与权重/KV dtype；
- batch、compiled context lengths、prefill/decode shape；
- device SKU、卡数与并行拓扑；
- QAIC Apps SDK/compiler/runtime；
- vLLM 与 plugin revision。

`UNKNOWN`：本次没有证明仓库的 cache key 已覆盖以上全部字段。生产系统应保存独立 manifest，而不是只依赖文件名。

## 动态调度的折衷

AoT 通过 compiled context lengths、padding/bucket 将 continuous batching 映射到有限 shape。bucket 少会增加 padding；bucket 多会增加编译时间、artifact 数量和内存压力。compile miss 不应在未知成本下阻塞线上请求。

Eager 模式动态性较好，但当前功能面较窄。仓库 feature matrix 将 prefix caching、MLA、async output 列为未实现或计划项；AoT 支持更多 speculative decoding、LoRA、disaggregated serving、multimodal、embedding 和 PP 组合。

## KV、attention 与并行

- Platform 选择 QAIC-specific attention/worker；
- AoT 支持硬件原生 `mxfp6` 和 `mxint8` KV 等能力，Eager 并非等价支持；
- TP 通过多个 QID/device 执行，PP 和 P/D 等高级能力主要集中于 AoT；
- plugin 有 KV connector，但 connector 存在不等于每种 LoRA、quantization、multimodal 组合都兼容。

## 版本约束

仓库 README 明示 main/release/development branch 对 vLLM 版本不同；当前材料中 v0.15 与 v0.23 路线并存。AoT requirements 还记录 compressed-tensors 与 torch/vLLM 的版本冲突。部署必须选择一条完整路线，不能混装两个模式的依赖。

## 主要难点

1. 把动态请求映射到静态 QPC shape；
2. 管理 artifact 与 SDK/设备的严格 tuple；
3. 在大容量但相对有限的外存带宽下优化 decode；
4. 保持 vLLM request/KV 生命周期与 session 内部 slot 一致；
5. 明确 Eager 与 AoT feature 差异，避免用总功能表误导；
6. 证明取消、抢占、QPC/session failure 后的 slot 和 KV 回收。

## 证据结论

- 双模式、worker/runner、plugin registration：`SOURCE_IMPLEMENTED`。
- 仓库 feature matrix 与支持模型表：`RELEASE_CONTRACT`，仍需绑定具体 branch/release。
- Qualcomm 规格：`VENDOR_CLAIM`。
- runtime correctness、artifact cache 完整性和性能：`UNKNOWN`。

