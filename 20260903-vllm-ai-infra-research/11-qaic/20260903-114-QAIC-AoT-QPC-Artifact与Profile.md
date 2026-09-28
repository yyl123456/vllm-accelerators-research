# QAIC AoT、QPC Artifact 与 Profile

## 文档契约

- **唯一问题/Owner**：AoT路径如何把动态serving请求约束到预编译QPC profiles。
- **依赖输入**：53 artifact key；54 AoT；111 SDK；112 mode selection。
- **唯一输出**：QPC identity、profile admission、activation与fallback合同。
- **Out of scope**：Pyt mode（113）、普通scheduler细节（115）。
- **源码基线**：vllm-qaic `3212cc...` 的 AoT worker/runner、model_loader QAIC session、requirements与CI marker。

AoT worker不在每step任意生成kernel，而是加载QPC/session并激活到device。artifact包含允许的batch/context/sequence/device/core配置；runner必须把`SchedulerOutput`映射到某profile，padding/packing后提交并unpack output。

identity至少包含model/weights transformation、QEfficient/toolchain/Apps SDK、QPC schema、SKU/ISA、quant/dtype、profiles、device partition、plugin/vLLM/runtime。shared QPC若漏任一因素会加载失败或静默错。

profile admission在scheduler/runner之前可预测最安全。超过max context/batch或unsupported feature应reject；切到Pyt/eager只有在同weights/KV/sampling语义且明确配置时才安全，默认不作透明fallback。

QPC activation消耗device resources/physical channels；多个profiles/replicas的resident策略影响启动和capacity。编译主机与部署机分离时需artifact签名、atomic publish和兼容probe。

验收覆盖profile边界、padding correctness、错误SDK/SKU/QPC、并发activation、device reset、rolling upgrade、cold/warm latency；与reference逐token比较。CI存在`qaic_aot_mode` marker只是测试分类，不等于本环境通过。

## 固定源码 profile 解析

`worker/worker.py::QaicWorkerAoT` 构造 `QaicModelRunnerAoT`；`model_loader/qaic.py::QaicCausalLM` 检查 QPC path，创建 `QAICInferenceSession`，读取 `binding_index_map`、`allowed_shapes`、KV info 与 decode K profiles，再调用 `np_run/complete_inf`。

profile key 至少覆盖 input binding names/dtypes/shapes、prefill length、decode K、logits outputs、KV handoff 与 multimodal resolution。`qaic_custom_mm_processor.py` 会将输入分辨率约束到 QPC 已编译集合。

`SOURCE_IMPLEMENTED`：上述 loader/session/profile symbols；`RECOMMENDATION`：admission 在提交前匹配 allowed shape；`UNKNOWN`：QPC metadata 是否足以自动校验全部 firmware/SKU tuple。
