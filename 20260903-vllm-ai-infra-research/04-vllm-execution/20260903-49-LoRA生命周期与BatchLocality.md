# LoRA 生命周期与 Batch Locality

## 文档契约

- **唯一问题/Owner**：adapter 从注册、驻留、batch 映射到驱逐的权重状态机。
- **依赖输入**：43 的 base model/weight contract；42 的 request→batch row mapping。
- **唯一输出**：LoRA identity、residency、kernel mapping 与 cache compatibility contract。
- **相邻篇不得重述**：grammar 状态归 49A；多特性交叉约束归 49B；租户公平归 14。
- **证据基线**：vLLM `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` 的 `vllm/lora/`、`v1/worker/lora_model_runner_mixin.py`、GPU LoRA utils。

## 1. 状态与 Owner

adapter 的逻辑状态可建模为 `KNOWN → HOST_LOADED → DEVICE_RESIDENT → BATCH_REFERENCED → EVICTABLE → EVICTED`。管理器拥有 adapter ID→module/slot 映射；runner 拥有当步 batch row→adapter slot 映射；kernel 只消费已冻结的 mapping。request 不能直接持有可复用 device slot 作为永久身份。

稳定 identity 至少绑定 adapter content/revision、base model revision、rank 与 target module 集合、dtype/quant 格式。名称或路径不是内容 identity。

## 2. 加载与驱逐事务

加载需验证 target layers、rank/shape、scaling，再分配 host/device buffers 并原子发布。部分 layer 加载失败不能标 resident。驱逐只能选择无 batch 引用且无 in-flight kernel 的 slot；slot 重新分配前需完成 device fence 并更新所有 mapping generation。

adapter churn 会把 IO/H2D 与 compile/packing 延迟泄露到 TTFT。locality-aware routing 可将请求送往已驻留 replica，但可能造成热点，policy 归 63/144。

## 3. Forward 与并行

batch 可混合 base 和多个 LoRA，kernel 依据 row/token mapping 应用低秩 delta。TP/PP/EP 下 adapter 参数需按 base layer placement 分片；遗漏某 rank/layer 会静默得到部分 adapter。fused kernel 的 max rank、数量、dtype/alignment 均是 capability 约束。

## 4. Cache/Artifact Identity

LoRA 改变 hidden/KV，因此 prefix KV、P/D transfer descriptor 和任何捕获有效权重的 compile/trace artifact 都必须绑定 adapter identity。若 runtime 在 kernel 参数中动态读取 adapter，graph 可复用性仍需证明 mapping/address 稳定。

## 5. 失败与验证

覆盖加载中 cancel、并发同 ID 不同 revision、evict 与 in-flight 交错、slot generation reuse、mixed batch、TP/PP layer coverage、prefix 隔离、worker restart。记录 load/evict/fallback 和 resident bytes。

## 6. 证据结论

- `SOURCE_IMPLEMENTED`：固定 revision 存在 LoRA manager 与 runner mixin 路径。
- `INFERENCE`：slot generation/fence 是异步安全所需合同，不代表所有实现已显式编码。
- `RECOMMENDATION`：以 content identity 和 reference-counted residency 做唯一真源。
- `UNKNOWN`：各第三方 backend 的混合 LoRA kernel、graph 与并行组合需设备证据。

