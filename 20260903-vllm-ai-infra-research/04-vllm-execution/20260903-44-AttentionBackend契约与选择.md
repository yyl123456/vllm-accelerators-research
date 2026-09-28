# AttentionBackend 契约与选择

## 文档契约

- **唯一问题**：给定 model、KV spec、dtype、device 和 feature，如何选出语义兼容的 attention implementation。
- **In scope**：backend capabilities、selector、metadata builder、prefill/decode differences、fallback。
- **Out of scope**：PagedAttention 原理（30）、通用 custom ops（45）、vendor backend 细节（10-13 组）。
- **依赖输入**：31-34 的 KV spec/layout；71 的平台 capability；43 的模型层类型。
- **唯一输出/Owner**：attention implementation 的约束求交、metadata 与数值 ABI。
- **相邻篇不得重述**：45 只拥有非 attention runtime op；51 只拥有 compiler rewrite。
- **证据基线**：vLLM `bb363db9...` 的 `v1/attention/backends/`、registry/selector、platform CUDA/ROCm implementations。

## 1. backend 是多接口组件

Attention backend 不只是 forward kernel。它通常定义支持的 KV spec/layout、metadata type/builder、state shape、kernel implementation、cascade/prefix/quant/MLA/SWA 等能力，并参与 connector 对 layer state 的解释。

## 2. selection 是约束求交

候选集合需同时满足 device/platform、compute capability、dtype/quant、head size、attention type、block size/layout、prefix cache、sliding window、spec decode、graph/compile、distributed mode、encoder/cross attention。平台可给 priority，但 priority 只能在兼容候选内排序。

用户强制 backend 时，若约束不满足应列出 invalid reasons 并拒绝；静默 fallback 会使性能/数值/feature 声明不可审计。

## 3. prefill 与 decode

prefill 的 query length 大、可用矩阵化；decode query 很短、访问长 KV。backend 可能采用不同 kernels、metadata 和 workspace。chunked prefill 在同一 batch 混合阶段，要求 metadata 能表达每 sequence 的 query/context boundary，而不是全 batch 二元模式。

## 4. ABI 不变量

- attention output 与 reference 在声明 tolerance 内；
- KV write/read layout 与 `KVCacheSpec` 一致；
- causal/window/sink/cross mask 正确；
- GQA/MLA head mapping 正确；
- variable lengths、page table、partial blocks 正确；
- graph replay 不持有过期 metadata address；
- TP/DCP/PCP 的 local ranges 与 collective contract 一致。

## 5. 固定源码接口与选择结果

`vllm/platforms/interface.py::Platform.get_attn_backend_cls()` 是平台约束入口；`vllm/v1/attention/backends/registry.py::AttentionBackendEnum` 解析实现；`vllm/v1/attention/backends/abstract.py` 定义 backend/metadata contract；model runner 将 KV cache、block table、query metadata 交给 layer/backend。

选择 key 至少包含 attention type（MHA/GQA/MLA/混合）、head size、dtype/KV dtype、block/page size、sliding window、quant、device/platform、compile/graph 和并行方式。平台强制替换用户选择时必须记录 desired/effective backend。

失败需区分：startup 不兼容、shape/bucket miss、kernel runtime failure、数值不一致和 performance fallback。`SOURCE_IMPLEMENTED`：上述接口；`INFERENCE`：backend selection 是 layout ABI negotiation；`RECOMMENDATION`：启动 artifact 保存完整 predicate；`UNKNOWN`：vendor kernel 的长上下文数值/性能。

## 6. 评价方法

按 prefill/decode、batch/context/head/dtype 分别做 correctness 和 roofline；记录 kernel choice 而非只记录配置名；测 launch、workspace、compile time 和 graph compatibility。单一模型 E2E 成功不能覆盖 selector 的负路径。

**架构判断**：attention backend 是 KV ABI 与 hardware kernel 的交界。第三方接入最危险的做法是复用一个 backend 名称，却改变它对 layout、completion 或 quant metadata 的解释。
