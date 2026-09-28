# TT DeviceGroup、ReleaseModelSpec 与验证

## 文档契约

- owner：部署清单、版本 tuple、能力声明和分级验收。
- 依赖输入：`121`–`125` 的硬件/软件/runtime contract。
- 唯一输出：可发布的 TT backend 证据包格式。
- 不重复：不重述单个 kernel、scheduler 或 trace 机制。

## 名称澄清

本计划沿用了 `DeviceGroup`、`release_model_spec.json` 作为部署概念名，但在固定的 plugin 与 tt-metal revision 中没有找到名为 `release_model_spec.json` 的文件，也没有找到作为插件公共 ABI 的 `DeviceGroup` class。可见的是 `dp_discovery.py` 的 visible-device groups、TTNN mesh/distributed API、model class capability/spec hooks，以及外部 serving/deployment config 的注释引用。

因此本篇不伪造文件契约：`release model spec` 被定义为**应由发布系统持有的规范化 manifest**；其真实外部仓库/schema 在本轮证据中为 `UNKNOWN`。

## 建议的 manifest 主键

```text
hardware SKU + physical topology + logical mesh
+ firmware/runtime + tt-metal SHA
+ vllm-tt-plugin SHA + vLLM SHA
+ model implementation + weight artifact digest
+ serving mode + shape/capability profile
```

## 必要字段

| 类别 | 字段示例 | 为什么必须 |
|---|---|---|
| 硬件 | SKU、device IDs、mesh、fabric | 防止逻辑 grid 冒充物理可用性 |
| 软件 | driver/firmware/runtime/tt-metal/plugin/vLLM | ABI 与行为共同演进 |
| 模型 | arch、weights digest、generator path | 同名模型实现可不同 |
| 容量 | token pool、max length、max seqs、KV spec | 决定 admission 与 OOM 边界 |
| 能力 | prefix、chunked prefill、device sampling、async、multimodal | 不允许由平台名推断 |
| trace | mode、region、bucket/guard set | 捕获产物不是通用二进制 |
| 证据 | commands、artifacts、golden、perf environment | 将“声称”绑定可复现证据 |

## 验证阶梯

1. `SOURCE_IMPLEMENTED × STATIC_REVIEWED`：固定 SHA 可达，entry point、模型注册和 guards 静态可证。
2. `EXPERIMENT_RESULT × HOST_VALIDATED`：配置解析、capability matrix、scheduler/reload 单测通过；host stub 不算设备证据。
3. `EXPERIMENT_RESULT × DEVICE_SMOKE`：mesh open、load、一次 prefill/decode、abort/close。
4. `EXPERIMENT_RESULT × DEVICE_NUMERIC`：多 shape、多长度、多 sampling 与长 decode 对 golden；报告容差和 mismatch。
5. `EXPERIMENT_RESULT × STRESS_VALIDATED`：并发、preemption、abort、prefix/trace hit-miss、重复启动和故障注入。
6. `EXPERIMENT_RESULT × PERFORMANCE_MEASURED`：固定频率/拓扑/版本，分解 TTFT、TPOT、throughput、利用率和通信。
7. `RELEASE_CONTRACT × PRODUCTION_QUALIFIED`：manifest 与上述 artifacts 原子发布，可回滚且不会混搭版本行。

## 兼容/能力决策规则

- 未声明 capability 默认 `False`，不能乐观启用。
- 自动降级必须进入 effective-config artifact。
- host test PASS 只形成 `EXPERIMENT_RESULT × HOST_VALIDATED`，不得写成“支持 N300/T3K”。
- 单设备 smoke 不证明多 device fabric。
- 单模型/单 shape 数值 PASS 不证明 capability matrix。
- 性能数字没有完整 tuple 与 warmup/measurement protocol 时只能保留为 `VENDOR_CLAIM × UNVALIDATED` 或 `UNKNOWN`，不得形成可比较结果。

## 当前固定 revision 结论

- `SOURCE_IMPLEMENTED × STATIC_REVIEWED`：plugin entry points、TTPlatform/Worker/Runner、named mesh grids、per-model capabilities；host tests 的存在仍是静态事实，不代表其在本轮运行。
- `VENDOR_CLAIM × UNVALIDATED`：150 的 `WEB-TT-01..05` 描述 TT-Metalium/TTNN 软件层与硬件编程模型。
- `UNKNOWN`：本机 N300/T3K availability、固定 tuple 的 device/numeric/performance 结果，以及外部 release manifest 的权威 schema。
- `RECOMMENDATION`：发布门禁读取一份 schema-validated manifest，拒绝运行时临时拼接未经验证的 tuple。
