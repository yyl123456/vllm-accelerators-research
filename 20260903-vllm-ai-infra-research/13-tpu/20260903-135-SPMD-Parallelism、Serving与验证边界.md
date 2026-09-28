# SPMD Parallelism、Serving 与验证边界

## 文档契约

- owner：TPU mesh sharding、多 host executor、Pathways/DP/TP/PP/CP 与发布验证。
- 依赖输入：`60`–`66`、`130`–`134`。
- 唯一输出：TPU 并行组合决策和证据等级。
- 不重复：不重述 XLA 分层、runner 或 compile key。

## SPMD 不是“一个 Python 进程等于一个 device”

JAX `Mesh`、`NamedSharding` 与 `PartitionSpec` 可以让单个 host process 控制多个 local/global devices；vLLM 也可使用 multiprocess DP、PP multiproc 或 multi-host Ray executor。固定 platform 的 `_resolve_multiprocess_dp()` 明确区分 online multiprocess DP 与 single-process SPMD DP，并禁止 multiprocess DP 和 attention-DP/Pathways 的某些组合。

## 并行轴映射

| 轴 | 分割对象 | 主要收益 | 主要代价 |
|---|---|---|---|
| TP/model axis | weights/hidden/heads | 模型容量与矩阵并行 | 每层 collective |
| DP/data axis | requests/batch | throughput | weights replication、路由 |
| attention DP | attention/KV data dimension | decode/attention扩展 | 专用 sharding/collective |
| PP | layers/stages | 跨 host 容量 | bubble、stage state/transfer |
| PCP/CP | prefill sequence/work | 长 prompt 并行 | partition/merge 与限制 |
| EP | experts | MoE 容量/计算 | all-to-all、负载不均 |

是否支持必须读取固定 revision 的 `ShardingConfigManager`、model strategy、kernel 与 tests；不能从 JAX 理论表达能力推断 vLLM 组合已实现。

## Pathways 与 direct PJRT

Pathways proxy 使用不同的初始化、拓扑和 process constraints；固定源码要求 proxy 模式禁用 vLLM V1 multiprocessing，并提前完成全局 platform resolution。它不是 direct PJRT 的透明 transport。任何 Pathways 支持结论都必须单列 runtime/version/topology 和故障语义。

## 多 host 故障

collective participant、host process、PJRT client 和 vLLM rank 的 epoch 必须一致。任一 host 丢失通常使全局 executable/collective 失效；只重试一个 request 或一个 rank 可能死锁。恢复边界应是重建 communicator/client、重新加载/编译必要 executable、恢复或丢弃 KV lease，并由 router 停止向不健康 replica 发流量。

## 固定 repo 的证据资产

repo 包含 tensor/pipeline/data/expert/sequence parallel、async scheduler、hybrid KV、structured/spec decode、multimodal、offload、KV transfer 和 Ray executor 等测试，以及 release/nightly support matrices、`verified_commit_hashes.csv`。这些资产的存在只形成 `SOURCE_IMPLEMENTED × STATIC_REVIEWED`；只有取得目标 TPU、模型、shape、tuple 的 CI artifact 后，才可另列 `EXPERIMENT_RESULT × DEVICE_NUMERIC + STRESS_VALIDATED`。

## 发布门禁

1. 固定 vLLM 与 tpu-inference verified commit pair。
2. 固定 JAX/jaxlib/libtpu/TorchAX/Pathways（若用）和 TPU generation/topology。
3. host/source tests 与真实 TPU kernel numeric 分开报告。
4. 对 model×dtype/quant×parallel×feature×shape 做组合验证。
5. stress host loss、compile miss、abort、KV transfer、rolling restart。
6. 性能报告分解 TTFT/TPOT/goodput/compile/collective/HBM，并注明 cold/warm。

## 结论分类

- `SOURCE_IMPLEMENTED × STATIC_REVIEWED`：SPMD sharding、multiple executors、Pathways guards 与大量测试矩阵入口。
- `INFERENCE`：多 host 恢复通常必须提升到 replica/engine epoch。
- `RECOMMENDATION`：support matrix 的每格分别标注 02 定义的事实来源类型与验证成熟度，不使用 SOURCE/HOST/DEVICE 等未定轴缩写。
- `UNKNOWN`：本轮没有 TPU 硬件、CI provenance 和生产 SLO 证据，不能声称 production-ready。
