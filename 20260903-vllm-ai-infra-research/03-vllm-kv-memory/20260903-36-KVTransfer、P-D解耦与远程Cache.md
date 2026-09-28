# KV Transfer、P-D 解耦与远程 Cache

## 文档契约

- **唯一问题**：producer 计算的 KV state 如何被 consumer 可靠接管，并何时可宣告可执行/可释放。
- **In scope**：connector scheduler/worker split、metadata、load/save hooks、P/D handshake、ownership transaction、lease、故障。
- **Out of scope**：本地 hash（33）、通用 distributed parallelism（60-66）、具体网络硬件（93）。
- **证据基线**：vLLM `bb363db9...` 的 `distributed/kv_transfer/`、`v1/worker/kv_connector_model_runner_mixin.py`、NIXL lease design；DistServe/Sarathi 仅作系统背景。

## 1. P/D 解耦为何存在

prefill 常为大矩阵、算力密集且决定 TTFT；decode 每步 token 少、反复读取 weights/KV，常为 memory/launch/collective 受限并决定 ITL。把两者放在不同 instance 可独立配置 batch、parallelism 与资源比例，减少互扰；代价是必须搬运 KV、维护路由 affinity，并把一次请求扩展为分布式事务。

只有当减少的排队/干扰收益大于 transfer、handshake、额外排队和失败恢复成本时，解耦才提升 goodput。

## 2. connector 的双面合同

V1 connector 通常有 scheduler-side 和 worker-side：

- scheduler side 决定外部可命中 tokens、构造每 step metadata、跟踪 request finish；
- worker side 注册真实 KV tensors，在 forward 前 `start_load_kv`，按 layer 等待/装载，在计算后 `save_kv_layer`，并在适当边界 `wait_for_save`；
- factory/role 将同一 connector 以 SCHEDULER 或 WORKER 身份实例化。

metadata 是 control-plane intent，DMA/RDMA 是 data-plane completion。收到 metadata 不等于 bytes ready；copy API return 也未必等于 remote visibility。

## 3. ownership transaction

一次安全 handoff 至少经历：

1. producer 为 request/version 冻结可迁移的 finalized groups/blocks；
2. 发布包含 model/layout/dtype/group/parallel mapping 的 descriptor；
3. consumer 预留 destination blocks，禁止被 allocator 复用；
4. 传输并校验所有必要 groups；
5. consumer 原子标记 KV-ready，scheduler 才允许依赖这些 tokens 执行；
6. producer 收到 commit/lease 终止条件后释放 source；
7. 任一步失败都能 abort、超时并回收双方资源。

若第 5 步逐 layer 暴露而 kernel 又读取未完成 layer，会产生 partial-read；若第 6 步过早，RDMA source 被复用；过晚则泄漏容量。

## 4. local hit 与 remote hit 合并

consumer 可能已有本地 prefix，同时 remote 提供后缀。必须计算 per-group overlap、缺口和共同 boundary，避免重复 allocation 或覆盖本地更优 state。固定 revision 的 allocation 参数显式区分 local new-computed 与 external-computed tokens，并可 `delay_cache_blocks`，因为 transfer 在未来 step 才完成。

hybrid groups 进一步要求 remote 无条件提供某些递推 state，不能只比较 token count。transfer group subset 也意味着未传 groups 必须能本地重建或明确不参与 target forward。

## 5. parallel mapping

P 与 D 的 TP degree 可能不同。state 要按 head/layer/token shard 进行 gather、scatter 或多 peer 读取；映射必须处理 GQA/MLA、DCP/PCP 与 layout。网络 bytes 可能不变，但连接数、small transfers、registration 和 synchronization 会放大。

因此 compatibility key 至少含：model revision、KV spec digest、quant mode/scales、layout、block size、P/D TP/PP/DCP/PCP、connector version、endianness/device ABI。只匹配 model name 不够。

## 6. lease 与 failure semantics

producer 不应无限持有等待 D 的 blocks。NIXL design 采用 per-request lease/heartbeat：consumer 在等待/传输期间续租，transfer 完成或请求结束后停止；consumer crash 后 lease 到期回收。heartbeat 在 forward loop 中处理意味着长 forward 会延迟心跳，duration 必须覆盖该 tail，而不是只看平均值。

必须定义：router 重试是否复用 request/version；重复 consumer 谁获 ownership；producer crash 如何回退重算；网络 partition 时是 fail closed 还是等待；cancel 如何传播；late completion 如何识别 destination generation，避免写入重用 block。

## 7. 性能预算

端到端 handoff：

`T_handoff = T_route + T_metadata + T_register/handshake + T_queue + T_copy + T_visibility + T_commit`

应按请求 bytes、并发 transfers、拓扑和 TP mapping 测 p50/p99，并拆分等待与传输。理论 link bandwidth 不能替代 effective bandwidth；尤其是小 block、跨 NUMA、RDMA registration miss 和 compute/communication contention。

## 8. 验收矩阵

- 正常：0/local-only/remote-only/mixed hit，所有 group 完整；
- 边界：partial final block、不同 TP、quantized KV、hybrid/MTP；
- 竞态：cancel during load、preempt during save、duplicate/reordered metadata；
- 故障：producer/consumer/router crash、partition、timeout、late DMA；
- 资源：lease expiry、destination rollback、source double-free 防护；
- 数值：transfer 后继续 decode 与同实例 baseline token-by-token 一致或满足声明 tolerance。

## 9. 架构结论

P/D 不是一个“打开 connector”的性能开关，而是 KV ownership 的分布式提交协议。**事实**：固定 revision 暴露 scheduler/worker hooks、多个 connectors、external-token allocation 与 NIXL lease。**UNKNOWN**：任一组合在特定 NIC、topology、vendor backend 上的 correctness 和收益，必须以完整 release tuple 和故障注入证明。

