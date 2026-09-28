# TPU MXU、HBM、ICI 与 Topology

## 文档契约

- owner：TPU 计算、存储和互联的硬件约束。
- 依赖输入：`90`–`95` 的通用硬件模型。
- 唯一输出：硬件特征如何约束 serving shape、sharding 与同步。
- 不讨论：XLA 软件栈、plugin runtime 或并行策略；由 `131`、`132`、`135` 持有。

## 官方硬件模型

TPU chip 包含一个或多个 TensorCore；TensorCore 由 MXU、vector unit 与 scalar unit 组成。MXU 使用 systolic array 承担矩阵乘，vector unit 适合 activation/softmax 等，scalar unit 处理控制与地址工作。官方资料指出 v6e/TPU7x 的 MXU 为 `256×256` MAC 阵列，早期版本通常为 `128×128`；具体 TensorCore/MXU 数量、HBM 和互联随代际变化。

这直接带来 shape 约束：大而规整的矩阵容易填满 MXU；decode 的小 batch/短 M、ragged token 与 padding 会降低有效利用率。vector/scalar 工作、HBM 搬运和 collective 不能用峰值 MXU FLOPS 解释。

## HBM 与 host 边界

权重、KV、activation、编译 workspace 和通信 buffer 竞争 HBM。host 与 TPU 之间存在 infeed/outfeed 或 PJRT buffer transfer/dispatch 语义；host 读取结果前需要 runtime completion。`HBM capacity × chips` 不能直接当作单模型可用内存：sharding、replication、reserved/runtime memory 以及不同 host 的局部可见性都会改变预算。

## Slice、Pod、ICI 与 DCN

- slice：同一 Pod 内通过 ICI 连接的一组 chips。
- topology：依代际可为 2D 或 3D，shape 不是简单设备计数。
- single-host/multi-host：决定控制、故障与 host-device placement 边界。
- multislice：slice 内仍走 ICI，slice 间通过 DCN；延迟与带宽域不同。

因此并行轴必须映射到拓扑：高频、细粒度 TP collective 更适合低延迟 ICI 邻接；跨 DCN 的同步会显著影响 decode。ICI resiliency 可绕过链路故障，但官方明确存在临时性能退化；“作业仍活着”不等于 SLO 不受影响。

## v5e 例子只用于说明代际差异

官方 v5e 页面列出每 chip 16 GB HBM、800 GiB/s HBM bandwidth、400 GB/s 双向 ICI、每 host 8 chips、2D torus。本文不把这些数字套到 v6e/v5p/TPU7x，也不据此估算固定 plugin revision 的性能。

## 架构决策表

| 现象 | 首查约束 | 不能直接归因 |
|---|---|---|
| decode TPOT 高 | small-M、collective、host sync | MXU 峰值不足 |
| compile shape 多 | ragged/padding/bucket | TPU 不支持动态请求 |
| HBM OOM | replication、KV、workspace、cache | 标称容量错误 |
| 多 host 抖动 | ICI/DCN、rank barrier、host skew | 单 kernel 慢 |

## 结论分类

- `VENDOR_CLAIM × UNVALIDATED`：150 的 `WEB-TPU-01/02` 给出 MXU/TensorCore、slice/topology、ICI/multislice 定义；标签见 02。
- `INFERENCE`：decode 更易受 padding、sync 与 memory traffic 支配。
- `RECOMMENDATION`：所有性能报告记录代际、slice shape、host count、sharding 和 ICI/DCN 域。
- `UNKNOWN`：本轮没有 TPU 设备上的固定 revision 性能或故障注入数据。

## 权威入口

- https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm
- https://docs.cloud.google.com/tpu/docs/v5e
