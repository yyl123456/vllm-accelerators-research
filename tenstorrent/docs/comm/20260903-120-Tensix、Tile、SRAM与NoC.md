# Tensix、Tile、SRAM 与 NoC

## 文档契约

- owner：Tenstorrent 单芯片计算与数据搬运模型。
- 依赖输入：`90` 的通用加速器数据通路术语。
- 唯一输出：把 Transformer 算子还原为 tile、core-local SRAM、NoC 与 DRAM 事务。
- 不讨论：产品 mesh、vLLM 调度、模型 release tuple；分别由 `121`、`124`、`126` 持有。

## 证据边界

- `SOURCE_IMPLEMENTED`：tt-metal `c634b1ca4c10eea5037d80767ce5f9e2912a401f`。
- `VENDOR_CLAIM × UNVALIDATED`：150 的 `WEB-TT-02..05` 是 TT-Metalium Tensix、memory、matmul lab 厂商资料；标签定义见 02。
- `UNKNOWN`：本机没有 Tenstorrent 设备，本文没有带宽、周期或端到端性能实测。

## 1. 计算不是“GPU kernel 的换皮”

Tensix core 同时包含矩阵 FPU、vector/SFPU、unpacker、packer、RISC-V data-movement processors 与本地 SRAM。典型程序拆成 reader、compute、writer 三类 kernel：reader 通过 NoC 将 DRAM tile 放入 circular buffer；compute 等待 tile、执行矩阵或向量运算；writer 将结果 tile 搬回 DRAM。并行来自不同 processor 与 core 的流式重叠，而不是依赖透明 cache。

官方示例的基础 tile 为 `32×32`。tile 是布局、搬运和计算共同使用的粒度，不等于 vLLM 的 token block。将两者混为“block”会导致错误的容量公式：token block 属于 KV allocator；tile 属于张量物理布局和 kernel contract。

## 2. 数据路径与背压

```text
DRAM --NoC DMA--> reader --push--> circular buffer
                                  | wait/front
                                  v
                             unpacker/FPU/SFPU
                                  | reserve/push
                                  v
DRAM <--NoC DMA-- writer <-- circular buffer
```

生产者必须先 reserve，写完再 push；消费者先 wait，读完再 pop。任一端 tile count、page size、address generator 或 core mapping 不一致，表现可能是停滞、越界、错误 tile 或静默数值错误。架构审查必须把“算子正确”拆成：地址正确、同步正确、layout 正确、数值正确、性能合理五类证据。

## 3. SRAM、DRAM 与 NoC 的架构影响

Wormhole Tensix core 的本地 SRAM 被官方文档描述为约 1.5 MiB。它同时承载 circular buffers、代码/运行时区域和中间值；可用空间不能直接按名义容量计算。DRAM 容量大但每次搬运需显式编排；NoC 决定跨 bank、跨 core 的吞吐与拥塞。

对 attention 而言，Q tile 通常复用而 K/V tile 流过计算阵列；decode 的小 M 使矩阵单元利用率和搬运固定成本更敏感。对 MLP 而言，权重分片、tile padding 与 core grid 决定是否能保持计算流水。对 elementwise/SFPU，若张量在 DRAM 间往返，融合收益可能大于算术优化。

## 4. 映射决策点

| 决策 | 主要约束 | 失败模式 |
|---|---|---|
| tile layout | shape、padding、dtype、op contract | 转置/分片语义错误 |
| core grid | 并行度、SRAM、NoC 路径 | 空闲 core 或热点 |
| CB 深度 | producer/consumer 速率与 SRAM | 背压或空间不足 |
| DRAM sharding | bank 并行与连续访问 | bank 冲突、低带宽 |
| 融合边界 | 中间值寿命与 kernel 支持 | 重编译爆炸或 DRAM 往返 |

## 5. 推理栈必须暴露的事实

模型层至少应向运行时声明支持的 mesh、最大 batch/context、KV layout、prefill/decode shape 集、sampling 位置与 trace 能力。运行时不能由“设备总内存”推导这些语义，因为 SRAM 布局、trace region、模型常驻权重和 per-model kernel 配置都参与约束。

## 结论分类

- `SOURCE_IMPLEMENTED`：tt-metal 中 TTNN/TT-Metal op 最终构造 program、kernel、circular buffer 与 data movement。
- `INFERENCE`：decode 对小 shape 和 host/device 边界更敏感，这是硬件数据流推论，不是本文性能测量。
- `RECOMMENDATION`：新 backend 的第一份硬件契约应是 tile/layout/memory-flow 表，而非笼统算子清单。
- `UNKNOWN`：固定 revision 在具体 SKU、频率、固件下的实测 roofline。

## 权威入口

- https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html
- https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.html
- https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/matmul/lab1/lab1.html
