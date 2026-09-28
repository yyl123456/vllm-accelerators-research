# BlockTable 与物理 KV 布局

## 文档契约

- **唯一问题**：逻辑 blocks 如何落到 device allocation、stride 和 kernel metadata。
- **In scope**：`KVCacheTensor`、`KVCacheLayout`、worker `BlockTable`、slot mapping、一致性时序。
- **Out of scope**：分页动机（30）、逻辑 spec（31）、hash ownership（33）。
- **证据基线**：vLLM `bb363db9...` 的 `kv_cache_layout.py`、`kv_cache_interface.py`、`v1/worker/block_table.py`、`v1/worker/gpu/block_table.py`。

## 1. 物理合同不是一个 shape 字符串

当前统一逻辑五维轴为 `[L, B, H, N, C]`：layer、physical block、head、token slot、content。`KVCacheLayout` 用 stride permutation 表示 LBHNC、LBNHC、LHBNC、BLHNC、BLNHC、BHLNC 等排列。相同逻辑 axes 的不同 stride 会改变：

- 单 layer view 是否连续；
- 一个 page 是否为连续 byte run；
- block 是否最外层，适合按 block transfer；
- kernel coalescing 与 vectorization；
- connector 是否可直接 gather/scatter。

因此 layout 是 ABI，不能只作为 performance hint。

## 2. backing allocation 与 alias

`KVCacheTensor` 描述总 bytes、按 L 顺序的 layers、`layer_stride`、`block_stride` 与 offset。layer-compact layout 可让每层占连续区；block-outermost layout 可让同一 block 的多层 pages 聚合。多个 group 的 tensor address range 可能 overlay；正确性依赖“同一 block ID 在任一时刻只由一个 group ownership 使用”。

这种 alias 能减少 backing allocations，却将 allocator bug 放大为跨层/跨组污染。审计必须同时看 byte intervals 与 ownership timeline，不能只检查 tensor shape。

## 3. worker BlockTable 的职责

core 输出 request 的每组 block IDs；worker 维护 batch row，执行 add、append、swap/remove 等更新，并生成 attention backend 所需 block table 和 slot mapping。典型 slot：

`slot = physical_block_id * block_size + intra_block_offset`

对于混合 group、DCP/PCP 或特殊 layout，真实映射可能还有 shard/group 变换；上述公式只是基础层。

## 4. step 时序不变量

1. scheduler 先完成逻辑 allocation，再发布 scheduler output；
2. worker 在 launch 前更新对应 row/table；
3. kernel completion 之前，相关 block IDs 不能回到 free queue；
4. preemption/abort 的逻辑 free 必须与 device in-flight 生命周期隔离；
5. batch compaction 后 request row 与 positions 必须同步变化；
6. multi-rank 每个 rank 对相同 logical request 使用兼容 shard mapping。

异步执行最危险的是 core 已认为 step 完成并复用 block，而 device 仍读取旧地址。Future 的“结果可取”若不等于 stream/event completion，必须有额外 fence；此处仅定义需求，完成语义归 28。

## 5. layout 选择矩阵

| 需求 | 倾向 | 代价 |
|---|---|---|
| layer kernel 独立访问 | layer compact | block 级跨层传输可能分散 |
| P/D 按 block 搬运 | block compact/outermost | layer view stride 更复杂 |
| head-major kernel | H 更外层 | token contiguous 性可能下降 |
| page eviction/copy | page 连续 | 未必符合 compute tile |

不存在对所有 backend 最优的 layout。平台插件需宣告支持集合，并由 engine 选出 model、kernel、connector 的交集。默默 transpose 会引入额外 bandwidth 和 workspace，必须进入容量/延迟模型。

## 6. 验证设计

- 构造非单调 physical IDs，验证跨 page token 输出；
- 覆盖 page boundary、最后 partial page、null block；
- 对每种 layout 检查 byte interval 不越界、不非预期重叠；
- 在 abort/preempt 与 async launch 交错时加 canary，检测 use-after-free；
- connector round-trip 后以 reference attention 验证，而非只比 copy checksum；
- 多 rank 对 block-table digest 与 shard-local slot 做一致性检查。

## 7. 架构判断

**事实**：固定 revision 将 layout 表达为明确 stride permutation，并由 core 解析后供 workers 采用。**建议**：把 layout negotiation 纳入 backend capability contract；禁止 backend 以隐式默认值解释公共 config。**UNKNOWN**：任一第三方硬件对全部排列的 kernel/transfer 覆盖度，需插件代码和设备测试分别证明。

