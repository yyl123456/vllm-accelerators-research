# PagedAttention 的数学与分页动机

## 文档契约

- **唯一问题**：为什么自回归 attention 的 KV state 需要分页，以及 block table 如何保持 attention 数学不变。
- **In scope**：KV 增长模型、碎片、逻辑 token 到物理 page 的寻址、kernel 读取语义。
- **Out of scope**：具体 `KVCacheSpec` 类型、prefix hash、远程 KV；分别由 31、33、36 负责。
- **输入/输出**：输入为序列 token 与每层 K/V；输出为等价 attention 结果和可重定位的 page ownership。
- **证据基线**：vLLM `bb363db9...` 的 `docs/design/paged_attention.md`、`vllm/v1/worker/block_table.py`；PagedAttention 论文。

## 1. 问题不是 attention 公式，而是 state 的生命周期

单头 attention 在位置 `t` 的核心仍是：

`o_t = softmax(q_t K_0:t^T / sqrt(d)) V_0:t`

连续 KV tensor、分页 KV tensor 的数值契约相同：参与 softmax 的历史 token、head、维度和顺序不得变化。分页只改变 `K_i/V_i` 的地址解析。若 kernel 因 page 边界漏读、重读或错序，便不是性能退化，而是数值错误。

在线 serving 无法提前知道请求何时结束。为每个请求预留 `max_model_len` 会产生内部碎片；按实际长度连续增长又要求搬迁或大块 allocator。并发请求长度离散、完成时间无序，连续 allocation 还会形成外部碎片。page 把增长单位压到固定 block，使释放和复用无需搬动其余 token。

## 2. 两级地址翻译

设 block size 为 `B`，逻辑 token 位置 `p`：

- logical block index：`b = floor(p / B)`；
- intra-block offset：`u = p mod B`；
- physical block id：`P = block_table[request, b]`；
- 最终地址再由 layer/head/layout/dtype 计算。

`block_table` 因而等价于页表，但不能把类比扩展成 OS 虚存：这里通常没有 page fault 自动恢复，也没有硬件 TLB 保证一致性。scheduler/core 分配 ownership，worker 将 block IDs 写入设备侧表，attention backend 消费它。

## 3. 分页解决与不解决的事

它直接解决：

1. 请求按 block 增长，不为未到达 token 预留完整上限；
2. 请求完成后，离散 blocks 可立即归还共享 pool；
3. 多个逻辑序列可引用相同的只读 prefix blocks；
4. beam/parallel sampling 可通过共享已计算 prefix，写时再分配尾部。

它不自动解决：

- page 内最后一个 block 的内部碎片；
- KV 本身随 context 线性增长；
- block table 更新与 kernel 读取之间的同步；
- prefix 是否语义等价；
- heterogeneous attention 的 page size 协调；
- remote transfer 的一致性与租约。

## 4. block size 是跨层决策

小 block 减少尾部浪费并提高 prefix 命中粒度，却增加 block table、hash、allocator 操作和不规则访存；大 block 反之。它还进入 scheduler 对齐、kernel tile、DCP/PCP 分片、Mamba state packing 和远程传输粒度，不能由 allocator 单独决定。

粗略尾部浪费上界为每请求每组 `< B` tokens；但物理浪费要乘 layer/head/content bytes。真实优化目标不是最小化 `B`，而是在 memory efficiency、metadata/launch 开销和 backend 支持集合之间找 Pareto 点。

## 5. kernel 的正确性不变量

- 对任一有效 token，logical-to-physical 映射唯一且在本 step 内稳定；
- block ID 不得在尚有 in-flight kernel 读取时重新分配；
- padding/null block 必须被 mask，不能贡献 logits；
- K 与 V 的 layout、dtype、scale metadata 必须与 backend 解码一致；
- context length 是语义边界，allocated capacity 不是有效 token 数；
- GQA/MQA 的 query heads 到 KV heads 映射不因分页改变。

## 6. 架构师决策点

必须把四个粒度同时列在设计表中：hash granularity、scheduler granularity、physical page granularity、kernel tile granularity。把它们统称为“block size”会掩盖对齐条件和放大因子。

验收需要三类证据：随机 page permutation 下与连续 reference 数值一致；混合长度/频繁回收下 ownership 不冲突；不同 `B` 下 memory waste、metadata 与 kernel throughput 曲线。单一吞吐数字无法证明设计正确。

## 7. 失败模式

- stale block table 指向已复用 block，表现为跨请求污染；
- slot mapping off-by-one，仅在 page boundary 出错；
- speculative token 回滚后仍暴露未确认 KV；
- context mask 采用 capacity 而非有效长度，读取垃圾值；
- layout permutation 被 backend 默认为另一顺序，结果静默错误。

## 8. 结论边界

**事实**：当前 vLLM 用 block IDs 将请求逻辑位置映射到非连续 KV pages。**推论**：分页是动态 batching 和 prefix sharing 的基础 memory primitive。**未知**：没有目标 backend 的 kernel/allocator 测量，不能声称某个 block size 最优。

