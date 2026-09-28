# Hybrid KV、Mamba 与 Sliding Window

## 文档契约

- **唯一问题**：不同 state 保留规律的 layers 如何共享一个有限 block pool，并形成可执行的共同 prefix。
- **In scope**：hybrid groups、uniform page、full/SWA/chunked-local/Mamba manager、hit intersection、sparse retention。
- **Out of scope**：通用 hash（33）、quant/offload（35）、remote transfer（36）。
- **证据基线**：vLLM `bb363db9...` 的 `docs/design/hybrid_kv_cache_manager.md`、`core/kv_cache_coordinator.py`、`single_type_kv_cache_manager.py`。

## 1. hybrid 的本质是 retention function 不同

full attention 需要保存从 0 到当前位置的所有 K/V；sliding window 只需近期窗口；chunked-local 按 chunk 保留；Mamba 保存递推 state，其 bytes/token 和“可从哪个 checkpoint 恢复”均不同。统一 scheduler 必须回答两个问题：每组需要哪些 physical slots，以及各组共同能从哪个 token position 继续。

后一问题不能用 `max(hit_i)`；可执行 prefix 必须满足所有参与 forward 的 state。通常是受约束的交集，并可能要求重算 junction 周围 token。

## 2. uniform page 的约束

共享 pool 要求各 groups 的 physical page bytes 可统一。对层数比例规整的模型，可把同类 layers 分组以得到相同 page size；比例不整齐时需 padding layers/state，产生内存浪费。Mamba state 可能远大于 attention 单 token KV，系统可能扩大 attention block size并 padding Mamba state，导致数百 token 的 block，影响 hash/admission/kernel 粒度。

因此“支持 hybrid model”至少包括：配置可构造、容量可接受、hit 算法正确、backend kernel 覆盖、实际模型质量验证五层证据。

## 3. coordinator 分层

顶层 `KVCacheManager` 对 scheduler 提供 lookup/allocate/free。`KVCacheCoordinator` 管多个 single-type managers 和共享 `BlockPool`。固定 revision 根据配置选择 no-prefix、unitary 或 hybrid coordinator；每个 manager 负责自身 allocation、skip/free 与 hit 规则。

coordinator 要求 scheduler block size 是 hash block size 和各 group block size 的倍数。这是跨组推进的最小共同边界；不满足时 token budget 可能停在某组无法表示的位置。

## 4. hit intersection

full attention 从左向右寻找连续 prefix；SWA 可关注候选位置之前的窗口，因此查找方向/条件不同。full+X 通常先取得 full-attention 上界，再在 X 中寻找不超过该边界的可恢复点。固定 revision 还返回 `shared_prefix_boundary`，用于 pin 稀疏 retention group 尚未缓存的 junction，避免跨请求 reuse 被回收破坏。

对于 connector 路径，hybrid per-group local hits 可能分叉：full tail 被逐出而更深的 Mamba checkpoint 仍在，或相反。实现必须在 remote state 能否补齐之前保守 reconcile，不能把某一组的深 hit 当作全模型 hit。

## 5. allocation 的事务性

多 group allocation 不能逐组“边命中边申请”而无回滚：早期 group 的 external allocation 可能驱逐后续 group 尚未 touch 的 cache-hit blocks。固定 revision 明确采用两阶段：先 touch 所有 groups 的 local hits，再为 external tokens 分配。一般不变量是：要么所有组获得同一推进所需 state，要么该 step 不可提交。

MTP/EAGLE 还会写入 lookahead/draft state。只有 finalized KV 可进入 prefix cache；可重新 prefill 的尾部必须排除。scheduler block 小于 speculative lookahead 时，prefix caching 可能被配置拒绝。

## 6. sparse retention 与 memory accounting

SWA 释放窗口外 blocks 可节省 resident memory，但 logical positions 仍需 null/mask 语义。capacity predictor 必须与实际 recycle rule 相同，否则 admission 认为全序列可容纳，执行中却因保留边界差异 OOM，或反向过度保守。

单一 global LRU 会让不同成本/复用价值的 groups 竞争：一个 Mamba checkpoint page 和一个 full-attention page 的重算成本未必相同。当前策略存在不代表它对所有 workload 最优。

## 7. 架构验收

- 对每个 model layer 输出 spec/group/page padding 审计表；
- 枚举所有 group hit-length 组合，验证结果是可恢复边界；
- 低内存压力下交错 eviction，比较完整 recompute；
- chunked prefill、MTP、preemption、remote hit 组合测试；
- 分开报告 padding waste、实际 resident bytes、hit saved compute；
- 无设备 kernel 证据时，将 vendor hybrid 支持标 `UNKNOWN`。

## 8. 结论

hybrid KV 不是为不同层“多建几张 block table”这么简单；它是 state-machine checkpoint、共同推进边界和共享 pool 公平性的联合问题。架构评审的关键是逐组证明恢复条件，并证明 allocation/eviction 是跨组原子的。

