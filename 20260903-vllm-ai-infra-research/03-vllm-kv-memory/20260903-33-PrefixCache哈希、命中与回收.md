# Prefix Cache 哈希、命中与回收

## 文档契约

- **唯一问题**：何时可把旧请求的完整 blocks 作为新请求的等价前缀，以及如何安全回收。
- **In scope**：block hash chain、extra keys、refcount、free/eviction queue、命中边界、salt 与事件。
- **Out of scope**：physical layout（32）、hybrid hit 交集（34）、remote cache（36）。
- **证据基线**：vLLM `bb363db9...` 的 `core/block_pool.py`、`kv_cache_manager.py`、`block_hashes.py` 与 `docs/design/prefix_caching.md`。

## 1. 命中是语义声明

相同 token IDs 不是充分条件。block hash 必须把会改变 KV 的上下文纳入 identity，例如父 block hash、token block、LoRA/adapter identity、multimodal feature identity、cache salt，以及实现规定的其他 extra keys。遗漏任一影响 forward 的因素会把 cache 命中变成 silent wrong result。

hash collision 也不能由“概率很低”替代威胁模型：是否采用加密 hash、是否面对恶意多租户、collision 后有无二次校验，取决于信任域和安全要求。

## 2. 为什么只复用完整 block

固定 revision 的 `get_computed_blocks` 声明 computed blocks 必须完整。完整性使 block hash 对应固定 token interval，减少 partial overwrite 与 visibility 问题。即使 prompt 全命中，也会把最大 hit 限到 `prompt_length - 1`，因为仍需计算最后 token 才能得到 logits；受 allocation 对齐限制，可能重算整个最后 block。

这说明 prefix hit tokens 不等于“完全零算力”，hit rate 也不能直接换算延迟收益。

## 3. 状态机与数据结构

`BlockPool` 同时维护 free blocks 的 eviction order 和 `block_hash+group_id → block` 索引。block 可处于：

- allocated and referenced：活跃请求持有，不可驱逐；
- cached but unreferenced：在 free/eviction queue，hash 可命中，也可被重用；
- free and uncached：无有效 hash，可直接分配；
- null/masked：逻辑洞，不进入正常 hash reuse。

命中 acquisition 增加引用；请求完成后引用归零但 cached block 可以保留。allocator 需要新 block 时按策略逐出 unreferenced cached entry，清除其旧 hash 后再赋予新 owner。

## 4. hash chain 与 group 隔离

父 hash 将 block 序列绑定成 prefix chain，防止相同局部 token block 在不同历史下误命中。`group_id` 进入 key，使不同 KV groups 独立缓存和驱逐。事件中的 parent hash、token range、extra keys 用于让外部消费者重建 lineage；事件报告不是 ownership 真源。

`cache_salt` 可阻止不可信请求跨租户共享首 block lineage，但只有 salt 的生成、传播和生命周期都正确才形成隔离。若 salt 被复用、日志泄露或未进入首 block identity，安全目标不成立。

## 5. 回收与并发不变量

- `ref_cnt > 0` 的 block 不得驱逐；
- 插入新 hash 前必须删除 block 上所有旧 hash aliases；
- partial→full promotion 只能更新同一 block 的合法 lineage；
- free 顺序应使 tail 更早 eviction，保留更常见的前缀；
- reset prefix cache 只能在没有无法释放的引用/状态时宣告成功；
- cache event 允许延迟，但不能反向驱动本地 allocator；
- abort 与 cache promotion 交错时，只能发布 finalized token 的 hash。

## 6. 统计的正确解释

应同时报告 request-level hit、token-level hit、可用 hit、因最后 token/对齐导致的 recompute、命中查找 CPU 成本、eviction churn 和下游 latency saved。只报告 hit ratio 会奖励巨大但无价值的 blocks，也无法区分 cache 污染。

prefix cache 容量不是越大越好：缓存 blocks 与活跃 decode 竞争同一 pool。过度保留会提高 hit，却触发 preemption，从 goodput 角度可能更差。

## 7. 故障注入

1. 同 tokens、不同 adapter/MM/salt，必须 miss；
2. 强制 hash collision，验证安全策略；
3. 命中后并发 eviction，引用必须保护；
4. partial block、全 prompt 命中、跨 block stop；
5. cache reset 与运行请求交错；
6. event consumer 丢包/重放不能改变本地 correctness。

## 8. 架构结论

**事实**：当前本地 APC 通过 block hash lineage 与 refcounted pool 复用完整 blocks。**推论**：它是有条件的 memoization，不是透明存储层。**建议**：租户隔离、模型/adapter revision 与 MM identity 必须成为正式 cache-key contract。**UNKNOWN**：真实 workload 下 retention policy 是否提升 goodput，需 trace replay 测量。

