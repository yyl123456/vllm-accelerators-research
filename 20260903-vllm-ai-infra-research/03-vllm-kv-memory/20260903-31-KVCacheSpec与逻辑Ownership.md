# KVCacheSpec 与逻辑 Ownership

## 文档契约

- **唯一问题**：谁定义每层 state 的逻辑形状、容量与 group ownership。
- **In scope**：`KVCacheSpec`、`KVCacheGroupSpec`、`KVCacheConfig`、core/worker 配置契约。
- **Out of scope**：物理 stride 和 block table（32）、cache hash（33）、容量预算（35）。
- **证据基线**：vLLM `bb363db9...` 的 `vllm/v1/kv_cache_interface.py`、`kv_cache_spec_registry.py`、`core/kv_cache_utils.py`。

## 1. 三层契约

`KVCacheSpec` 描述“一层需要怎样的 state”，核心字段/派生量包括 `block_size`、`tokens_per_state`、`state_content_size_bytes`、`page_size_bytes`、最大 memory usage 与每请求最大 block-table entries。它是逻辑需求，不是 allocator 中的实际 block。

`KVCacheGroupSpec` 将可共享同一 block table 的层绑定为一组，包含 layer names、组级 spec、是否为 EAGLE/MTP draft group、是否参与外部 KV transfer。组是 allocation/transfer 的一致性边界。

`KVCacheConfig` 是 engine-core 解析后的全模型合同：统一 `num_blocks`、物理 tensor 描述、groups、retention policy 和 layout。worker 应采用它初始化 cache，不能在各 rank 独立猜测。

## 2. 类型系统承载硬件语义

当前 `KVCacheSpecKind` 包含 full attention、MLA、sliding window、sliding-window MLA、Mamba、chunked-local、sink、encoder-only、cross-attention 与 unknown。类型差异不只是名字：

- token 对 state 的增长规律不同；
- 可被回收的历史范围不同；
- prefix-cacheability 和 hit 判定不同；
- page content、quant scales、kernel state 数可能不同；
- DCP/PCP 对有效 block size 的约束不同。

把所有 state 强制解释为标准 K/V `[token, head, dim]` 会破坏 Mamba、MLA 或 cross-attention 的语义。

## 3. Ownership 表

| 对象 | 创建/裁决 owner | 消费者 | 不变量 |
|---|---|---|---|
| layer spec | attention backend/model inspection | config builder | 同一 layer 只有一个有效 spec |
| group | KV config builder | manager、worker、connector | group 内共享 block-table 语义 |
| `num_blocks` | memory profiling + config merger | pool、workers | 所有 participating ranks 一致 |
| request→blocks | core KV coordinator | scheduler output、worker table | core 是逻辑 ownership 真源 |
| physical tensors | worker | attention backend/connector | 符合 core 下发 layout/config |
| block 生命周期 | core pool/coordinator | scheduler | in-flight 使用结束前不得复用 |

## 4. core 与 worker 不能各自拥有真相

core 知道请求状态、cache hit、free/refcount 和 admission；worker 知道真实 device allocation、tensor address 与 kernel 完成。系统依靠 `KVCacheConfig` 和每 step 的 block IDs 将两者连接。若 worker allocation 少于 core 宣告的 blocks，故障应 fail closed；若 worker偷偷扩大或改 layout，connector 和 metrics 会得到不一致视图。

多 rank 下应先聚合可用 memory，再选择所有必要 rank 都能兑现的 config。最大值会令最小容量 rank OOM；按 rank 独立配置会使 collective/kernel metadata 分叉。

## 5. group 的代价

group 减少 allocator 和 table 数量，但要求成员在给定请求下共享相同 slot pattern。group 太粗会浪费 memory 或错误复用；太细会增加 metadata、hash 和调度工作。相同 attention 类型也不充分：dtype、head geometry、page bytes、transfer eligibility、layout 必须兼容。

`transfer_group_ids` 明确外部传输子集；因此“本地 cache groups”与“可迁移 state groups”不能默认相等。draft-only 或 backend-private state 可能不在可移植合同内。

## 6. 配置形成的验证问题

1. 模型所有需要 state 的 layer 是否均登记，sharing layer 是否明确 alias？
2. 每个 spec 的 page bytes 是否与 tensor descriptor 完全一致？
3. scheduler block size 是否为 hash 和各 group block size 的公倍数？
4. 最大 sequence 是否能由 block-table row 表达？
5. quant mode 对应的 scale/zero-point state 是否计入？
6. 每 rank config digest 是否相同，差异是否在允许的 shard 字段内？

## 7. 架构决策与 UNKNOWN

新增 backend 首先应实现/复用 spec 合同，而不是从 CUDA tensor shape 反推。若硬件要求私有 layout，应把它显式编码在 config/layout capability 中，并提供转换或声明不可 transfer。

**事实**：固定 revision 中 spec/group/config 分层存在，config 可选择 transfer groups。**推论**：这是 platform-independent logical ABI。**UNKNOWN**：第三方插件若绕过该配置并持有私有 state，需逐插件源码与运行证据确认，不能从注册成功推出兼容。

