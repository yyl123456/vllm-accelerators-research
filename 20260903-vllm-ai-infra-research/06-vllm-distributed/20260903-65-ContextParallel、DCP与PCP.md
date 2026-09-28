# Context Parallel、DCP 与 PCP

## 文档契约

- **唯一问题/Owner**：DCP 与 PCP 各自切分哪个 execution/state 轴，当前 revision 的 rank 与 KV 约束是什么。
- **依赖输入**：60 的 rank namespace；31/32 的 KV group/layout；44 的 attention backend。
- **唯一输出**：prefill-work shard 与 decode-KV shard 的严格区分及组合验收合同。
- **相邻篇不得重述**：TP weight shards 归 61；P/D transfer 归 36；互联评价归 93。
- **证据基线**：vLLM `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` 的 `vllm/config/parallel.py`、`distributed/parallel_state.py`、KV spec/attention 中 DCP/PCP 分支。

## 1. 两条不同的轴

`SOURCE_IMPLEMENTED`：PCP（prefill context parallel）扩展参与 prefill 的 process world，将长 prompt 的 token/attention work 分给多个 ranks；它**不增加 KV-cache shard count**。PCP 的主要收益对象是 prefill work，不可据此把每 rank decode KV 容量除以 PCP size。

`SOURCE_IMPLEMENTED`：DCP（decode context parallel）才是分片 decode KV/context 的轴，使 ranks 对历史 state/attention reduction 分工。DCP world 与 TP/PCP 不是任意独立整数；其配置可取性与 rank group 构造需要满足固定 revision 的关系和 backend support。

## 2. 为什么 softmax 需要联合归约

`INFERENCE`：context shards 各自只看到部分 keys/values，不能分别 softmax 后拼接。数值稳定实现需先联合全局 max，再联合 exp sum，最后合并 weighted value；causal/window mask、RoPE positions 和 logical token ranges保持全局坐标。具体 collective/algorithm 由 attention backend 实现。

## 3. KV Ownership

DCP 下 global logical block/token 映射到 shard-local KV；`KVCacheSpec` 的有效 block size、head/context布局和 `dcp_world_size_for_kv_cache_spec` 参与配置。prefix hash仍描述逻辑token lineage，但命中后所有必要shards都必须可用。

PCP预填充产生的最终KV必须落入decode所期望的DCP/TP布局；如果 PCP 和 DCP groups不同，handoff/redistribution属于显式转换，不能假设同一page bytes可原样复用。P/D再跨instance时，descriptor必须包含两个布局。

## 4. 调用与状态序列

1. ParallelConfig验证sizes及world关系；
2. ParallelState建立PCP/DCP相关groups与rank坐标；
3. KV config按DCP而非PCP决定decode shard；
4. runner为prefill输入生成PCP token ranges；
5. attention backend执行局部统计与跨rank归约；
6. prefill结束后KV以decode owner布局提交；
7. decode每step读取sharded KV并归约输出。

## 5. 失败边界

- 任一context rank遗漏 collective 会使整个request/rank group不可提交；
- 非整除token长度需padding/mask，不能少算尾部；
- rank重启后的旧KV shard与collective epoch不可复用；
- prefix/SWA回收必须在所有shards保持共同progress；
- backend只支持TP不代表支持DCP或PCP。

## 6. 性能判断

PCP收益随长prefill compute增长，但增加partition/collective；DCP可降低每rank decode KV压力，却让每token attention需要通信。短context、小batch或慢互联下，通信latency可能超过收益。需要分别测TTFT、ITL、per-rank KV、collective bytes/p99和最慢rank。

## 7. 验收矩阵

覆盖 PCP-only、DCP-only、二者组合及×TP；非整除context、page/window边界、prefix hit、P/D transfer；与非CP reference逐token比较；记录实际rank table和KV bytes。未运行目标backend时保持 `UNKNOWN`。

## 8. 证据分类

- `SOURCE_IMPLEMENTED`：PCP不增加KV shard count、DCP参与KV分片和相关size约束来自固定源码。
- `INFERENCE`：布局转换/全局softmax归约为语义所需，具体实现逐backend验证。
- `RECOMMENDATION`：release矩阵分开列PCP与DCP，不使用笼统“context parallel supported”。
- `UNKNOWN`：四个第三方插件对两种模式的真实覆盖与性能，留给各vendor专题和设备验证。

