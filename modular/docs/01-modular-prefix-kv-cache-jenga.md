# Modular (MAX Engine) 针对 Prefix KV Cache 内存碎片化的处理机制深度调研

> 调研日期：2026-09-30  
> 源码基线：`modular/third_party/modular` (`e700d92fe1ba720701d5b9d233c791590199bfb1`)  
> 核心模块：`max/python/max/pipelines/kv_cache/paged_kv_cache/`

---

## 1. 核心问题背景：Prefix Caching 为何会导致内存碎片化？

在具备前缀缓存（Prefix Caching）的 LLM 推理系统中：
1. **外部碎片化与空洞**：长短不一的 Prompt 共享部分前缀（例如一段 150 个 token 的 System Prompt），导致部分块（Block）长期驻留且处于被锁定或共享状态，在内存中形成“散落孤岛”。
2. **异构/混合架构下的块尺寸不兼容（混合 KV 缓存）**：
   现代大模型（如 Gemma-2/3/4、Qwen-VL、DeepSeek-V2/V3 等）往往并存多种不同的 Attention 机制：
   - 全局注意力（Full/Global Attention）vs 滑动窗口注意力（Sliding Window / Local Attention）；
   - 文本 KV 缓存 vs 视觉/多模态 KV 缓存；
   - 线性注意力/RNN 循环状态（Mamba/Recurrent State）vs 经典 MHA/GQA KV。
   不同类型的 Cache 每个 Token/Block 所占用的字节数天差地别（例如 Global 块是 4MB，Sliding 块是 2MB）。若各自独占静态池，当某一类请求增多时，会导致**严重的跨 Cache 类型内存碎片化与无法互借（Stranding）**。

---

## 2. Modular MAX 的核心创新机制：Jenga Block Pool（积木式动态块池）

Modular 在 MAX Engine 中自研并引入了名为 **Jenga**（叠叠乐积木）的通用页池机制（源码位于 `max/pipelines/kv_cache/paged_kv_cache/jenga_block_pool.py`），通过**“巨大块（Huge Block）↔ 微小块（Little Block）双层几何细分”**彻底解决了碎片化与异构缓存互借问题。

### 2.1 机制一：LCM 几何最小公倍数统一物理对齐（消除外部碎片）
Jenga 首先根据模型中所有并存的 Cache 类型（Global、Sliding、VLM 等）的 Page Size，计算它们的最小公倍数（LCM）：
- 源码定义：`compute_jenga_ratios()` 与 `JengaGeometry` (`jenga_block_pool.py:51-155`)。
- **物理划分**：整个物理显存被划分为若干个等长的大块（`HugeKVCacheBlock`，例如 64MB 或 128MB）。
- **动态细分（Tiling）**：每个 Huge Block 可以按需细分成不同 Cache 的 Little Block（`LittleKVCacheBlock`）。
  例如：一个 Huge Block 可以刚好切分为 4 个 Global 块（`ratio=4`），或者刚好切分为 2 个 Sliding 块（`ratio=2`）。

```text
Huge Block 0 (Null) | Huge Block 1 (Global)  | Huge Block 2 (Sliding) | Huge Block 3 (Free)
[ N | . | . | . ]   | [ 4 | 5 | 6 | 7 ]      | [   2   |   3   ]      | [  待动态认领  ]
```

### 2.2 机制二：Parked ↔ Claimed 双状态机与惰性驱逐（Lazy Eviction）
为了让 Prefix Cache 既能长期保留提升命中率，又不会在显存紧张时造成死锁，Jenga 设计了精妙的状态机：
1. **Claimed 状态**（被认领）：当某个 Cache（如 Global）分配了该 Huge Block 中的 Little Block，且至少有一个请求正在引用（`ref_cnt >= 1`），该 Huge Block 被该 Cache 独占。
2. **Parked 状态**（停泊/挂起）：当该 Huge Block 中所有 Little Block 的请求都已推理结束，`ref_cnt` 归零，该 Huge Block 立即进入 `Parked` 状态，推入全局 `free_huge_blocks` 队列。
   * **保留 Commit**：在 Parked 状态下，所有的 Little Block **依然保留在各自的 `prefix_caches` 哈希字典中**！
   * **命中唤醒（Touch）**：若后续请求命中该 Prefix，调用 `touch()`，该 Huge Block 瞬间从 `free_huge_blocks` 中移除，重新恢复为 `Claimed`，**无需任何内存搬运或重新分配**。
   * **跨 Cache 强行征用（Eviction）**：只有当另一种 Cache（如 Sliding）显存耗尽，不得不从 `free_huge_blocks` 抢夺这个 Parked 的 Huge Block 时，系统才会调用 `uncommit_block()`，将旧 Cache 的 Prefix 索引注销并重新格式化为新 Cache 的页面。

### 2.3 机制三：LRU 双向链表与引用计数双重回收
在 `block_utils.py:324-450` 中，Modular 实现了专用的 `_FreeKVCacheBlockQueue`（基于双向链表的 O(1) 队列）：
- 正在被引用的 Block：`ref_cnt >= 1`，只留在活跃请求的映射表中；
- 空闲但缓存的 Block：`ref_cnt == 0`，**同时存在于 `prefix_cache` 字典和 `free_block_queue` 尾部**。
- **分配优先级 (`_prefer_claim_huge`)**：
  新请求需要分配 Block 时，优先复用完全干净的空闲 Huge Block；只有当没有多余 Huge Block 时，才从 `free_block_queue` 头部淘汰（Evict）最久未被访问的 Prefix Block。

---

## 3. 对比总结：高通 AOT vs Modular MAX 处理 Prefix 碎片化

| 对比维度 | 高通 QAIC AOT 静态编译路线 | Modular MAX Engine (Jenga Paged KV) |
|---|---|---|
| **基本存储单元** | 整段最大上下文槽位（`block_size = max_model_len`，如 4K/8K） | 细粒度逻辑块（`LittleKVCacheBlock`）+ 物理巨大块（`HugeKVCacheBlock`） |
| **内存分配方式** | 启动期静态锁定固定 Slot，`batch_indices = block_table[:, 0] - 1` | 运行时按需动态分页映射（PagedAttention），按 token 递增追加 |
| **Prefix Caching 支持** | **直接不支持 / 显式禁用**（抛出 `NotImplementedError`，强制设为 `False`） | **原生全面支持**（基于 Token Hash 的全前缀树匹配） |
| **碎片化应对策略** | **不保留、即用即弃**：请求结束整槽释放，新请求线性从头覆盖写入 | **Jenga 动态整块切分 + Parked/Claimed 惰性淘汰**：跨类型互借无内存空洞 |
| **多模型/混合架构支持** | 仅支持编译时固定好的单一结构，无法在不同 KV 类型间动态调配内存 | 通过 LCM（最小公倍数）统一对齐，Sliding Window、Global 与 Recurrent 自由互借 |
