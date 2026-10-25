# DeepSeek V4.1 推理系统拆解 — 阅读笔记

> **资料性质**：二手技术分享（slide deck），讲者个人解读，**不是** DeepSeek 官方文档，也未经本工作区源码复核。
> **原始文件**：`talk2-deepinfer.pdf`（32 页，Google Drive，2026-10-24 最后修改，2026-10-25 归档）。
> **引用方式**：行内页码指 PDF 页码，如 `[p.4]`；引用本资料时请标注 `[材料]`。
> **配套文件**：`meta.json`（元数据）、`extracted.txt`（全文，图表未包含）。

---

## 0. 一句话结论

这份材料把 DeepSeek V4.1 / DeepInfer 的生产部署拆成四件事：**池化 KV（EPD 分离 + 3FS）**、**两套 KV 的命中与重放（Block KV / SWA）**、**三层负载均衡（EPLB → DPLB → PD 扩缩容）**、**按接受率变长验证的 DSpark**。核心观点是：池化存储让 Prefill 路由可以不做 cache-aware，而真正难的是每一步"最慢那张卡"的削峰。

---

## 1. 系统形态：EPD 分离 + 池化 KV `[p.2–3]`

- EPD = Encoder / Prefill / Decode 三部分分离，系统里还有**单独的专用存储节点**。
- P 和 D **共享一个池化的 KV 存储**；因此 Prefill 路由**不做任何 cache-aware**，前缀命中靠池化层，而不是"追着有前缀的卡发"。
- 讲者把重点放在几个 load balancer 上：EPLB（专家）、DPLB（DP rank）、以及 PD 扩缩容。

> 对照：开源世界缺少好用的池化 KV，组件多为本机、互不相通；即使声称互通也常走 TCP 或低效 RDMA。所以**开源 router 只能做 cache-aware** `[p.6]`。这与 DeepSeek 自建池化存储后的"router 不需要 cache-aware"并不矛盾，前提不同。

---

## 2. 并行策略：Prefill 小 EP，Decode Wide-EP `[p.4–5]`

| 角色 | 策略 | 讲者给出的理由 |
|---|---|---|
| Prefill | 小 EP（如 EP4） | compute bound，只要能吃满本卡算力；EP4 → EP8 不改善总吞吐 |
| Decode | Wide-EP | 大概率 memory bound，用通信换带宽，每卡留更多显存，撑更大 batch、放更多冗余专家 |
| Encoder | 单卡 | 便宜卡就够 |
| TP / CP | 暂不开 | TP 用峰值吞吐换延迟；延迟不是大问题时尽量不开 |

**Prefill 用 PP 还是 EP** `[p.5]`：中小部署讲者更看好 PP，大部署看好 EP。
- 理论上 PP 不改变峰值吞吐，只多一点流水线气泡。
- V4.1 跨层共用 KV cache，且模型"每一层都不一样"，对 PP 不友好。
- 要完整复刻 DeepSeek 的 EP 方案，需要不少流量和 prefill worker；只有几个 prefill worker 的中小部署，PP 更合适。

> 这是讲者的工程判断，不是官方结论；文档本身"没聊 prefill PP"。

---

## 3. 池化 KV 的存储底座：3FS + EC `[p.7–10]`

讲者先用数据库史做类比：share-nothing 当年赢在"网络和协议做不好池化"；如今 RDMA 标配（8 计算 NIC + 1 存储/管理 NIC），带宽/延迟不再稀缺，稀缺的变成 GPU 的弹性、利用率和运维方便，于是走向存算分离 `[p.7]`。

**存储规格（讲者按 PCIe 4.0 时代估算）** `[p.8]`：
- 每台 20 块 30.73 TB SSD，单台接近 600 TB；要求读 40 GB/s、写 5 GB/s。
- 近似规格 Samsung PM1733a（已停产）；20 块理论读可达 150 GB/s、写 82 GB/s，实际要扣 EC、多副本开销。
- KV cache 不能挑着存，所有用户都要写，SSD 长期承压，**寿命是主要问题**。

**EC 与 3FS** `[p.9]`：
- 10+2 EC 存在 3FS 上，读一次打到 11 台机器；每张 GPU 对应逻辑容量 14.4 TB，EC 膨胀 20% 后 17.28 TB。
- 读直接向 11 个节点发请求，顺带改善尾延迟（讲者推荐 *The Tail at Scale*、EC-Cache OSDI'16）。
- 3FS 是 chain replication，这里应是单副本（代码支持 chain 长度 1）；3FS 本身不感知 EC，只管 placement：12 个 shard 放到 12 台机器，并保证每台 shard 均衡、挂一台后的重建流量也均衡（BIBD 平衡不完全区组设计）。
- **EC 相关计算逻辑没有开源**，只看到一些脚本。

**TTL** `[p.10]`：agent 3 天、chatbot 7 天。chatbot 隔几天续聊比例更高，TTL 只给 3 天会掉 0.5–1% 命中率。Decode 不做 offload，不受 cache 容量限制；**Decode 产生的 KV 也会写盘**，这样输出也能被命中。

---

## 4. 两类 KV 与有界重放 `[p.11–14]`

V4.1 把 KV 分成两套、分开存储 `[p.11]`：

| | Block KV（CSA） | SWA KV |
|---|---|---|
| 内容 | top-k + 压缩后的 KV | 最近 128 个 token 的滑动窗口 |
| 显存分配 | 按 128 token（压缩前）分页，每请求一张 block table | 每请求一个固定槽位（StateBlock） |
| 持久化 | 3FS | 本机内存，只在 128 的整数倍位置存快照 |
| 跨机命中 | 任何机器直接读 | Entrypoint 记录快照在哪台机器，走 400G VLAN 拉回 |

**SWA 槽位管理** `[p.12]`：
- 启动时静态预留，**不和 Block KV 互相挪用**；按 `max_num_seqs × 1.2` 预留，剩下显存才给 Block KV。
- 槽内按 `pos % 134` 当环形 buffer；多出的 6 个位置给 MTP 投机失败回退。
- 每请求 SWA 快照约 1.4 MB（20 层 × 134 × 528 B）。
- 调度时两边都分配得到才放行。

**命中与重算** `[p.13]`：
1. 先查 Block KV 得到命中长度 L1（128 的倍数，最多到 L−134）。
2. 若 L1 处有 SWA 快照，直接 replay。
3. 若快照缺失，从 L1−128 重算最后 128 个 token 来重建 SWA；Block KV 照常复用、不覆盖。
4. 重算时注意力窗口截断在这 128 个 token 内（**bounded replay**），得到的是**近似** SWA。
5. **SWA 丢失的代价是常数 128 token，与上下文多长无关。**

**关键路径重叠** `[p.14]`：匹配只查元数据，但后面要用命中长度算 compute quota，所以必须在路由前做完；拉取在另一个进程异步做，与请求排队重叠；换入在 engine 进程的线程里逐层做，与前向计算重叠。

---

## 5. 换入路径：四种搬法，选 GDR `[p.15–16]`

| 搬法 | 固定开销 / 是否占 SM | 备注 |
|---|---|---|
| 1. Copy engine 逐个拷 | 每块一次 `cudaMemcpyAsync`，独立 DMA，不占 SM | 调用次数多 |
| 2. Copy engine 批量拷 | `cudaMemcpyBatchAsync`（CUDA 12.8+），不占 SM | batch 大小要控制，CPU 提交有开销 |
| 3. SM gather | kernel 读 host 写 HBM，**占 SM**，且与前向同队列 | 会影响前向调度 |
| 4. GDR | 网卡 RDMA write，shm 直写显存，不占 SM，换入线程不调 CUDA API | 完全绕开 CUDA |

- DeepSeek 选 **GDR**。
- Block KV 一个 block 一层 22,784 B；命中 64K token 即 512 个 block，源和目的地址都是散的。
- `[p.16]` 是 GB300 实测的 GDR 每请求开销图（不含搬数据），**曲线数值未被文本提取**。

---

## 6. 在线 EPLB：重排 + 冗余 `[p.17–24]`

**两种手段** `[p.17]`：
- **重排（rearrange）**：按观测负载把专家重新分到各卡，不增加显存；分静态（启动指定）和动态（在线）。
- **冗余（redundancy）**：同一专家权重放多张卡，decode 时多读权重，用显存换吞吐。
- 冗余越多收益越小，要找合适点，case by case；DeepInfer 开源的是**在线 EPLB**。

**在线统计** `[p.18]`：
- DeepEP dispatch 支持传 GPU 计数数组（参数默认 None）。
- 每卡只统计自己卡上的专家；每步把计数拷到全局统计 buffer（窗口 N 步），拷完清 0。

**副本选择** `[p.19]`：同卡轮流分，不同卡错开起点，卡之间不通信；`c_e` 为专家 e 的副本数，`i` 为 token 在本卡这一批里的序号，23333 是质数，用来打乱报数顺序。

**重排频率** `[p.20]`：
- Prefill：每卡冗余 4 个专家，窗口 100 步，每 1000 步重排一次（中间 900 步统计会被覆盖，只看最近 100 步）。
- Decode：每卡冗余 1 个专家，窗口 1000 步，每 10000 步重排一次。
- 两边都大约 **10 分钟一次**；重排时同步搬权重：同一 NVLink 域走 NVLink，跨域走 NCCL `batch_isend_irecv` 点对点。

**优化目标** `[p.21]`：降低**每一步最慢那张卡**的负载，核心是**削峰而非平均**；在给定 worker 和专家槽位下，让观测窗口内每一步"最慢卡"总耗时最小。

**第一步：冗余名额给最热专家** `[p.22]`：过热专家放哪张卡都扛不住，只能复制；每轮给一个专家加一份副本，直到名额用完，每层单独算。伪代码：

```python
tokens_per_expert_all_diff = tokens_per_expert + arange_logical * 1e-4
for i in range(num_redundant_experts):
    score = tokens_per_expert_all_diff / logical_count      # 当前每份负载
    score_added = tokens_per_expert / (logical_count + 1)   # 多给 1 副本后的负载
    values, indices = score.max(-1, keepdim=True)           # 每步负载最高的专家
    values = values.expand_as(score).contiguous()
    score.scatter_(-1, indices, score_added.gather(-1, indices))
    values.scatter_(-1, indices, score.max(-1, keepdim=True).values)
    redundancy_indices = values.sum(0).argmin(-1)           # 选总分最低的专家
```

**第二步：Best Fit Decreasing 装箱** `[p.23]`：按负载从大到小，把专家放到"每步最忙卡负载之和"最小的卡上；先放热专家，闲专家填缝。实际按窗口内每一步都算。

**实测** `[p.24]`：
- 2 台 GB300（EP8/DP8），SWE-smith 真实 agent prompt（p50 约 7k token），greedy 固定输出 1024 token，c512。
- V4.1-Flash 384 专家 top-6，每卡 48 个。
- **EPLB 端到端快约 2%，收益几乎全来自重排**；只加冗余不重排反而略亏（多出的槽位只是多读权重）。
- **EP8 规模有限，结论不一定能推到 EP64。**

---

## 7. DPLB：Prefill / Decode 两套均衡 `[p.25–30]`

**问题** `[p.25]`：Prefill 和 Decode 都是 DP + EP，每层都要等 attention 最慢的那张卡，再一起进 MoE all-to-all。两者 attention 耗时由不同因素决定，表面是两个 load balancer，实际都围绕 DPLB。DeepSeek 自评：**吞吐达到完全均衡时约为离线 benchmark 的 80%**。

**Prefill 分桶** `[p.26]`：按上下文长度分 4 桶：`<16K`、`16–64K`、`64–256K`、`≥256K`。
- 原因：attention 与上下文长度非线性；同样 1K token，500K 上下文的比 10K 的慢；不同长度下 attention 与 MoE 占比不同。
- 所有机器共享 3FS，上下文增长时请求可在桶间迁移而不掉命中率；引擎也会按负载在桶间迁移（都是 prefill worker），每桶有单 batch 算力配额上限。
- 讲者认为"分桶比较生产化，开源 router 不知道有没有人做"，Together 之前探索过。

**Prefill 主动拉取** `[p.27]`：每个引擎有成本估算器 `cost(q, c)` 算 compute quota，约束的是 attention 耗时而非整个 step；引擎按待处理请求和剩余 quota 自己拉，**不做 cache-aware，是 pull-based**；DP 组内按 least compute quota 派发，组间和组内都用 compute tokens 作负载指标。

**Prefill 组 batch** `[p.28]`：DP 调度器按"时间预算"组 batch（按 DualPath 是 **300 ms**），再在 DP 组内 all-reduce min 收紧；队列 FIFO。估算器思路：离线跑耗时表，scheduler 查表估到合适时间再组 batch。

**Decode 均衡** `[p.29]`：同时平衡**请求数**和 **KV 总长度**，不分桶。请求数决定 GEMM 类算子，KV 长度决定 attention。
- 按 DualPath：DP 组之间用 total tokens 平衡；组内先算 KV 平均线，放宽 **5% band**；线内选请求数最少的，都超了选 KV 最少的。
- 短请求发到并发低但 KV 总长度大的卡（填 batch 维度），长请求发到 KV 总长度小的卡。

**Vision Encoder 与扩缩容** `[p.30]`：P 随机选一台机器，机器内部把图像交给活最少的 worker；PD 扩缩容时 prefill 看整体 compute tokens，decode 看 num running（系统整体 batch size）。

---

## 8. DSpark：按接受率变长验证 `[p.31]`

- Confidence head 预测每个 draft token 的接受率。
- 大 batch 下固定投 5 个去验证代价高；按接受率和离线 profile 的吞吐曲线，**动态决定每请求验证几个**。
- 每步验证 token 数从 5 降到 **3.2–4.4**，平均接受长度基本不变（**3.29 vs 3.20**）。
- 数据来自 DeepInfer 部署文档，**EP64**。

---

## 9. 可复用结论与工作区关联

**可复用结论**
1. **池化 KV 改变路由语义**：有池化层时 Prefill 可以完全不 cache-aware，命中率由存储而非 router 保证；没有池化层时 router 才被迫 cache-aware。
2. **削峰而非平均**是 EPLB 的目标函数，这决定了算法必须看"每一步最慢卡"，而不是窗口平均负载。
3. **冗余不等于收益**：冗余增加显存与权重重读，必须配合重排；EPLB 的主要收益来自重排。
4. **SWA 用 bounded replay 把缺失代价变成常数**，是长上下文/混合 KV 场景值得借鉴的工程折中。
5. **换入路径是独立优化维度**：GDR 绕过 CUDA/SM，避免与 forward 抢资源。

**与本工作区的连接**
- DeepInfer 软件栈总览：[`deepseek/README.md`](../../../deepseek/README.md)、[`deepseek/docs/README.md`](../../../deepseek/docs/README.md)。
- KV Cache / 前缀缓存 / 内存管理对照：[`20260903-vllm-ai-infra-research/03-vllm-kv-memory/`](../../../20260903-vllm-ai-infra-research/03-vllm-kv-memory/)。
- 调度与选型权衡：[`20260903-vllm-ai-infra-research/14-decisions/`](../../../20260903-vllm-ai-infra-research/14-decisions/)。
- 本地 DeepInfer 源码：`deepseek/third_party/DeepInfer/`（若已检出）。

> 注意：本资料可用于提出问题、建立假设、补充产业视角；**不能单独作为架构或兼容性结论的证据**。结论仍以固定 SHA 源码与 105 篇权威集为准。

---

## 10. 待核验清单（交给源码/官方文档）

- [ ] DeepInfer 源码中 EPLB 的两步算法、窗口/冗余参数是否与 slide 一致。
- [ ] SWA 的 `134` 槽位、`max_num_seqs × 1.2` 预留、`pos % 134` 环形实现。
- [ ] Block KV 单 block 单层 22,784 B 与 128-token 分页假设。
- [ ] GDR 换入路径（网卡 RDMA write、shm → HBM）在源码/部署文档中的对应实现。
- [ ] 3FS 上 10+2 EC、14.4 TB/GPU、TTL 策略是否有官方部署文档佐证。
- [ ] DSpark confidence head 与变长验证的调度接口。
- [ ] "EPLB 端到端 +2%" 的测试条件（2×GB300、EP8/DP8、c512）能否复现。
