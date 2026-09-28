# PCIe、NoC、ICI、Ethernet 与 Collective

## 文档契约

- **唯一问题**：片内、卡间、节点间互联如何改变parallelism与KV transfer的可行边界。
- **In scope**：拓扑、latency/bandwidth、DMA/RDMA、collective algorithms、contention。
- **Out of scope**：rank软件（60-66）、P/D protocol（36）、vendor命名细节。
- **证据基线**：本篇是通用 topology/path/collective 成本模型。NoC、ICI、PCIe 或 Ethernet 的具体实现只由 150 中对应厂商资料形成 `VENDOR_CLAIM × UNVALIDATED`；vLLM 可达通信路径需另有 `SOURCE_IMPLEMENTED × STATIC_REVIEWED`。

`INFERENCE`：可用路径可能跨 core/tile NoC、die/package link、card/host PCIe、专用 accelerator interconnect 和 NIC；也可能经 host memory/IOMMU 或 peer/RDMA direct。哪条路径存在是设备事实，必须单独举证。aggregate bidirectional peak 不等于单 flow 或 collective sustained。

collective成本由message size、world、algorithm和topology决定。ring偏带宽、tree偏latency，hierarchical先域内后域间；all-to-all对MoE更敏感于bisection和incast。decode的小collective常被startup latency支配。

TP/CP要求每层/每token频繁通信，应放高带宽低延迟域；PP传activation频率较低但有bubble；DP主要无模型内collective；EP依赖all-to-all；P/D传KV是大块但对TTFT有deadline。

NoC contention还发生在compute cores访问DRAM、DMA与collective共享链路时。理论上可overlap的通信可能抢同一memory/NoC资源而降低compute。

验证必须保存physical topology，测point-to-point和各collective在多size/concurrency下的p99，并做compute overlap与故障。不能从“支持PCIe/以太网”推断跨节点TP可用。

## 路径账本

每个 collective/transfer 记录 source/destination memory、staging、DMA、NoC/PCIe/专用链路/NIC hops、protocol、completion。相同 all-reduce 在 package/host/跨 host 是不同故障域。

| workload | 消息特征 | 首要指标 |
|---|---|---|
| TP decode | 高频小/中 | startup/p99 |
| TP prefill | 较大 | sustained BW/overlap |
| EP | all-to-all/倾斜 | bisection/tail |
| P/D KV | 大块/deadline | transfer+commit |

`RECOMMENDATION`：报告 algorithm/physical path；`UNKNOWN`：只凭链路名不能判断 collective 性能。
