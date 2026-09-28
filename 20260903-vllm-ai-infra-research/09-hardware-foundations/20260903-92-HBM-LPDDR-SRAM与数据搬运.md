# HBM、LPDDR、SRAM 与数据搬运

## 文档契约

- **唯一问题**：memory hierarchy的容量、带宽、延迟和显式搬运如何限制LLM serving。
- **In scope**：off-chip DRAM、on-chip SRAM/cache、DMA、tiling、roofline、KV/weights residency。
- **Out of scope**：互联（93）、completion（94）、vendor数值。
- **证据基线**：本篇只定义通用 memory/traffic 模型；不把任何 HBM、LPDDR、SRAM 名称或峰值视为设备事实。厂商实例引用 150 的具体 `WEB-*` ID，并使用 02 的正交标签。

`INFERENCE`：HBM、LPDDR 与片上 SRAM/cache 在容量、带宽、能效和可编程性上的取舍只能作为分析维度，不能仅凭名称排序。channel、controller、访问模式、ECC 和并发会改变 effective bandwidth；任一设备数值都需 `VENDOR_CLAIM × UNVALIDATED` 来源，并由 `EXPERIMENT_RESULT × PERFORMANCE_MEASURED` 独立验证。

decode每生成token读取大部分weights，低batch时常memory-bound；batch提高weight reuse但增加KV和latency。prefill的matrix reuse更高，常转向compute-bound。KV访问随context增长，paged/irregular layout降低有效带宽。

显式scratchpad架构要求compiler/runtime安排DRAM↔SRAM DMA和double buffering；cache-coherent架构由hardware管理但miss不可预测。两者都需tile fits模型：weights、activations、KV slice、scales、collective buffers和program占用同时进入SRAM。

roofline用operational intensity `ops/byte` 与sustained bandwidth/compute估上界，但需按实际quant bytes、re-read、padding和fusion计算。峰值HBM带宽不能解释small random pages或host-device copy。

验证分别测sequential/random、read/write、size/concurrency、compute overlap；E2E采集weights/KV/activation bytes与stall。容量模型保留runtime/headroom，避免datasheet容量全部分给model。

## Residency 与 traffic 模型

`M_total = weights + KV + activation_peak + workspace + graph/trace + communication + runtime + reserve`。每项标 replicated/sharded、lifetime、alignment、allocator owner。带宽分析用实际 bytes/token 或 bytes/request。

SRAM/L1 可保存 tiles/scales/partial reductions，但空间与 bank/port 受限；编译器不能安排 double buffering 时，名义 SRAM 不会自动降低 DRAM traffic。LPDDR/HBM sustained BW 还受粒度、bank、并发及 DMA/collective 竞争。

`RECOMMENDATION`：保存 per-rank memory ledger/data-movement trace；`UNKNOWN`：datasheet 到模型 sustained bytes/s 的系数。
