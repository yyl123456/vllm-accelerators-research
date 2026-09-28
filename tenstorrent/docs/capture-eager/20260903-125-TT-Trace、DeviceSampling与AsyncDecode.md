# TT Trace、Device Sampling 与 Async Decode

## 文档契约

- owner：capture/replay、sampling owner 与异步 decode 提交/回读协议。
- 依赖输入：`54` 的通用 trace 模型、`124` 的 step contract。
- 唯一输出：地址/shape/RNG/completion 的一致性条件。
- 不讨论：mesh 或调度容量；由 `121`、`124` 持有。

## Trace contract

TT trace replay 消除重复 host dispatch，但捕获对象通常绑定 model、mesh、shape/bucket、buffer address、program config 与 runtime state。`trace_mode` 可为 `all`、`decode_only` 或 `none`；`trace_region_size` 为 runtime 预留 trace 空间；`enable_model_warmup` 决定健康前是否完成预热/捕获。

trace 命中必须比较完整 guard key，不能只比较 batch size。KV page table、row mapping、sampling mode 或 mesh config 若影响 captured program，遗漏任一维度都可能 replay 错图。地址稳定性要求 buffer 在最后一次 replay completion 前不得释放或迁移。

## Device sampling owner

`TTPlatform` 读取 model class 的 `supports_sample_on_device`。只有声明能力的模型才接受 `sample_on_device_mode=all|decode_only`；否则 fail closed。`TTModelRunner.sample_tokens()` 与 generator 的 `sample_decode_on_device()` 形成 device path；host compatibility path 则使用 vLLM LogitsProcessor/sampler。

采样位置改变数据量与语义：device 采样只需回传 token/少量 metadata，host 采样需回传 logits；但 device 必须实现 temperature、top-k/top-p、seed/RNG progression 以及被声明支持的 processors。不能在 device 已推进 RNG 后再因 unsupported processor fallback 到 host，否则重试不可复现。

## Async decode 两阶段协议

固定插件的 `docs/DECODE_RELOAD_CONTRACT.md` 要求支持模型将 decode 拆为提交与回读：`decode_forward(..., read_from_device=False)` 后由 `read_decode_output(..., async_read=True)` 读取。`async_decode.py` 管理 pending step；`register_pending_async_step()` 记录尚未收割的执行。

```text
step n submit -> device execution pending
host schedules step n+1
step n readback -> validate exact submission identity
commit token/KV -> permit row/block reuse
```

能力由 `model_capabilities['supports_async_decode']` 明示。未声明时平台关闭 async scheduling；声明意味着模型承担 split API、reload protocol、准确回读和 abort/drain 责任，而不是仅“函数可以异步返回”。

## Reload 与失败边界

`TTDecodeReloadPlan` 描述 decode row 在 request transition 后需要重载的状态。异步路径必须把 reload version 与 submission identity 绑定；旧 completion 到达时不能覆盖新 request。abort 后可以丢弃 token，但必须 drain/确认设备工作后再释放相关 buffer/KV ownership。

block-output model 当前固定源码要求同步 serving、`max_num_seqs=1`、device-owned complete output，并对 `trace_mode='all'` 与 warmup 给出已验证配置警告；不能套用普通 autoregressive async 假设。

## 结论分类

- `SOURCE_IMPLEMENTED`：capability gates、`TTModelRunner.sample_tokens`、async pending/reload 类型与 block-output guards。
- `INFERENCE`：device sampling 的主要系统收益是减少 logits 回传和 host work，但实际收益需测量。
- `RECOMMENDATION`：把 submission id、trace key、reload version、RNG epoch 写入诊断事件。
- `UNKNOWN`：设备上的分布一致性、尾延迟与长期 replay 稳定性。

