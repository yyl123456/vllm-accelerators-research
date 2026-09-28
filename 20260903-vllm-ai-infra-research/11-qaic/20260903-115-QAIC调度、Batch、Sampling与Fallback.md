# QAIC 调度、Batch、Sampling 与 Fallback

## 文档契约

- **唯一问题/Owner**：QAIC两种mode下，vLLM调度输出怎样成为合法device batch并完成sampling。
- **依赖输入**：25 budget；42/46 batch与sampling；113/114 mode contracts。
- **唯一输出**：profile/batch限制、device/host sampling与fallback能力矩阵。
- **Out of scope**：单项feature原理；release验证（116）。
- **源码基线**：vllm-qaic `3212cc...` worker/model runner、async outputs、patch rejection sampler、spec decode。

Pyt mode可较动态地materialize batch，仍受kernel alignment、device memory和activation/channel限制；AoT mode必须命中QPC profile。scheduler token budget不是充分admission，需加入mode/profile predicate。

sampling可能在host或device相关路径完成。无论位置，processor顺序、RNG request ownership、TP/多device一致、logprob和grammar mask必须满足46/49A。async output若推迟D2H或sample，应以step sequence防stale output。

spec decode的draft/verify会引入不同batch shape与rollback；源码含draft model和rejection sampler patch，但支持需绑定mode/model/profile。MM/LoRA同理。

fallback分三类：kernel实现fallback、profile选择较大bucket、mode切换。前两者可能只影响性能；mode切换会改变artifact/KV/weights ownership，不能在in-flight request透明执行。所有fallback需metrics和capacity降额。

验收以batch size/context/profile边界、queue churn、slow output、sampler processors、spec partial accept和unsupported combination覆盖。`UNKNOWN`：无device无法判断fallback实际触发率与SLO。

## Completion 与 stale-output 防线

AoT session 的 `np_run()` 返回 execution-object identity，sync path调用 `complete_inf()`；async path延后完成/输出处理。该 identity 必须与 scheduler step、batch rows、QPC profile 和 request epochs绑定，随后才能提交 sampled tokens 或释放 slot。

| fallback | 是否可透明 | 条件 |
|---|---|---|
| 较大 QPC bucket | 可能 | mask/数值/容量等价 |
| kernel fallback | 可能 | op schema/数值/completion等价 |
| Pyt↔AoT mode | 否，默认 | weights/KV/session所有权变化 |

`SOURCE_IMPLEMENTED`：np_run/complete_inf 与 async config；`RECOMMENDATION`：metrics含 exec id/profile/fallback；`UNKNOWN`：abort 后 session work 是否可取消。
