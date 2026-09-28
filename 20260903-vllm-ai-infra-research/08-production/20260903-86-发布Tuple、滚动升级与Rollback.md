# 发布 Tuple、滚动升级与 Rollback

## 文档契约

- **唯一问题**：一个异构推理版本如何被唯一标识、灰度、升级和回退。
- **In scope**：release tuple、compatibility gates、canary、mixed-version protocol、artifact/cache、rollback。
- **Out of scope**：OOT漂移方法（74）、各vendor tuple（105/116/126/135）。
- **证据基线**：固定仓库SHAs、runtime/driver contracts与生产原则。

release tuple至少含vLLM/plugin SHA或package、framework、compiler/runtime、driver/firmware、kernel/collective libs、model/tokenizer/quant revision、hardware SKU/topology、parallel/config/features和container digest。“vLLM版本号+设备名”不可复现。

升级前执行source/import/config、startup、numeric、failure、performance、SLO五层门禁。canary流量需包含高风险features，不只短greedy prompt。比较以相同workload和quality为前提。

rolling期间router必须认识old/new capability与artifact/KV protocol。P/D两端、shared remote cache或adapter service若协议不兼容，不能任意混版；需版本握手或整体切换。local prefix/KV通常不跨进程持久，但remote state必须带schema/revision。

rollback不仅回镜像，还要处理新schema artifacts、cache、数据库/registry和in-flight requests。不可向旧版本加载新binary artifact；必要时generation隔离。触发条件包括error/SLO/quality/fallback/compile异常，而非只看进程健康。

发布证据保留tuple、命令、raw结果、canary时间窗和decision owner。没有目标硬件numeric/perf证据时可发布为实验/preview，但不得标production supported。

## 升级状态机

```text
publish immutable tuple -> isolated new replica -> readiness/numeric
 -> canary capability traffic -> compare SLO/quality -> shift traffic
 -> drain old epoch -> retire artifacts
```

mixed-version P/D、KV connector、remote cache、adapter protocol 必须先证明 wire/layout compatibility，否则 version-sticky routing。rollback point 必须早于不可逆 shared-state schema 变化。

`RECOMMENDATION`：manifest 与证据原子绑定，router 按 tuple/capability 路由；`UNKNOWN`：firmware/runtime 是否支持无损 rolling。
