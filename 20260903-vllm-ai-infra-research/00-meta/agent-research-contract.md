# Agent 研究任务契约与 Claim 规范

> **文档定位**：本规范是 `00-meta/20260903-02-证据等级与声称规则.md` 的机器可执行与工作流落地补充，供 Agent 在本工作区开展技术调研、分析或增量复核时遵照执行。

---

## 1. 研究任务标准输入结构

当 Agent 接收到技术调研或复核任务时，必须先在内部建立上下文输入结构。若用户未提供，必须明确标记为 `UNKNOWN`，严禁脑补假设：

```yaml
task_input:
  question: "具体的架构调研或实现机制问题"
  target_backend: "Ascend | QAIC | Tenstorrent | TPU | Generic-vLLM" # 必填
  target_model: "LLaMA-3-8B | Qwen-7B | ..."                         # 默认 UNKNOWN
  execution_mode: "AOT | Capture-Eager | Both | UNKNOWN"
  target_revision:
    vllm: "bb363db9a5ec2edc7b39e99b00af363a89d1fb81"                # 或指定 commit
    backend_repo: "指定 repo 及固定 SHA"
  available_repos: ["本地可直接核查的 third_party/ 路径清单"]
  time_horizon: "调研基线时间，如 2026-09"
```

---

## 2. 原子 Claim 最小字段标准

输出的技术结论必须结构化拆解为原子 Claim。在机器可读索引（如 `tools/claims_index.jsonl`）中，为方便 CLI 与轻量脚本处理，字段采用直接扁平键值对表示，但必须包含以下完整语义集合：

| 字段名 | 类型 | 说明与规范 |
|---|---|---|
| `claim_id` | string | 全局唯一 ID，格式：`CLAIM-<VENDOR>-<CATEGORY>-<NUM>`，如 `CLAIM-ASCEND-GRAPH-001` |
| `statement` | string | 原子陈述：一句话断言一个不可分割的技术事实（杜绝“支持某模型”这类复合句） |
| `applicable_tuple` | string | 适用环境元组，如 `Ascend 910B + CANN 8.0 + torch_npu 2.4`，未知写 `UNKNOWN` |
| `source_type` | enum | `02` 规范定义：`SOURCE_IMPLEMENTED` \| `RELEASE_CONTRACT` \| `VENDOR_CLAIM` \| `EXPERIMENT_RESULT` \| `INFERENCE` \| `RECOMMENDATION` \| `UNKNOWN` |
| `maturity` | enum | `02` 规范定义：`UNVALIDATED` \| `STATIC_REVIEWED` \| `HOST_VALIDATED` \| `DEVICE_SMOKE` \| `DEVICE_NUMERIC` \| `STRESS_VALIDATED` \| `PERFORMANCE_MEASURED` \| `PRODUCTION_QUALIFIED` |
| `repo` | string | 相对工作区的仓库路径，如 `third_party/vllm` |
| `sha` | string | 固定的基线 Commit SHA（必须与 `150` 文档严格一致） |
| `file_path` | string | 仓库内的具体源码文件相对路径（必须真实存在于本地对应 SHA 中） |
| `symbol` | string | 关键函数、类名或方法符号（必须真实存在于上述源码文件中） |
| `caller_chain` | string | 可达调用链说明，形如 `Caller.method -> A.func -> Target.symbol` |
| `affected_path_glob` | string | 变更扫描路径匹配 glob，如 `vllm/v1/worker/gpu/*` |
| `gaps_or_counter` | string | 反证、局限性或未开源盲区；无明确证据处写明缺失边界 |
| `owner_doc` | string | 本工作区权威 105 篇中的对应归属文档路径，如 `10-ascend/20260903-104-ACLGraph-Compiler与Shape策略.md` |
| `last_reviewed` | string | ISO 日期，如 `2026-09-28` |

---

## 3. “是否支持”类问题的回答闸门（Verification Gates）

严禁直接回答“支持”或“不支持”这种二元结论。面对“某个 Backend 是否支持某种特性/模型”时，必须分七个闸门独立断言：

1. **发现与注册（Discovery）**：Plugin 或 Platform 能否被 vLLM entrypoint 动态加载并识别？
2. **权重加载（Loading）**：Loader 能否在特定 dtype/layout 下切分并映射参数？
3. **图构建与编译（Compilation）**：静态编译器（XLA/QPC/TT-MLIR）或 Capture 引擎能否生成有效可执行产物？
4. **运行时执行（Execution）**：Runner 能否成功下发输入并驱动硬件流完成推演？
5. **数值精度（Numerics）**：在真实设备上的 logits/loss 是否在 tolerance 误差范围内？
6. **特性组合（Feature Matrix）**：Streaming、Prefix Caching、Chunked Prefill、TP/PP 组合是否兼容？
7. **生产运维（Production）**：是否有连续压测、容错恢复、兼容性承诺？

> **降级铁律**：在当前无物理卡（Host-only）的研究环境下：
> - 仅有源码可达性的结论，成熟度最高为 `STATIC_REVIEWED`，严禁标定为 `DEVICE_SMOKE` 或 `DEVICE_NUMERIC`。
> - 仅有厂商网页声称的，必须标为 `VENDOR_CLAIM × UNVALIDATED`。
> - 闭源部分必须标为 `UNKNOWN`，严禁假设推导。

---

## 4. 两套标准交付输出模板

### 模板 A：精简问答模板（适用于普通交互与即时求证）

```markdown
**结论**：[极简明确答复，1-2 句话]

**证据链**：
- **仓库与 SHA**：`repo` (`<commit-sha>`)
- **源码入口与调用点**：`path/to/file.py:L123` (`Class.method()`)
  - 核心链路：`A()` -> `B()` -> `C()`
- **证据等级**：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`

**未决空缺 / 边界**：
- [明确列出未验证项、设备依赖或闭源未决边界，无证据写 UNKNOWN]
```

### 模板 B：变更复核与雷达报告模板（适用于增量 review 与系统化交付）

```markdown
# [主题/变更复核报告]

## 1. 变更范围与环境 Tuple
- **检查对象**：`<repo>` 从 `<base_sha>` 到 `<target_sha>`
- **对应权威 Owner**：`<105 篇归属路径>`

## 2. 变更摘要 (Diff Summary)
- 文件/模块改动类别（核心接口 / 调度逻辑 / 算子定制 / 无关改动）

## 3. 受影响 Claim 状态矩阵
| Claim ID | 原子陈述 | 原状态 | 现状态 | 证据链变化 |
|---|---|---|---|---|
| `CLAIM-xxx` | ... | `STATIC_REVIEWED` | `待核实 / 破坏` | 改动了 `method()` 签名 |

## 4. 深度证据与可达性复核
- 针对受影响 Claim 重新追踪源码 caller 与调用链...

## 5. 剩余风险与后续复核建议
- 列出需要硬件物理卡测试的未决项目
```
