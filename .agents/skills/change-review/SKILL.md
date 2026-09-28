---
name: change-review
description: 针对 vLLM 上游或各硬件后端第三方仓库的 commit 变更进行增量复核。通过 git diff/log 分析接口与逻辑变动，精确定位受影响的权威 Claim，并判定破坏性或兼容性影响。
---

# 变更复核与版本漂移分析规范 (Change Review)

当上游 vLLM 或某个硬件适配插件（Ascend、QAIC、TT、TPU）发生版本升级或 commit 推进时，本 skill 规范 Agent 如何进行增量影响复核，避免不必要的全文通读。

---

## 1. 触发时机与输入要求

- **触发场景**：上游 release 发布、commit SHA 推进、用户询问“某 commit 带来了什么影响”或“最新代码是否破坏了现有 Claim”。
- **标准输入参数**：
  - `repo_path`：目标仓库路径（如 `third_party/vllm`）
  - `base_sha`：基线 Commit SHA（默认查阅 `150` 文档）
  - `target_sha`：待评估的最新 Commit SHA 或分支（如 `HEAD`、`v0.6.2`）
  - `topic_focus`：关注主题（可选，如 `attention`、`worker`、`platform`）

---

## 2. 变更复核分析流程

1. **只读 Diff 检查**：
   - 严禁执行改写操作。执行 `git -C <repo_path> diff --name-status <base_sha> <target_sha>` 获取改动文件列表。
   - 获取简要日志：`git -C <repo_path> log --oneline <base_sha>..<target_sha>`。
2. **文件改动筛选与全量影响比对**：
   - 严禁盲目丢弃 docs/CI/tests 文件：文档、CI 配置和测试用例同样可能改变公开对外契约、暴露构建约束或提供关键反证。
   - 应按 Claim 的语义相关性与路径匹配进行全面审查。
3. **Claim 候选匹配**：
   - 对比改动文件路径与 `tools/claims_index.jsonl`（或 CSV）中登记的 `affected_path_glob`。
   - 提取所有落入改动范围内的 `claim_id`。
4. **深入调用链与符号核查**：
   - 针对命中的每个 Claim，检查具体 diff：
     - 函数签名是否改变？（参数增删、类型变化）
     - 返回值语义是否改变？
     - 控制流条件或 guard 是否被移除？
   - **判定状态**：
     - `UNMODIFIED`（未实质影响）：改动未波及该 Claim 依赖的函数与调用链。
     - `VERIFIED_BREAKAGE`（确认破坏）：接口被删除或行为发生根本改变。
     - `NEEDS_REVERIFICATION`（待核实）：代码有深层逻辑重构，需重新静态或设备求证。
5. **输出增量复核报告**：
   - 严格采用 `00-meta/20260903-03-Agent研究任务契约.md` 中的【模板 B】输出报告。
   - 若两次 SHA 之间无相关文件修改，明确报告“无影响（空变更）”，严禁虚构任何事件或兼容性破损。
