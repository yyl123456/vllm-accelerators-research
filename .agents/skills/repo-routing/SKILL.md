---
name: repo-routing
description: vLLM 与主流 AI 加速器多仓库定位与任务路由。当用户提出跨仓库、跨后端对比、或需要定位具体硬件厂商在 vLLM 适配中的源码路径、固定 SHA、权威文档 owner 时触发此 skill。
---

# 多仓库发现与任务路由规范 (Repo Routing)

在当前工作区中，vLLM 上游与各厂商的适配仓库以独立子目录形式存在于 `third_party/` 之中。本 skill 规范 Agent 如何准确定位仓库、读取对应规范并路由到正确的权威文档。

---

## 1. 仓库与权威映射表

| 目标后端 | 相对工作区路径 | 固定基线 Commit SHA | 权威 Owner 文档目录 | 厂商级规范入口 |
|---|---|---|---|---|
| **上游 vLLM** | `third_party/vllm/` | `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` | `20260903-vllm-ai-infra-research/02~08` | 工作区根目录 `AGENTS.md` |
| **华为昇腾 (Ascend)** | `huawei/third_party/vllm-ascend/` | `3546357838389aa7201d9b44e234f831e98b2fd4` | `20260903-vllm-ai-infra-research/10-ascend/` | `huawei/docs/README.md` |
| **高通 (QAIC)** | `qualcomm/third_party/vllm-qaic/` | `3212cc670130b7e5290b429f781dc978d7ecf430` | `20260903-vllm-ai-infra-research/11-qaic/` | `qualcomm/docs/README.md` |
| **Tenstorrent Plugin** | `tenstorrent/third_party/vllm-tt-plugin/` | `f6995475739201fd1d2883adec477ecb37f951d9` | `20260903-vllm-ai-infra-research/12-tenstorrent/` | `tenstorrent/docs/README.md` |
| **TT-Metal** | `tenstorrent/third_party/tt-metal/` | `c634b1ca4c10eea5037d80767ce5f9e2912a401f` | `20260903-vllm-ai-infra-research/12-tenstorrent/` | `tenstorrent/docs/README.md` |
| **Google TPU** | `google/third_party/tpu-inference/` | `fd33800041510b957cd2da6199742cce2b5fd113` | `20260903-vllm-ai-infra-research/13-tpu/` | `google/docs/README.md` |

---

## 2. 路由执行工作流

当用户提出一个技术调研或跨仓库对比任务时（例如：“比较 vLLM 对 Ascend 与 TPU 的 platform 发现路径”）：

1. **识别受影响 Backend**：
   - 提取问题中的核心关键词（如 Ascend、TPU、QAIC、TT、vLLM Core）。
2. **加载全局入口与目标子仓库规则**：
   - 先读取根目录 `AGENTS.md`。
   - 进入目标第三方仓库前，检查该子仓库是否存在自身的规则文件（如 `AGENTS.md`、`README.md` 等）；若不存在，显式记录“该子仓库未声明独立 AGENTS.md 规则，遵循根规范”。
   - 厂商目录的 `docs/README.md` 作为该厂商路线的架构导航，需配合查阅已证实事实与 UNKNOWN 空缺。
3. **确认本地仓库与 SHA 状态**：
   - 使用 `git -C <repo_path> rev-parse HEAD` 确认当前工作树是否处于基线 SHA。
   - 若本地目录缺失，必须立即在回答中明确标记缺失，严禁脑补推演。
4. **定位权威 Owner 文档**：
   - 优先查阅 `20260903-vllm-ai-infra-research/` 对应章节（严禁把 `legacy/` 或早期分析设计作为第一事实来源）。
5. **输出路由元数据摘要**：
   在回复或分析开始前，必须显式声明路由上下文：
   ```markdown
   > **本次调研路由上下文**：
   > - 涉及仓库：`third_party/vllm` (`bb363db9`), `huawei/third_party/vllm-ascend` (`35463578`)
   > - 权威 Owner：`10-ascend/20260903-102-vLLM-Ascend-Plugin激活与Platform.md`
   > - 已知未开源边界：CANN 底层 Dispatcher / TaskQueue 驱动交互 (UNKNOWN)
   ```
