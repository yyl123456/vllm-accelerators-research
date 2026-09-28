# 项目工作说明与研究指南

## 项目核心定位与目标

本工作区用于系统化调研与分析主流 AI 加速器厂商（Huawei Ascend、Qualcomm Cloud AI、Tenstorrent、Google TPU 等）在大语言模型（LLM）推理系统中的软件栈架构。

> **核心原则与权威入口**：
> 1. 本工作区**不是**上游 vLLM 官方仓库，而是围绕 vLLM 与主流加速器适配的**架构研究与证据审查知识库**。
> 2. **本工作区以 `AGENTS.md` 为唯一全局规范与任务入口**，所有 Agent 启动与任务派发统一以此为准。
> 3. **唯一权威文档集**：权威技术结论唯一存储于 [`20260903-vllm-ai-infra-research/`](./20260903-vllm-ai-infra-research/) 目录下的 **精确 105 篇文档**（清单以 [`20260903-架构研究文档重构执行计划.md`](./20260903-vllm-ai-infra-research/20260903-架构研究文档重构执行计划.md) 为唯一 Manifest）。
>    - `legacy/` 目录仅保留早期摘要草稿，已被权威集 supersede，**严禁**作为架构或兼容性结论引用。
>    - `analysis-design/` 与厂商 `docs/` 为历史参考与过渡索引，若有冲突以 105 篇权威集及固定源码为准。
>    - 工作流规范参见 [`00-meta/agent-research-contract.md`](./20260903-vllm-ai-infra-research/00-meta/agent-research-contract.md)。

调研聚焦两大核心推理执行路线的深度技术调研与对比分析：

1. **路线 A：AOT 静态编译路线（Ahead-of-Time Compilation）**
   - **核心机制**：离线或预先将完整计算图/子图进行图优化、算子融合、常量折叠与静态内存规划，生成设备执行二进制。
   - **关键技术点**：静态 Shape 限制与 Shape Bucketing（分桶）、StableHLO/MLIR/XLA 编译管线、权重离线预打包、静态 KV Cache 与设备私有内存排布、编译产物缓存与加载（Compilation Cache / Serialization）。
   - **典型代表**：Qualcomm QAIC（QEfficient / QAIC Compiler）、Google TPU（OpenXLA / PJRT / JAX）、Tenstorrent（TT-MLIR / TT-XLA）。
2. **路线 B：解析执行与动态 Capture 路线（Eager & Graph Replay）**
   - **核心机制**：基于 PyTorch 动态图/Eager 逐算子派发，结合图捕获（Graph Capture / Replay）与动态 JIT 技术消除 Python 开销与 Kernel 启动时延。
   - **关键技术点**：PrivateUse1 / C++ Dispatcher 接入、Graph Capture（类似 CUDA Graph / CANN Graph）、动态 Shape 处理、通用 PagedAttention/FlashAttention 算子定制、Runtime 任务流与下发队列（Stream/Event/Queue）。
   - **典型代表**：Huawei Ascend（Torch_NPU + CANN ACLNN + TorchAir Capture + vLLM-Ascend）、PyTorch 原生 Eager/CUDAGraph、Qualcomm torch-qaic eager 模式。

最终产出是形成高置信度的厂商软件栈调研报告、两条路线的技术选型评判矩阵，以及软硬件接口抽象建议。

---

## 协作者与 Agent 克隆指引（Git Submodules）

本工作区通过 `.gitmodules` 声明并锁定了 6 大核心第三方源码仓库，作为只读审查与证据追踪的锚点。

### 1. 完整拉取（含 Submodules）
外部协作者或新环境初始化时，建议运行：
```bash
# 克隆主仓库并自动递归检出所有绑定的第三方仓库固定 Commit SHA
git clone --recurse-submodules <本仓库URL>

# 或在已有仓库下初始化并更新子模块
git submodule update --init --recursive
```

### 2. 六大固定源码 Revision

| 仓库 | 相对路径 | 固定 SHA | 角色与用途 |
|---|---|---|---|
| **vLLM** | `third_party/vllm/` | `bb363db9a5ec2edc7b39e99b00af363a89d1fb81` | 上游执行引擎、KV 管理、编译及 Platform 插件契约基线 |
| **vLLM Ascend** | `huawei/third_party/vllm-ascend/` | `3546357838389aa7201d9b44e234f831e98b2fd4` | 华为昇腾适配插件、ACL Graph 捕获与 ModelRunner |
| **vLLM QAIC** | `qualcomm/third_party/vllm-qaic/` | `3212cc670130b7e5290b429f781dc978d7ecf430` | 高通 Cloud AI 适配插件、AoT 与 PYT 双模式选择 |
| **vLLM TT Plugin** | `tenstorrent/third_party/vllm-tt-plugin/` | `f6995475739201fd1d2883adec477ecb37f951d9` | Tenstorrent 适配插件、自定义 Engine/Runner 与分流 |
| **TT-Metal** | `tenstorrent/third_party/tt-metal/` | `c634b1ca4c10eea5037d80767ce5f9e2912a401f` | Tenstorrent 底层编程模型、Tensix 计算内核与 CommandQueue |
| **TPU Inference** | `google/third_party/tpu-inference/` | `fd33800041510b957cd2da6199742cce2b5fd113` | Google TPU vLLM 适配、XLA 编译缓存与 Shape 分桶 |

> **只读隔离铁律**：所有子模块目录仅作为只读对照基线。严禁直接在子模块中修改代码或产生未提交改动。

---

## 任务类型与路由导航

| 问题类型 | 核心关注点 | 必读权威文档 (105 篇范围) | 对应源码仓库 / 真实基线路径 |
|---|---|---|---|
| **vLLM 请求链与调度** | HTTP/OpenAI 请求转化、AsyncEngine、EngineCore、调度循环与 Preemption | `02-vllm-control/` (20260903-20 ~ 28) | `third_party/vllm/vllm/v1/engine/`, `vllm/v1/core/` |
| **KV Cache 与内存管理** | BlockManager、KVCacheSpec、物理 BlockTable、Prefix Cache、Hybrid KV | `03-vllm-kv-memory/` (20260903-30 ~ 36) | `third_party/vllm/vllm/v1/core/kv_cache_manager.py` |
| **执行模型与 Forward** | ModelRunner、InputBatch 数据契约、Worker 初始化与 Attention 选择 | `04-vllm-execution/` (20260903-40 ~ 49B) | `third_party/vllm/vllm/v1/worker/gpu/model_runner.py` |
| **编译优化与图捕获** | Torch.compile、CUDAGraph / ACL Graph、Shape 分桶与 JIT 消除 | `05-vllm-compile/` (20260903-50 ~ 55) | `third_party/vllm/vllm/compilation/` |
| **插件与 Backend 发现** | Platform 插件注册、PlatformEnum、Backend 激活与接口继承 | `07-platform-plugin/` (20260903-70 ~ 74) | `third_party/vllm/vllm/platforms/interface.py` |
| **Huawei 昇腾 (Ascend)** | NPUStream、TaskQueue、ACL Graph、双重异步、vllm-ascend Runner | `10-ascend/` (20260903-100 ~ 105) | `huawei/third_party/vllm-ascend/vllm_ascend/` |
| **高通 (Qualcomm QAIC)** | AoT 模式 vs PYT 模式切换、QPC Artifact、QEfficient 转换 | `11-qaic/` (20260903-110 ~ 116) | `qualcomm/third_party/vllm-qaic/vllm_qaic/` |
| **Tenstorrent (TT)** | Tensix、TT-Metal CommandQueue、多 Lane 调度、Device 端 Sampling | `12-tenstorrent/` (20260903-120 ~ 126) | `tenstorrent/third_party/vllm-tt-plugin/src/vllm_tt_plugin/`<br>`tenstorrent/third_party/tt-metal/` |
| **Google TPU / XLA** | XLA 编译管线、PJRT C ABI、tpu-inference 适配、Shape Bucketing | `13-tpu/` (20260903-130 ~ 135) | `google/third_party/tpu-inference/tpu_inference/` |
| **跨 Backend 选型与决策** | 架构差异对比、选型决策矩阵、硬件抽象与接入分期 | `14-decisions/` (20260903-140 ~ 144) | 跨模块对比 |
| **证据审查与增量复核** | 源码 SHA provenance、质询记录、审查遗留风险、增量 Commit 影响复核 | `15-evidence-review/`<br>`00-meta/agent-research-contract.md` | 本地 Git 工具与 claim 索引 |

---

## 常用本地静态复核工具

工作区提供单进程静态校验脚本（无需启动 vLLM、无需下载模型、无需硬件卡）：
```bash
# 校验 105 篇权威 Manifest、Markdown 相对链接、Claim 索引及源码位置有效性
python3 tools/verify_research.py

# 候选影响扫描：比对指定仓库 commit diff 并匹配受影响 Claim（增量复核）
python3 tools/review_changes.py --repo third_party/vllm --base bb363db9a5ec2edc7b39e99b00af363a89d1fb81 --target HEAD
```

---

## Agent 工具防错与避坑规则（系统级稳定性）

在多轮自动化工具调用或复杂长任务中，为避免工具层抛出异常导致流程挂起或中断，所有在本项目工作的 Agent 必须严格遵守以下执行纪律：

1. **文件读写顺序（Read-Before-Write/Edit）**：
   - 在调用 `edit` 或覆盖写入 `write` 任何现有文件之前，**必须先通过 `read` 工具读取该文件**。禁止凭借历史记忆直接编辑未读取文件，否则会触发系统文件观察策略拦截。
2. **权限参数规范（No Invalid Justification Escalation）**：
   - `justification` 参数**仅且必须**与 `sandbox_permissions`（权限提权申请）成对出现。
   - 在常规 `workspace-write` 许可范围内的标准 `edit`/`write` 操作中，**严禁多余传递 `justification` 参数**，否则会导致系统报错 `invalid escalation: justification is only valid together with sandbox_permissions` 并可能导致交互卡顿。
3. **报错自愈与连续执行**：
   - 若遇到工具报错（如路径错误、内容匹配差异），**严禁停止响应等待用户催促**，必须在同一轮次中立即重新 `read`、校准参数并重试，维持任务连续推进。

---

## 工作区目录结构说明

```text
.
├── .gitmodules                          # 6 大核心第三方仓库 Submodule 映射清单
├── .gitignore                           # 已配置忽略所有 third_party 目录及临时缓存
├── AGENTS.md                            # 全局规范与核心调研指引（唯一全局入口）
├── 20260903-vllm-ai-infra-research/     # 【核心唯一权威库】精确 105 篇全景研究文档
│   ├── 00-meta/                         #   - 阅读地图、证据规则、术语索引、Agent 任务契约附件
│   ├── 01-ai-infra/ ~ 08-production/    #   - vLLM 请求控制、KV、执行、编译、分布式、生产
│   ├── 09-hardware-foundations/         #   - 统一硬件模型
│   ├── 10-ascend/ ~ 13-tpu/             #   - 各加速器厂商深度架构剖析
│   ├── 14-decisions/                    #   - 跨后端技术选型评判矩阵
│   ├── 15-evidence-review/              #   - 源码 SHA 索引与证据审查
│   └── legacy/                          #   - 早期草稿（已被 supersede，仅供追溯）
├── analysis-design/                     # 历史分析设计文档与运行时接口参考（过渡补充）
├── tools/                               # 本地轻量静态校验与增量变更复核脚本
│
├── third_party/                         # 【通用第三方框架，Submodule 管理】
│   └── vllm/                            #   - 上游 vLLM (SHA: bb363db9)
│
├── huawei/                              # 【Huawei Ascend 路线】
│   ├── docs/                            #   - 华为调研文档 (aot, capture-eager, comm)
│   └── third_party/vllm-ascend/         #   - 华为适配代码 Submodule (SHA: 35463578)
│
├── qualcomm/                            # 【Qualcomm Cloud AI 路线】
│   ├── docs/                            #   - 高通调研文档 (aot, capture-eager, comm)
│   └── third_party/vllm-qaic/           #   - 高通适配代码 Submodule (SHA: 3212cc67)
│
├── tenstorrent/                         # 【Tenstorrent 路线】
│   ├── docs/                            #   - TT 调研文档 (aot, capture-eager, comm)
│   └── third_party/                     #   - TT 适配代码 Submodules (vllm-tt-plugin: f6995475, tt-metal: c634b1ca)
│
└── google/                              # 【Google TPU / XLA 路线】
    ├── docs/                            #   - 谷歌调研文档 (aot, capture-eager, comm)
    └── third_party/                     #   - 谷歌适配代码 Submodule (tpu-inference: fd338000)
```

---

## 调研原则与证据规范

1. **事实铁律（Ground Truth First）**：
   - 唯一合法证据源为本地真实源码、官方 Release、头文件/C ABI、官方手册及准确的 Commit/行号。
   - 文档 `15-evidence-review/20260903-152-独立Review问题、修复与剩余风险.md` 记录了 2026-09-03 对 105 篇静态文档的专项审阅，只代表当时文档的静态审阅边界，不代表硬件实测。
2. **严禁无依据推演（Zero Unfounded Speculation）**：
   - 遇到未开源模块（如高通闭源编译器内部、Torch-QAIC 内部 C++、TPU libtpu 内部），必须显式标记为“未知/空缺（`UNKNOWN`）”，不得凭常规逻辑脑补底层行为。
3. **第三方代码只读隔离**：
   - 所有外部仓库集中在对应的 `third_party/` 目录下，禁止随意修改；所有研究产出均归档至对应文档中。
