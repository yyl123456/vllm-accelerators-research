# materials/ — 用户资料库

这里专门存放**你收集到的外部资料**：PPT / 演讲、论文、技术报告、博客、数据集、截图等。
它与 `20260903-vllm-ai-infra-research/` 的 105 篇权威文档**分开管理**：资料是线索与背景，权威结论仍以固定 SHA 源码和权威集为准。

> Agent 入口：先运行 `python3 tools/materials.py search "关键词"`，或直接读 [`catalog.json`](./catalog.json) / [`INDEX.md`](./INDEX.md)。

---

## 1. 目录结构

```text
materials/
├── README.md              # 本文件：归档规范
├── INDEX.md               # 自动生成的人类可读总览（勿手改）
├── catalog.json           # 自动生成的机器可读索引（Agent 检索入口，勿手改）
├── slides/                # PPT / 演讲 / 分享
├── papers/                # 论文 / 技术报告 / 白皮书
├── blogs/                 # 博客 / 公众号 / 邮件列表
├── datasets/              # 数据集 / 表格 / 压测结果
├── other/                 # 其他（截图、录音转写等）
└── _inbox/                # 临时落盘区：先放进来，再按第 3 节归档
```

每条资料一个目录，目录名即 **id**，格式 `YYYY-MM-DD-<slug>`：

```text
materials/slides/2026-10-25-deepinfer-talk/
├── meta.json              # 元数据（必需）
├── notes.md               # 阅读笔记：结论、页码、待核验点（强烈建议）
├── extracted.txt          # PDF 抽取的全文（由工具生成）
└── talk2-deepinfer.pdf    # 原始文件（保留原文件名，便于溯源）
```

---

## 2. Agent 如何自动检索

三种方式，任选：

```bash
# 1) 全文检索（标题/摘要/标签/notes.md/extracted.txt 都命中）
python3 tools/materials.py search "EPLB 负载均衡"

# 2) 按类型/标签浏览
python3 tools/materials.py list --type slides
python3 tools/materials.py list --tag KV

# 3) 直接读机器索引
cat materials/catalog.json
```

查看单条资料（含元数据与文件清单）：

```bash
python3 tools/materials.py show 2026-10-25-deepinfer-talk
python3 tools/materials.py show 2026-10-25-deepinfer-talk --notes
```

搜索输出包含 `id`、命中词、片段与目录路径；Agent 拿到路径后应继续读 `notes.md`（结论）与 `extracted.txt`（原文）。

> 自动发现：根目录 [`AGENTS.md`](../AGENTS.md) 已把资料库写入强制路由规则，`.agents/skills/materials-lookup/SKILL.md` 提供检索工作流。新会话的 Agent 会同时看到这两处。

---

## 3. 新增资料（四步）

### Step 1：落盘到 `_inbox/` 或直接建目录

```bash
mkdir -p materials/slides/2026-10-25-my-deck
cp /path/to/deck.pdf materials/slides/2026-10-25-my-deck/
```

命名建议：id 用小写英文/数字/连字符；原始文件保留原名，必要时加前缀区分版本。

### Step 2：写 `meta.json`

最小示例：

```json
{
  "schema_version": 1,
  "id": "2026-10-25-my-deck",
  "title": "资料标题",
  "type": "slides",
  "status": "indexed",
  "added": "2026-10-25",
  "language": "zh",
  "vendors": ["DeepSeek"],
  "topics": ["推理系统", "EPLB"],
  "source": {
    "kind": "google-drive",
    "url": "https://drive.google.com/...",
    "file_url": "https://drive.google.com/file/d/.../view",
    "original_filename": "deck.pdf",
    "retrieved_at": "2026-10-25"
  },
  "source_file": {
    "path": "deck.pdf",
    "sha256": "<运行 sha256sum deck.pdf 得到>",
    "bytes": 123456,
    "pages": 32,
    "mime": "application/pdf"
  },
  "derived": ["extracted.txt", "notes.md"],
  "summary": "2–4 句话说明这份资料讲什么、价值在哪。",
  "key_points": ["要点 1", "要点 2"],
  "related": [{"path": "deepseek/README.md", "note": "对应的工作区文档"}],
  "authority": "secondary",
  "caveats": "二手材料/讲者观点/图表未提取等限制。"
}
```

必填字段：`id`、`title`、`type`、`added`、`summary`、`topics`、`source`、`source_file`。
其中 `id` 必须等于目录名，`type` 必须等于父目录名，`source_file.sha256` 用于校验文件未被替换。

### Step 3：抽取文本 + 写笔记（可选但推荐）

```bash
python3 tools/materials.py extract 2026-10-25-my-deck
```

- 首次运行会把 `pypdf` 下载到 `.cache/libs`（已 git 忽略），不污染系统环境。
- 抽取结果写入 `extracted.txt`，并自动去掉中文之间被 PDF 拆出的空格。
- 图表、公式、扫描件可能无法抽取，此时在 `notes.md` 里手动补关键信息。

`notes.md` 建议结构：一句话结论 → 分节要点（带页码）→ 可复用结论 → 与本工作区的关系 → 待核验清单。

### Step 4：重建索引并校验

```bash
python3 tools/materials.py index
python3 tools/materials.py verify
```

`index` 会重写 `catalog.json` 与 `INDEX.md`；`verify` 会检查元数据、id/type 一致性、源文件 SHA256、派生文件与索引新鲜度。

---

## 4. 证据等级与引用

| 等级 | 含义 | 用法 |
|---|---|---|
| `primary` | 官方文档、源码、标准、Release | 可直接支撑结论 |
| `secondary` | 讲者分享、二手解读、博客 | 提供线索/观点，需交叉验证 |
| `unverified` | 来源不明、无法追溯 | 仅作参考，不进结论 |

引用格式：

```markdown
[材料] 《资料标题》，materials/slides/2026-10-25-my-deck/，p.12
```

**资料不能替代权威证据**：涉及 vLLM / Ascend / QAIC / TT / TPU 的架构与兼容性结论，仍须回到固定 SHA 源码与 105 篇权威文档；资料与源码冲突时，以源码为准并记录差异。

---

## 5. PDF 文本提取说明

`tools/materials.py extract` 使用纯 Python 的 `pypdf`：

- 自动加载 `.cache/libs`，缺失时从 PyPI 下载 wheel 并解压（无需 pip）。
- 只提取文本层；扫描版 PDF 需要 OCR（本工具不内置）。
- 提取文本中的中文空格会被清理，但换行与页码标记保留，便于引用。

---

## 6. 当前清单

| id | 类型 | 标题 | 来源 |
|---|---|---|---|
| `2026-10-25-deepinfer-talk` | slides | DeepSeek V4.1 推理系统拆解：DeepInfer 开源代码与部署 Q&A 解读 | [Google Drive](https://drive.google.com/file/d/1ZKrCY7Q_RAhPXJ1Qy0WGxWClWIEy-wua/view) |

完整列表见 [`INDEX.md`](./INDEX.md)。

---

## 7. 常见问题

**Q：资料很多，目录会不会乱？**
A：按 `type` 分五类，每条资料一个 id 目录；检索走 catalog，不依赖目录层级。

**Q：能放视频/录音吗？**
A：可以，放 `other/` 或 `slides/`，在 `meta.json` 里注明时长/转写文件；大文件建议只存链接与摘要，避免仓库膨胀。

**Q：第三方代码为什么不能放这里？**
A：第三方仓库统一走 `third_party/` submodule 只读管理；`materials/` 只放"人读的资料"。

**Q：改了 meta 或新增文件后忘记 index 会怎样？**
A：`verify` 会报 `catalog.json 过期`，按提示运行 `index` 即可。
