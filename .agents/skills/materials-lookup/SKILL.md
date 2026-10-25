---
name: materials-lookup
description: 用户资料库（materials/）检索、引用与归档。当任务涉及“我找到的资料 / PPT / 演讲 / 论文 / 博客 / 外部参考”，或需要为技术结论补充外部线索、核对某份材料时触发。
---

# 资料库检索与归档（Materials Lookup）

`materials/` 是用户收集的外部资料的**唯一归档目录**，与 `20260903-vllm-ai-infra-research/`（105 篇权威集）和 `third_party/`（只读源码）分离管理。资料是线索与背景，不是架构结论的最终证据。

## 何时使用

- 用户提到“我找到的资料 / 之前的 PPT / 那个演讲 / 某篇论文 / 外部参考 / 材料里怎么说”。
- 需要为现有技术结论补充产业视角、实现线索或反例。
- 需要核对某份资料的具体页码/原话。
- 用户把新文件放进 `materials/_inbox/`，要求归档。

## 检索流程（按顺序）

1. **先查索引，不要凭记忆回答**：

   ```bash
   python3 tools/materials.py search "<关键词>"          # 中英文均可
   python3 tools/materials.py search "<关键词>" --json   # 需要结构化结果时
   ```

2. **确认条目并读内容**：

   ```bash
   python3 tools/materials.py show <id>          # 元数据 + 文件清单
   python3 tools/materials.py show <id> --notes  # 直接看笔记
   ```

   然后按需读 `materials/<type>/<id>/notes.md`（结论、页码、待核验项）与 `extracted.txt`（原文）。图表/公式可能在提取文本中缺失，必要时用 `read_image` 看原始文件里的截图。

3. **引用时带路径与页码**：

   ```markdown
   [材料] 《DeepSeek V4.1 推理系统拆解》，materials/slides/2026-10-25-deepinfer-talk/，p.24
   ```

4. **没命中时**：明确说“materials/ 中未检索到”，并给出 1–2 个建议关键词或建议归档；不要编造资料内容。

## 证据边界（重要）

- 资料默认 `authority: secondary`：可支撑“某人/某团队这样实现/这样认为”，不能单独支撑“vLLM / Ascend / QAIC / TT / TPU 的架构事实”。
- 架构与兼容性结论必须回到固定 SHA 源码与 105 篇权威集；资料与源码冲突时，以源码为准，并把差异记录到资料的 `notes.md` 或相关文档。
- 讲者实测数字（如 EPLB +2%、GDR 开销）要保留测试条件，不可外推。

## 归档新资料

1. 在 `materials/<type>/<YYYY-MM-DD-slug>/` 放原始文件（`type` ∈ slides/papers/blogs/datasets/other）。
2. 写 `meta.json`（必填字段见 `materials/README.md`），`id` 必须等于目录名，`source_file.sha256` 用 `sha256sum` 生成。
3. PDF 运行 `python3 tools/materials.py extract <id>`；写 `notes.md`（结论 + 页码 + 待核验项）。
4. 运行：

   ```bash
   python3 tools/materials.py index
   python3 tools/materials.py verify
   ```

## 常见错误

- 只读 `catalog.json` 的摘要就下结论：摘要只有 1–2 句，必须读 `notes.md` / 原文。
- 把资料当权威引用：资料只能作为 `secondary` 证据。
- 新增资料后忘记 `index`：`verify` 会报 `catalog.json 过期`。
- 改 `meta.json` 后没更新 `source_file.sha256`：`verify` 会报哈希不匹配。
