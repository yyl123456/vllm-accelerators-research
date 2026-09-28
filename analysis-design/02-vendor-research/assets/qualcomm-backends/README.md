# 高通后端调研配图

四张图由 PlantUML 1.2026.7 生成，`.puml` 为可编辑源文件，SVG 用于正文，PNG 用于预览与导入文档。

| 图 | 阅读目的 | 证据入口 |
| --- | --- | --- |
| vllm_component | 两条执行路线与模块职责 | vLLM 篇第 1、2、6 节 |
| vllm_sequence | AoT 提交与完成 | vLLM 篇第 3 节 |
| torch_component | 独立设备后端的功能面及取证范围 | PyTorch 篇第 1、2、8 节 |
| torch_sequence | 融合 RMSNorm 的参数、核内执行与证据缺口 | PyTorch 篇第 5 节 |

图是源码及官方描述的讲解性概括，不是完整调用图。灰色区域表示未取得内部实现，不能将虚线理解为已验证的 SDK 调用链。固定版本与源码位置见正文及 `source-manifest.json`。

渲染：`java -jar plantuml.jar -tsvg *.puml`；预览：将 `-tsvg` 改为 `-tpng`。
