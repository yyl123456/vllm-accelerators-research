# 源码证据与图件

`op-plugin-743b073/` 是 torch_npu 提交 `b262bae21efdcccc31c28fbb96ed85c6020df943` 的 `third_party/op-plugin` gitlink 指定提交 `743b073b88a99a505ab7086823376ff15f0316be` 中的原始文件子集，保留原文件行号与 LICENSE。

来源：`https://gitcode.com/ascend/op-plugin.git`。取得日期：2026-09-21。通过独立临时 Git 仓库 fetch 指定提交，再使用 git show 提取；未修改厂商 checkout，也未使用工作区另一个版本的 op-plugin 作为最终证据。文件 SHA-256 见 `source-manifest.json`。

`torch_npu_pipeline_architecture.puml` 为重写后的全景架构图，配套 SVG/PNG 为渲染结果。它标出四个职责阶段和三类 Host 线程，并将设备执行与 Host 对象回收画成并行分支。

`torch_npu_eager_sequence.puml` 是初版的补充时序图，保留供对照。两图中 CANN API 之后的 Driver/Firmware/硬件执行均为未展开的边界，不代表已经验证其内部实现。
