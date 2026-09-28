# Plugin 发现、激活与 Import 语义

## 文档契约

- **唯一问题**：OOT package 何时被发现、import、执行，并如何成为唯一 current platform。
- **In scope**：Python entry points、plugin groups、allowlist、lazy initialization、多进程重复加载、冲突。
- **Out of scope**：Platform职责（71）、版本漂移（74）、安全威胁细节（85）。
- **证据基线**：vLLM `bb363db9...` 的 `vllm/plugins/__init__.py`、`platforms/__init__.py` 与各插件 packaging metadata。

vLLM通过 `importlib.metadata.entry_points` 发现 groups。platform group为 `vllm.platform_plugins`；general、I/O processor、stat logger、endpoint另有不同进程域和加载策略。`VLLM_PLUGINS` 可作为allowlist；endpoint因扩大HTTP攻击面默认要求显式allowlist，而其他groups默认可能加载全部发现项。

platform plugin entry point加载后通常返回可判定当前环境的platform class path/factory。`current_platform` 首次访问时lazy解析；因此任何更早读取platform的import都可能触发插件代码。若多个platform同时激活或没有唯一winner，应明确失败，不能依赖entry-point遍历顺序。

general/platform plugins会在API、core和worker等多个进程分别加载。plugin initialization必须幂等、不能假设process 0已有全局状态，也不能在discovery阶段提前建立不可fork的device context。import exception当前可能被记录后继续，这意味着“已安装”不等于“已激活”。

验证要分别检查distribution metadata、entry point discovery、allowlist、factory返回、唯一platform、每进程日志和最终class；还要覆盖两个插件冲突、editable/非editable安装、namespace污染和spawn workers。结论必须区分 discovered/imported/selected/initialized/executed 五级状态。

## 固定源码调用链

`vllm/plugins/__init__.py::load_plugins_by_group()` 枚举并加载 group；`vllm/platforms/__init__.py::resolve_current_platform_cls_qualname()` 汇总 probes并解析唯一平台；`current_platform` 是 lazy proxy。

| 状态 | 可证明内容 | 不能证明 |
|---|---|---|
| discovered | distribution metadata可见 | import成功 |
| imported | module执行未抛错 | 被选中 |
| selected | current platform class唯一 | device ready |
| initialized | worker/runtime建立 | model可执行 |
| executed | 某路径运行 | 全功能/生产支持 |

冲突、allowlist 拼写、子进程 distribution path 和 import side effects 都应可归因。`SOURCE_IMPLEMENTED`：固定 loader/resolver；`RECOMMENDATION`：输出五级状态与 distribution/version；`UNKNOWN`：vendor package 在所有 launcher 下一致加载。
