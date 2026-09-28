# vLLM TT Plugin 与自定义 Engine

## 文档契约

- owner：插件发现、平台注册、配置改写与 engine/worker 注入。
- 依赖输入：`70`–`74` 的 vLLM plugin contract，`122` 的 TT 栈。
- 唯一输出：固定 revision 的可达集成调用链。
- 不讨论：调度算法或 trace；由 `124`、`125` 持有。

## 固定 revision 与入口

插件固定为 `f6995475739201fd1d2883adec477ecb37f951d9`。`pyproject.toml` 注册 general plugin `tt_model_registry = vllm_tt_plugin.entrypoints:register` 与 platform plugin `tt = vllm_tt_plugin.entrypoints:platform_plugin`。前者触发 TT model registry；后者返回 `vllm_tt_plugin.platform.TTPlatform`。

## 启动调用链

```text
Python entry points
  -> entrypoints.register / platform_plugin
  -> register_tt_models_from_plugin
  -> TTPlatform.check_and_update_config(VllmConfig)
  -> select TTWorker
  -> select TTScheduler or TTLaneCoordinator
  -> optional TTCoreEngineLauncher
  -> TTWorker.init_device/load_model/determine_available_memory
  -> TTModelRunner.execute_model/sample_tokens
```

`TTPlatform.check_and_update_config()` 不是简单的 device name hook。它 pin runner mode、注册模型、检查 model capabilities、改写 scheduler/worker、约束 prefix caching、device sampling、async decode、DP lanes，以及 block-output model 的 admission contract。因此升级 vLLM 时，最大风险是这一函数依赖的 config 字段与生命周期顺序变化。

## 配置的 desired/effective 差异

| 用户请求 | effective 行为 |
|---|---|
| prefix cache | 模型未声明或 sliding-window 不兼容时自动关闭并 warning |
| async scheduling | 模型未声明 `supports_async_decode` 时自动关闭并 warning |
| device sampling | 显式请求而模型不声明能力时 fail closed |
| block-output + 不支持组合 | startup `ValueError`，而不是运行时降级 |
| DP single-execute model | 可能改写为进程内 TT lanes |

这要求运维同时记录原始参数和 effective config；否则一次自动降级会被误报为已支持功能。

## 进程级状态与故障域

`TTPlatform` 在 class level 保存 admission 相关状态。固定源码对 block-output model 明确限制同一进程的 engine 共存，并在配置异常时恢复 previous handle。该设计把进程变成资源/一致性边界：测试必须覆盖失败启动后的重试、旧 engine 仍存活、GC/traceback 持有引用等路径。

## 兼容面

真正的兼容 tuple 至少包括 vLLM SHA、plugin SHA、tt-metal SHA、runtime/firmware、model generator、模型权重格式、mesh 和配置模式。entry point 成功加载只证明 discovery；worker 初始化只证明部分 runtime；首 token 正确也未覆盖长上下文、并发、abort、trace replay 或数值稳定性。

## 证据分类

- `SOURCE_IMPLEMENTED`：上述 entry point、`TTPlatform`、`TTWorker`、`TTModelRunner` 和 launcher 符号。
- `INFERENCE`：`check_and_update_config` 是升级的高耦合点。
- `RECOMMENDATION`：将 config diff 和 capability decisions 输出为结构化启动证据。
- `UNKNOWN`：固定 tuple 的设备启动与压力结果。

