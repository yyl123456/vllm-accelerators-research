# PlatformInterface 职责与稳定性

## 文档契约

- **唯一问题**：Platform class 在 vLLM 中裁决哪些全局策略，以及哪些接口是脆弱适配点。
- **In scope**：device/dispatch/distributed/attention/quant/config validation/worker selection/compile hooks/capabilities。
- **Out of scope**：plugin discovery（70）、custom runner（72）、vendor实现。
- **证据基线**：vLLM `bb363db9...` 的 `platforms/interface.py` 与内建platforms。

Platform 是策略聚合点：`device_type/name/capability`、PyTorch dispatch key、Ray resource/env、device visibility、distributed backend、supported dtypes/quantization、attention selection、compile/pass hooks、worker/executor class、config/input validation、device communicator与topology probes。

接口中既有stateless方法，也有会触碰当前device/runtime的方法。config构造早期只能调用stateless能力；若插件在该阶段查询device，会破坏多进程启动或无设备的frontend。方法默认实现并不自动适合OOT：例如CPU fallback dispatch key可能只帮助Python registration，不代表目标设备能执行CPU op。

稳定性风险来自上游新增方法、签名/返回语义变化、默认值改变和调用时序变化。Python继承让旧插件仍能import，却可能继承不正确默认实现，这是比显式AbstractMethodError更危险的兼容失败。

插件应提供机器可读capability predicates，而非散落环境判断；对不支持组合在config validation早期拒绝。验证使用interface conformance、调用序列trace、上游版本矩阵和negative features。**结论**：Platform选中只建立policy入口，不证明Executor、op或模型coverage。

## 固定源码职责簇

`vllm/platforms/interface.py::Platform` 可分为静态 identity、device/runtime、attention/quant/op、distributed/communicator、compile hooks、config/request validation、worker/executor routing。内建 platforms 展示差异，但不是 OOT 默认答案。

| 调用阶段 | 允许依赖 | 高风险副作用 |
|---|---|---|
| config/frontend | package与静态 config | 初始化 device/global runtime |
| engine bootstrap | resolved topology/env | 修改全局状态不可回滚 |
| worker init | local device/runtime | 二次解释 visible IDs |
| request validation | effective config/model | 隐式加载/编译 |

新增非 abstract 方法会让旧插件继承可能错误的默认行为。`SOURCE_IMPLEMENTED`：interface 与 builtin overrides；`RECOMMENDATION`：升级时生成 override/default diff；`UNKNOWN`：未 override default 的动态可达性。
