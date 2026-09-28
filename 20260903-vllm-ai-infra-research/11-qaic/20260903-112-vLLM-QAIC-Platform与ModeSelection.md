# vLLM QAIC Platform 与 Mode Selection

## 文档契约

- **唯一问题/Owner**：plugin如何激活QAIC并在PyTorch eager与AoT/QPC路径间选择。
- **依赖输入**：70-72 plugin contracts；111 software modes。
- **唯一输出**：entry point、QaicPlatform、Worker/Runner mode selection链。
- **Out of scope**：两种mode内部（113/114）、scheduler/sampling（115）。
- **源码基线**：vllm-qaic `3212cc670130b7e5290b429f781dc978d7ecf430`，其pyproject开发依赖固定vLLM `v0.23.0`。

`setup.py`注册platform entry `qaic = vllm_qaic:register`，另以general plugin注册KV connector。`register()`返回QAIC platform class；`QaicPlatform`负责device/config、attention、worker/executor和mode相关选择。

源码存在`QaicWorkerPyt/QaicModelRunnerPyt`与`QaicWorkerAoT/QaicModelRunnerAoT`两条明确路径；它们共享vLLM Scheduler/Core接口，但execution ownership不同。Pyt路径以PyTorch/custom ops处理动态batch；AoT路径加载预编译profiles/QPC并映射请求。

mode是deployment/model级重大决策，不应在请求失败时随意切换：weights格式、KV ownership、supported models/features、memory profile、artifact和sampling都可能不同。config validation需在startup给出唯一effective mode与所选classes。

`QaicUniProcExecutor`说明固定SHA至少有自定义单进程执行路径；多device/distributed能力需另看communicator和AoT profile，不从class名外推。

验证分discovered/imported/selected/device-ready/mode-ready；故意缺SDK/QPC、错SKU/unsupported model应早失败。`SOURCE_IMPLEMENTED`证明双mode classes存在；`UNKNOWN`为固定SHA在实际SDK/device上的完整feature coverage。

## 实际 selection predicate

固定 `platform_base.py::QaicPlatform` 以 Torch QAIC 是否安装确定 class-level `is_aot`，并据此选择 `QaicWorkerAoT` 或 `QaicWorkerPyt`；`check_and_update_config()` 又正规化 `enforce_eager`、async、spec、P/D、vision 等组合。mode 不是单由 QPC 文件是否存在决定。

```text
package/runtime availability -> is_aot -> worker class
 -> config guards/effective mode -> runner/session init -> mode-ready
```

Pyt mode 明确拒绝 disaggregated serving/spec，并关闭 async scheduling；AoT 也有 vision/feature guards。`SOURCE_IMPLEMENTED`：`platform_base.py` 和 worker classes；`RECOMMENDATION`：记录 predicate inputs/effective mode；`UNKNOWN`：安装组合与真实 SDK ABI。
