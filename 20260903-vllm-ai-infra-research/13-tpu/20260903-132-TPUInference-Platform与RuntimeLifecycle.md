# TPU-Inference Platform 与 Runtime Lifecycle

## 文档契约

- owner：vLLM 选择 TPU platform 后的 config、executor、worker 和 device lifecycle。
- 依赖输入：`70`–`74`、`131`。
- 唯一输出：固定 SHA `fd33800041510b957cd2da6199742cce2b5fd113` 的可达启动链。
- 不讨论：runner 内部数据或 compile bucket；由 `133`、`134` 持有。

## 发现方式

与典型 OOT platform entry point 不同，固定 vLLM revision `vllm/platforms/__init__.py` 在检测到 `tpu_inference` 包时直接返回 `tpu_inference.platforms.tpu_platform.TpuPlatform`；`vllm/platforms/tpu.py` 也代理该实现。`tpu-inference/pyproject.toml` 在固定 SHA 中仅有 license header，没有 entry-point 表。因此不能把 TT/Ascend 的发现机制复制到 TPU 文档。

## Platform contract

`TpuPlatform` 声明 `device_type='tpu'`、`dispatch_key='XLA'`、Ray resource key `TPU`、可见性变量 `TPU_VISIBLE_CHIPS`。`set_device()` 是 no-op，因为 JAX/libtpu 管理设备；`mem_get_info()` 聚合 `jax.local_devices()` 的 memory stats；pin memory 不支持；per-request seed 被 `validate_request()` 拒绝。

## 配置改写顺序

`TpuPlatform.check_and_update_config()`：

1. 解析 multiprocess DP 与 SPMD/Pathways 互斥。
2. 初始化 `ShardingConfigManager`。
3. 检查 MLA、hybrid prefix/spec、multimodal normalization 等组合。
4. 根据 Pallas attention backend 选择 KV page/block size。
5. 设置 `TPUWorker`。
6. 单 host 选择 uni/multiproc；multi-host 可选择 Ray executor V1/V2。
7. 限制 KV connector、continue-decode 与 PP/pooling 组合。

自动改写意味着用户 desired config 与 effective config 可能不同，必须在启动 artifact 中同时保存。

## Worker lifecycle

```text
TPUWorker.init_device
 -> establish visibility/distributed context
 -> select jax devices and topology order
 -> construct TPUModelRunner(mesh/devices)
 -> load_model
 -> determine_available_memory
 -> initialize KV cache
 -> compile_or_warm_up_model
 -> execute_model / sample_tokens
```

`determine_available_memory()` 使用 JAX HBM usage、`gpu_memory_utilization`、model/runtime/transfer buffers 估算可供 KV 的空间。这里沿用了上游参数名 `gpu_memory_utilization`，语义实际是 TPU HBM budget；监控与文档必须避免字面误读。

## 故障边界

- JAX 初始化通常是进程级、顺序敏感的；fork/import 顺序错误可能早于 model load 失败。
- `jax.devices()` 的 global devices 与 `jax.local_devices()` 的 host-local devices 不同。
- unknown multihost backend 当前 warning 后选择 uni；生产门禁应判断是否符合 operator intent。
- runtime/PJRT fatal 或 host loss 通常影响 worker/engine，不应假装为单 request 可重试。

## 证据分类

- `SOURCE_IMPLEMENTED`：上述 platform/worker/config symbols 与固定 SHA。
- `INFERENCE`：JAX/PJRT 初始化扩大进程故障域。
- `RECOMMENDATION`：readiness 分成 platform-selected、devices-ready、model-loaded、compiled 四级。
- `UNKNOWN`：本轮未执行 TPU startup 和 recovery。

