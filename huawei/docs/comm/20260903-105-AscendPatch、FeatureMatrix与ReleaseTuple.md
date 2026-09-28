# Ascend Patch、Feature Matrix 与 Release Tuple

## 文档契约

- **唯一问题/Owner**：固定plugin revision对upstream修改了什么，如何把功能声称约束到可复现release。
- **依赖输入**：74 patch方法；86 release tuple；102-104实现地图。
- **唯一输出**：patch debt、feature evidence matrix和发布门禁。
- **Out of scope**：重复解释单项机制。
- **源码基线**：vllm-ascend `3546357838389aa7201d9b44e234f831e98b2fd4`；vLLM `bb363db9...`。

## 1. Patch surface

固定SHA的 `vllm_ascend/patch/` 覆盖distributed、scheduler balance/profiling、KV coordinator/utils、structured output、spec/PP、model-specific behavior、fused MoE、block/input batch、Triton/UVA等。patch数量与广度意味着兼容性不能只看Platform/Worker public interfaces。

每项patch需ledger：upstream target/symbol、原因、apply order、依赖、测试、移除条件。网页main `patch/__init__.py` 甚至记录将upstream weight-transfer factory的`nccl`键替换成HCCL实现的方案，说明配置字符串与真实backend可能不一致；必须以固定SHA实际内容为准。

## 2. Feature matrix维度

行应为model/task或基础能力，列至少含SKU(A2/A3/310P/950)、engine/runner、dtype/quant、TP/PP/DP/EP/CP、prefix/spec/MM/LoRA/grammar/P-D、graph。单元不是勾选，而是事实来源类型与验证成熟度的二元组，例如 `SOURCE_IMPLEMENTED × STATIC_REVIEWED`、对应来源类型 `× DEVICE_NUMERIC`、`RELEASE_CONTRACT × PRODUCTION_QUALIFIED`，并附 tuple 链接；字典见 02。

官方安装页当前列出多产品images并要求整行version compatibility；这证明release配套原则，不证明固定SHA支持网页所有当前features。release notes中的历史“V0 only/V1 soon”也只能归属对应release时间点。

## 3. 完整Tuple

包括hardware product/firmware/driver、CANN、PyTorch/TorchNPU、Triton Ascend/ATB、vLLM/plugin SHA、wheel device type/container、model/quant、parallel/features/env/patch digest。HCCL与graph相关环境也需记录。

## 4. 发布门禁

static patch apply/import → device init/op/collective → model greedy numeric → feature组合 → failure/soak → performance/SLO。任何patch无法定位或继承默认漂移视为P0。unsupported组合早拒绝。

## 5. 当前结论

- `SOURCE_IMPLEMENTED`：插件是广泛custom worker/runner/backend加patch的OOT集成。
- `VENDOR_CLAIM × UNVALIDATED`：150 的 `WEB-ASCEND-01` 要求 release components 成套，不任意混版；若具体 release matrix 对版本作出承诺，才提升为 `RELEASE_CONTRACT`。
- `UNKNOWN`：本轮没有Ascend硬件、driver/CANN环境，不能把source coverage提升为设备支持或性能结论。
