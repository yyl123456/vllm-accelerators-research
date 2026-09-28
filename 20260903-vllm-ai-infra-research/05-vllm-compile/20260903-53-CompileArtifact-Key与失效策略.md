# Compile Artifact Key 与失效策略

## 文档契约

- **唯一问题**：何时两个执行环境可以安全复用同一编译产物。
- **In scope**：cache key inputs、content hashes、serialization、atomic save、load validation、invalidation/GC。
- **Out of scope**：pass语义（51）、capture buffers（52）、release组合总论（142）。
- **依赖输入**：43 的模型 identity；50/51 的 graph/pass identity；71 的 platform target。
- **唯一输出/Owner**：semantic/binary artifact key、发布、加载、失效与 GC 合同。
- **相邻篇不得重述**：86/vendor release docs 只引用本篇 key，不另建 artifact 定义。
- **证据基线**：vLLM `bb363db9...` 的 `compilation/caching.py`、`compiler_interface.py`、decorator AOT load/save。

## 1. artifact 是 executable ABI

key 至少绑定：model/code revision与有效 config、weights/quant format中影响图的部分、vLLM/compiler/framework版本、passes及其source/order、target device architecture、driver/runtime ABI、dtype、parallel layout、dynamic guards/profiles、custom op binaries和关键 flags。

漏 key 会 stale reuse 并可能静默错；过度 key 只会降低命中。correctness 优先于 cache rate。

## 2. 两层 identity

推荐分离 semantic graph key 与 target binary key。前者描述模型/变换语义，后者再绑定设备 toolchain/ABI。这样可定位 miss 来源，也避免把跨设备可移植 IR 与不可移植 binary 混为一谈。

## 3. 原子发布

多 worker/进程并发编译时，artifact 应写临时对象、校验完整后 atomic publish；reader不能看到 partial file。metadata含 schema version、content digest、producer tuple。load后仍需验证 guards、symbols/custom ops和目标设备。

## 4. invalidation

显式配置变化可 key miss；不可见环境变化需 version/ABI probe。corruption应 quarantine并重建，而非反复 crash。GC不能删除正被 mmap/load/replay 的产物；应有引用或generation目录。

## 5. 安全

编译产物是可执行代码，shared cache需权限、来源和完整性控制；不可信租户不能注入可加载 artifact。日志不能泄露包含凭据的环境或远程模型地址。

## 6. 固定源码 cache identity

`vllm/compilation/compiler_interface.py` 区分 compiler adapter，并描述 Dynamo bytecode 与 Inductor shape compilation/cache；`vllm/ir/op.py` 的 `compute_hash()` 将 op state 分别影响 vLLM compile cache 与 AOTAutograd/Inductor caches；`InductorPass.uuid()` 也进入代码 cache identity。

最小 key closure 为 model/weights、graph/ops/passes、torch/vLLM/compiler/runtime、device arch、dtype/quant、shape guards、parallel/topology 与 feature modes。artifact 发布必须先写临时对象、校验 manifest/hash，再原子切换引用；不能就地覆盖活跃 worker 正在 mmap/load 的文件。

`SOURCE_IMPLEMENTED`：cache/hash/adapter interfaces；`RECOMMENDATION`：manifest 记录 key 每一维和 build provenance；`UNKNOWN`：跨 host/driver 的 binary portability。

## 7. 验收

逐个扰动 key因素确认应 hit/miss；并发 writer/crash；旧 schema/坏 digest；cross-device拒绝；cold/warm启动时间和bytes。固定 revision 含 hash/serialization机制不代表每个第三方编译器都覆盖全部因素。
