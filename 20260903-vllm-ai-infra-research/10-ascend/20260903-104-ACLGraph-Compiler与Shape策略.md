# ACLGraph Capture、算子内存与 Shape 策略

## 文档契约

- **问题/Owner**：Ascend 如何捕获/复用稳定执行；capture 期间算子内存由谁分配、路由和释放。
- **端到端边界**：vLLM-Ascend → torch_npu/OpPlugin → ACL/CANN Runtime → Driver HAL。
- **Out of scope**：闭源 `libopapi` 内部 executor/tiling 实现、内核内部片上存储分配、固件和 KMD 的物理页细节。
- **分析日期**：2026-09-10。

## 结论

**ACLGraph capture 的整体编排者是 `torch_npu::NPUGraph`，但不存在一个包办所有 capture 内存的单一管理器。**

1. **图内输出 Tensor、临时 Tensor 等 PyTorch device allocation**：由 `torch_npu::NPUCachingAllocator` 管。`NPUGraph::capture_begin` 在调用 CANN 的 `aclmdlRICaptureBegin` **之前**先建立 graph `PrivatePool` 路由；此后匹配 capture stream 的分配进入该私有池。capture 结束只停止路由，不释放池；直到 `NPUGraph::reset`/析构时才降低池引用计数，且只有图不再使用、Tensor 引用也释放后，缓存块才可真正归还。
2. **ACLNN 算子的 workspace**：由 OpPlugin 先调用 `aclnnXxxGetWorkspaceSize` 决定大小，再由 torch_npu 分配地址并传给 `aclnnXxx(workspace, size, executor, stream)`。当前 ACLGraph 明确禁止 `TASK_QUEUE_ENABLE=2`，而 OpPlugin 仅在值为 2 时选择 V2，因此 **capture 的真实路径是 V1 → `NPUCachingAllocator` → graph `PrivatePool`**。独立的 `NPUWorkspaceAllocator` 是 V2 路径，不参与当前 `NPUGraph` capture。
3. **CaptureModel、TaskInfo、capture stream、Event/Notify、SQ/CQ 等控制面资源**：由 CANN Runtime 的 `Context`、`Stream`、`CaptureModel` 管理。它们是“捕获任务序列”的资源，不等同于 PyTorch Tensor/ACLNN workspace 内存池。
4. **最终新增 HBM/DDR allocation**：torch_npu 缓存未命中时才调用 `aclrtMallocAlign32`；ACL 进入 Runtime `DevMalloc`，Runtime 选择页策略并调用 Driver HAL `halMemAlloc`。所以 CANN/Driver 执行底层分配，但不管理 PyTorch graph pool 的复用和生命周期。
5. **vLLM-Ascend 是策略层**：它决定何时按 `BatchDescriptor` capture/replay、是否让多个图共享全局 graph pool、持有哪些输入/输出引用；它不直接分配底层 HBM。

最准确的简答是：**torch_npu 负责 capture 内存的上层编排和地址稳定性；当前 ACLGraph 的 Tensor 和 ACLNN workspace 都由 `NPUCachingAllocator::PrivatePool` 管，CANN Runtime 负责被捕获的任务模型和执行资源，Driver 负责最终设备内存映射。**

## 一次 capture 的完整流程

### 1. vLLM 选择图与共享池

`ACLGraphWrapper` 按 `BatchDescriptor` 缓存一个 `ACLGraphEntry`。首次命中创建 `torch.npu.NPUGraph`，在 `torch.npu.graph(aclgraph, pool=global_graph_pool)` 中真正跑一次 `runnable`；后续同 key 直接 replay。源码明确标注 capture 中的 `output` 由 PyTorch ACLGraph pool 管理：[vLLM-Ascend `acl_graph.py` L147-L202](https://github.com/vllm-project/vllm-ascend/blob/3546357838389aa7201d9b44e234f831e98b2fd4/vllm_ascend/compilation/acl_graph.py#L147-L202)。

vLLM 自己维持稳定输入地址，并决定强/弱引用输出。Graph 捕获的是地址，不会在 replay 时重新创建普通输出 Tensor；因此输入、输出、KV、block table 和自定义 workspace 的地址生命周期必须覆盖所有 replay。

### 2. torch_npu 先打开私有池路由，再让 CANN 进入 capture

`NPUGraph::capture_begin` 的关键顺序是：

```text
选择/创建 MempoolId
  → NPUCachingAllocator::beginAllocateToPool(device, pool, stream_filter)
  → HostAllocator::begin_allocate_to_pool(pool, filter)
  → aclmdlRICaptureBegin(stream, mode)
  → markCaptureBegin(device)
  → aclmdlRICaptureGetInfo() 取得 model_ri/capture_id
```

原代码特意说明 `beginAllocateToPool` 必须先于 `AclmdlRICaptureBegin`，避免 capture 状态与 allocator 并发 free 的窗口：[torch_npu `NPUGraph.cpp` L284-L329](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUGraph.cpp#L284-L329)。这说明 **graph pool 不是 CANN 在 capture begin 时创建的；它是 torch_npu 在进入 CANN capture 前建立的 allocator 路由规则。**

ACL 接口本身只是把 `aclmdlRICaptureBegin` 下沉为 `rtStreamBeginCapture`：[CANN Runtime `model_ri.cpp` L49-L59](https://gitcode.com/cann/runtime/blob/fbbfae1f25c3b7e0699959df38b0d33a4d5c2a81/src/acl/aclrt_impl/model_ri.cpp#L49-L59)。官方文档也明确说 Begin/End 之间的任务不会立即执行，而是暂存在模型运行实例中：[昇腾 `aclmdlRICaptureBegin` API 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/appdevgapi/aclcppdevg_03_1782.html)。

### 3. 算子先创建输出，再申请 workspace，再提交任务

```text
PyTorch/OpPlugin 创建输出 Tensor
  → aclnnXxxGetWorkspaceSize(inputs, outputs, &workspace_size, &executor)
  → torch_npu 分配 workspace
  → aclnnXxx(workspace_addr, workspace_size, executor, acl_stream)
  → CANN Runtime 将 kernel/memcpy/event 等任务记录到 CaptureModel
```

V1 宏通过 `unsafe_empty_workspace(workspace_size)` 分配 workspace，再把指针传给算子执行 API：[OpPlugin `op_api_common.h` L343-L403](https://gitcode.com/Ascend/op-plugin/blob/90e9f78a68e7b30d743bce859e0a12d960b32ad2/op_plugin/utils/op_api_common.h#L343-L403)。该 overload 使用 `NPUCachingAllocator::get()->allocate`：[torch_npu `OpPreparation.cpp` L413-L429](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/framework/utils/OpPreparation.cpp#L413-L429)，所以在 capture stream 上会被自动路由进 graph 私有池。

`EXEC_NPU_CMD` 仅当 `TASK_QUEUE_ENABLE == 2` 时选择 V2，否则选择 V1：[OpPlugin `op_api_common.h` L485-L493](https://gitcode.com/Ascend/op-plugin/blob/90e9f78a68e7b30d743bce859e0a12d960b32ad2/op_plugin/utils/op_api_common.h#L485-L493)。与此同时，`NPUGraph::capture_begin` 明确检查并拒绝值 2：[torch_npu `NPUGraph.cpp` L240-L247](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUGraph.cpp#L240-L247)。所以不能把 V2 的 per-stream `NPUWorkspaceAllocator` 当作当前 ACLGraph capture 内存路径。

| 内存 | 大小由谁决定 | 地址由谁分配/复用 | capture 后谁持有 |
|---|---|---|---|
| 输出/临时 Tensor | PyTorch op meta/实现 | `NPUCachingAllocator::PrivatePool` | Tensor 引用 + graph pool |
| ACLNN workspace V1 | `aclnnXxxGetWorkspaceSize` | `NPUCachingAllocator::PrivatePool` | graph pool |
| ACLNN workspace V2 | `aclnnXxxGetWorkspaceSize` | 每-stream `NPUWorkspaceAllocator` | 当前 ACLGraph 禁止该模式，不属于 capture 路径 |
| executor/tiling host 临时对象 | 闭源 Op API + OpPlugin glue | Op API/线程本地 huge-mem 机制 | 调用后 release/uninit；内部细节不可见 |
| Task/SQE/CaptureModel | CANN Runtime | Runtime object/task/resource managers | CaptureModel |
| SQ/CQ、Notify、Event | CANN Runtime/Driver | Runtime + Driver | CaptureModel/stream 生命周期 |

### 4. allocator 如何把 capture 分配导入 PrivatePool

`beginAllocateToPool` 创建或增加 `PrivatePool` 引用，并向 `allocation_scopes_` 加入 `(mempool_id, stream_filter)`：[torch_npu `NPUCachingAllocator.cpp` L2218-L2292](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L2218-L2292)。

每次 `malloc` 根据当前 stream 调用 `get_pool`。匹配 filter 就选择该 `PrivatePool` 的 small/large block pool，否则走普通全局池：[torch_npu `NPUCachingAllocator.cpp` L2685-L2708](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L2685-L2708)。池内先复用缓存块；没有合适块才调用 `AclrtMallocAlign32`：[torch_npu `NPUCachingAllocator.cpp` L2984-L3032](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L2984-L3032)。

PrivatePool 的作用是图仍活着时不把其空闲块混回普通 eager pool，避免固定地址被无关 Tensor 占用。多个图可共享一个 `MempoolId`，但调用方必须保证它们不会并发复用冲突地址。replay 复用 capture 时固化的地址，不再执行 Python allocator 流程。

#### Segment 与 Block 的准确含义

`Segment` 不是 `PrivatePool`，也不是整张 ACLGraph 预先规划的一块内存。`SegmentInfo` 在 allocator 接口中被定义为“一次连续的底层 malloc”，在当前 NPU 非 expandable 路径中对应一次 `AclrtMallocAlign32` 返回的连续地址范围：[torch_npu `NPUCachingAllocator.h` L86-L110](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUCachingAllocator.h#L86-L110)、[torch_npu `NPUCachingAllocator.cpp` L2931-L3035](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L2931-L3035)。

一次 Tensor/workspace 请求先成为一个 `Block`；如果没有可复用 Block，allocator 按规则选择底层 segment 大小：不超过 1 MiB 的请求使用 2 MiB segment，1–10 MiB 的请求通常使用 20 MiB segment，更大的请求按 2 MiB 向上取整：[torch_npu `NPUAllocatorConfig.h` L16-L18](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUAllocatorConfig.h#L16-L18)、[torch_npu `NPUCachingAllocator.cpp` L2771-L2778](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L2771-L2778)。segment 可通过 `prev/next` 拆成已分配 Block 和剩余空闲 Block：[torch_npu `NPUCachingAllocator.cpp` L201-L260](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L201-L260)、[拆分实现 L1428-L1464](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L1428-L1464)。

因此一个 `PrivatePool` 最终可能持有零个、一个或多个 segment，取决于 capture 前是否已有可复用块以及 capture 中的峰值内存布局；“多个”不是强制数量。snapshot 以 `prev == nullptr` 的 head block 作为 segment 起点，沿 `next` 汇总其中所有 Block，直接证明了 `Segment 1:N Block` 关系：[torch_npu `NPUCachingAllocator.cpp` L2096-L2156](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L2096-L2156)。

### 5. CANN Runtime 捕获任务，而不是接管 PyTorch Tensor allocator

Runtime 的 `Context::StreamBeginCapture` 创建 `RT_MODEL_CAPTURE_MODEL`，配置 software SQ 能力，并把 stream 加入 capture model：[CANN Runtime `context_aclgraph.cc` L300-L340](https://gitcode.com/cann/runtime/blob/fbbfae1f25c3b7e0699959df38b0d33a4d5c2a81/src/runtime/feature/aclgraph/context_aclgraph.cc#L300-L340)。

capture 状态下，普通 `Stream::AllocTask` 转向 `AllocCaptureTaskImpl`；它把任务分配给 capture stream、累计 SQE 数并记录任务公共信息，容量不足时还会扩展级联 capture stream：[CANN Runtime `stream_capture.cc` L72-L127](https://gitcode.com/cann/runtime/blob/fbbfae1f25c3b7e0699959df38b0d33a4d5c2a81/src/runtime/feature/aclgraph/stream_capture.cc#L72-L127)。这部分管理的是任务描述与执行队列资源，并没有替换 torch_npu 的 Tensor allocator。

### 6. capture end、replay 与最终释放

```text
aclmdlRICaptureEnd
  → markCaptureEnd
  → NPUCachingAllocator::endAllocateToPool
  → HostAllocator::end_allocate_to_pool
  → 图标记为可 replay；PrivatePool 继续保留
```

源码：[torch_npu `NPUGraph.cpp` L342-L375](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUGraph.cpp#L342-L375)。`endAllocateToPool` 只是删除分配路由，不释放已捕获地址。

replay 只调用 `AclmdlRIExecuteAsync(model_ri, stream)`：[torch_npu `NPUGraph.cpp` L393-L408](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUGraph.cpp#L393-L408)。普通情况下不会重新跑 `GetWorkspaceSize`、不会重新创建输出 Tensor，也不会重新走 graph-pool allocation。

销毁图时 `NPUGraph::reset` 先 `releasePool`，再销毁 CANN `model_ri`：[torch_npu `NPUGraph.cpp` L429-L478](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUGraph.cpp#L429-L478)。`releasePool` 只把图引用计数减一；计数为零才将池标成 freeable，而且仍要等残留 Tensor 引用释放：[torch_npu `NPUCachingAllocator.cpp` L2364-L2395](https://github.com/Ascend/pytorch/blob/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L2364-L2395)。

### 7. 缓存未命中后的底层设备分配

```text
NPUCachingAllocator（当前 ACLGraph 路径）
  → aclrtMallocAlign32
  → ACL aclMallocMemInner
  → Runtime DevMalloc（32-byte align / memory policy）
  → NpuDriver::DevMemAllocOnline/Offline
  → halMemAlloc
  → KMD/设备内存管理（开源证据到 HAL API 边界为止）
```

ACL 的 `aclrtMallocAlign32Impl` 进入 `aclMallocMemInner`：[CANN Runtime `memory.cpp` L410-L421](https://gitcode.com/cann/runtime/blob/fbbfae1f25c3b7e0699959df38b0d33a4d5c2a81/src/acl/aclrt_impl/memory.cpp#L410-L421)。Runtime `DevMalloc` 做 32 字节对齐并调用 driver，失败时触发隐式内存池 trim 后重试：[CANN Runtime `api_impl.cc` L2430-L2450](https://gitcode.com/cann/runtime/blob/fbbfae1f25c3b7e0699959df38b0d33a4d5c2a81/src/runtime/api/impl/api_impl.cc#L2430-L2450)。Driver 根据 huge/default/1G page policy 选择路径：[CANN Runtime `npu_driver_mem.cc` L1026-L1108](https://gitcode.com/cann/runtime/blob/fbbfae1f25c3b7e0699959df38b0d33a4d5c2a81/src/runtime/driver/npu_driver_mem.cc#L1026-L1108)，最终公开 HAL 契约是 `halMemAlloc/halMemFree`：[CANN Driver `ascend_hal_base.h` L2640-L2670](https://gitcode.com/cann/driver/blob/6e2914c1828fd3f5dbc38507f270cce8398cea57/pkg_inc/ascend_hal_base.h#L2640-L2670)。

## 所有权总表

| 层 | 管什么 | 不管什么 |
|---|---|---|
| vLLM-Ascend | capture key/bucket、graph cache、全局 pool 共享策略、输入输出引用 | HBM 页和 allocator block |
| torch_npu `NPUGraph` | Begin/End/Replay 编排、pool 与 graph 绑定、地址生命周期协调 | CANN task/SQ 固件执行 |
| `NPUCachingAllocator` | Tensor/V1 workspace 的 PrivatePool、缓存块、引用计数、最终回收资格 | CaptureModel 任务结构 |
| `NPUWorkspaceAllocator` | `TASK_QUEUE_ENABLE=2` 时 V2 workspace 的 per-stream 复用块 | 当前 `NPUGraph` capture 明确禁用此模式 |
| OpPlugin/ACLNN | 算子输出约定、workspace size、executor、执行 API | graph pool 的生命周期 |
| CANN Runtime | CaptureModel、capture stream、Task/SQE、Event/Notify、SQ/CQ、底层 malloc policy | PyTorch Tensor 的活跃引用和 graph pool 路由 |
| Driver/HAL | 最终 device virtual/physical memory allocation/free 接口 | 上层图语义 |

## Guard 与生命周期

| 变化 | 必须动作 | 静默复用风险 |
|---|---|---|
| batch/token/feature shape | 新 bucket 或 eager/reject | 错 mask/越界 |
| KV/runner buffer address | recapture 或稳定 address pool | 写旧 request |
| model/LoRA/quant | 新 artifact namespace | 错 weights/scales |
| HCCL topology/group epoch | recapture/重建 group | collective hang |
| CANN/driver/runtime | 整体失效 | binary/task ABI |

## 风险与证据边界

- **不要把 V2 workspace 行为外推到 ACLGraph。** 当前代码同时证明 V2 由 `TASK_QUEUE_ENABLE=2` 选择、`NPUGraph` 又拒绝该值，因此 ACLGraph capture 走 V1 graph pool。未来若这项限制改变，必须按新版本重新审计。
- **共享 graph pool 不代表图可安全并发。** 共享池允许地址复用，调用方必须保证共享图的生命周期/执行次序不会让同一块存储同时承担冲突用途。
- **弱引用只降低 Tensor 活跃引用，不等于立即归还 HBM。** graph pool 仍可缓存块，CANN Runtime/Driver 也可能有自己的 cache。
- **闭源边界**：`aclnnXxxGetWorkspaceSize` 如何计算、executor/tiling 内部是否申请额外内存、固件/KMD 如何落物理页，无法仅凭当前开源仓库证明。
- **无设备实测边界**：本文证明代码路径和所有权，不声称 capture 峰值、碎片率或 replay 收益。当前 capture 应重点采集 `NPUCachingAllocator` 的 `allocated_bytes`、`reserved_bytes`、graph pool snapshot 与设备侧实际 free memory；不应把 V2 workspace allocator 统计混入。

## 源码基线

| 仓库 | Commit | 原始仓库 |
|---|---|---|
| vllm-ascend | `3546357838389aa7201d9b44e234f831e98b2fd4` | [GitHub](https://github.com/vllm-project/vllm-ascend/tree/3546357838389aa7201d9b44e234f831e98b2fd4) |
| torch_npu | `e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa` | [GitHub](https://github.com/Ascend/pytorch/tree/e9e5c8e2eef2fcc03af8dbd3cb7561728fbbb7aa) |
| op-plugin | `90e9f78a68e7b30d743bce859e0a12d960b32ad2` | [GitCode](https://gitcode.com/Ascend/op-plugin/tree/90e9f78a68e7b30d743bce859e0a12d960b32ad2) |
| CANN Runtime | `fbbfae1f25c3b7e0699959df38b0d33a4d5c2a81` | [GitCode](https://gitcode.com/cann/runtime/tree/fbbfae1f25c3b7e0699959df38b0d33a4d5c2a81) |
| CANN Driver | `6e2914c1828fd3f5dbc38507f270cce8398cea57` | [GitCode](https://gitcode.com/cann/driver/tree/6e2914c1828fd3f5dbc38507f270cce8398cea57) |
