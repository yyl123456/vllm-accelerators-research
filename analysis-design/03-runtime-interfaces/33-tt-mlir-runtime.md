# [tt-mlir/runtime]

> 来源：[飞书知识库原文](https://terapines.feishu.cn/wiki/XSNWwpleBico1Skoo3ccUjEvn8f)  
> 迁移版本：revision 33；飞书最后更新时间：2026-09-08T07:51:16Z  
> 本地同步日期：2026-09-08

tt-mlir/runtime/include/tt/runtime/runtime.h

tt-mlir/runtime/lib/runtime.cpp

tt-mlir/runtime/lib/ttmetal/runtime.cpp



注释 1 [runtime.cpp (line 1281)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/runtime.cpp:1281) 是统一 runtime 分发层的实现；上层实际包含的公开头文件是 [runtime.h (line 15)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/include/tt/runtime/runtime.h:15)。

接口位于 `tt::runtime` 命名空间，主要负责将调用分派给 TTNN、TTMetal 或 Distributed runtime。

## Runtime 配置与选择

| 接口 | 作用 |
| --- | --- |
| setMlirHome(path) | 设置 MLIR 安装/资源路径 |
| setMetalHome(path) | 设置 tt-metal 根目录 |
| setMemoryLogLevel(level) | 设置内存日志级别 |
| getAvailableDeviceRuntimes() | 返回可用设备后端，如 TTNN、TTMetal |
| getCurrentDeviceRuntime() | 获取当前设备后端 |
| setCurrentDeviceRuntime(runtime) | 显式选择 TTNN 或 TTMetal |
| setCompatibleDeviceRuntime(binary) | 根据 executable 类型自动选择兼容后端 |
| getAvailableHostRuntimes() | 返回可用 host runtime |
| getCurrentHostRuntime() | 获取当前 host runtime |
| setCurrentHostRuntime(runtime) | 选择本地或 distributed host runtime |

实现集中在 [runtime.cpp (line 128)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/runtime.cpp:128)。

## 系统与硬件描述

| 接口 | 作用 |
| --- | --- |
| getCurrentSystemDesc(...) | 查询当前机器、芯片、grid、L1/DRAM 等系统描述 |
| getArch() | 返回芯片架构，如 Wormhole、Blackhole |
| getNumAvailableDevices() | 查询可用设备数量 |

`SystemDesc` 也是编译器进行静态内存规划时获取 L1/DRAM 可用区间和对齐要求的来源。

## Distributed runtime

| 接口 | 作用 |
| --- | --- |
| launchDistributedRuntime(options) | 启动分布式 controller/worker runtime |
| shutdownDistributedRuntime() | 关闭分布式 runtime |
| getWorkerDebugStats() | 查询 worker 调试统计 |

这部分不是 TTMetal 单设备执行的必经路径。

## Host Tensor 创建

| 接口 | 作用 |
| --- | --- |
| createBorrowedHostTensor(...) | 用外部 host 内存创建非 owning Tensor，不复制数据 |
| createOwnedHostTensor(...) | 复制数据并创建 owning Host Tensor |
| createUnsafeBorrowedHostTensor(tensor) | 从已有 owning Tensor 创建共享其内存的 borrowed Tensor |
| createMultiDeviceHostTensor(data, ...) | 从多份 host 数据创建多设备 Tensor |
| createMultiDeviceHostTensor(shards, ...) | 从已有 Tensor shards 组合多设备 Tensor |
| createMultiDeviceBorrowedHostTensor(...) | 从多份外部内存构造 borrowed 多设备 Tensor |
| createScalarTensor(scalar) | 创建标量 Tensor |

此外还有使用 `TensorDesc` 的便捷重载。

## Device Tensor 创建

| 接口 | 作用 |
| --- | --- |
| createEmptyTensor(device, layout, shape, stride, itemsize) | 按布局在 host 或 device 创建空 Tensor |
| createEmptyTensor(device, layout, TensorDesc) | 上述接口的描述符版本 |

注意：

- TTNN 后端通常会在这里动态分配设备地址。
- TTMetal executable 的内部激活 buffer 不主要通过这个公开接口创建，而是由 `MCQExecutor` 解释 `CreateBufferCommand`，绑定 executable 中的固定地址。

## Tensor 元数据查询

| 接口 | 作用 |
| --- | --- |
| isTensorAllocated(tensor) | 判断 Tensor 当前是否仍有有效存储 |
| getTensorDataType(tensor) | 获取数据类型 |
| getTensorDataBuffer(tensor) | 获取 Tensor 数据的 host 字节副本 |
| getTensorElementSize(tensor) | 获取单个元素字节数 |
| getTensorVolume(tensor) | 获取物理元素数量 |
| getTensorLogicalVolume(tensor) | 获取逻辑元素数量 |
| getTensorShape(tensor) | 获取 shape |
| getTensorStride(tensor) | 获取 stride |
| getTensorDesc(tensor) | 一次性获取完整 Tensor 描述 |
| getTensorRetain(tensor) | 查询 runtime 是否保留 Tensor |
| setTensorRetain(tensor, retain) | 设置 Tensor 保留状态 |

声明见 [runtime.h (line 139)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/include/tt/runtime/runtime.h:139)。

## 设备生命周期与 Mesh 管理

| 接口 | 作用 |
| --- | --- |
| openMeshDevice(options) | 打开一个物理设备或 MeshDevice |
| closeMeshDevice(device) | 关闭父 MeshDevice |
| createSubMeshDevice(parent, shape, offset) | 从父 Mesh 创建子 Mesh |
| releaseSubMeshDevice(submesh) | 释放子 Mesh |
| reshapeMeshDevice(device, shape) | 改变 Mesh 的逻辑形状 |
| getMeshShape(device) | 获取 Mesh shape |
| getDeviceIds(device) | 获取 Mesh 中的物理设备 ID |
| getMappedDeviceIds(shape) | 查询指定 Mesh shape 会映射到哪些设备 |
| getNumHwCqs(device) | 获取 hardware command queue 数量 |

TTMetal 后端中，`openMeshDevice()` 最终调用：

```text
tt_metal::distributed::MeshDevice::create(...)
```

见 [ttmetal/runtime.cpp (line 280)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/runtime.cpp:280)。

## 设备内存与缓存信息

| 接口 | 作用 |
| --- | --- |
| isProgramCacheEnabled(device) | 查询 tt-metal program cache 是否开启 |
| clearProgramCache(device) | 清除 program cache |
| getL1SmallSize(device) | 查询保留的 L1 small 区域大小 |
| getTraceRegionSize(device) | 查询 trace buffer 区域大小 |
| getNumDramChannels(device) | 查询 DRAM channel 数量 |
| getDramSizePerChannel(device) | 查询每个 DRAM channel 大小 |
| getL1SizePerCore(device) | 查询每个 core 的 L1 大小 |
| deallocateBuffers(device) | 批量释放设备 allocator 管理的 buffer |
| dumpMemoryReport(device) | 输出 allocator 内存报告 |
| getMemoryView(device) | 获取 DRAM、L1、L1Small、Trace 的使用视图 |
| readDeviceProfilerResults(device) | 读取设备 profiler 结果 |

声明见 [runtime.h (line 173)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/include/tt/runtime/runtime.h:173)。

## Tensor 布局与设备迁移

| 接口 | 作用 |
| --- | --- |
| toHost(tensor, untilize, blocking) | 把设备 Tensor 搬到 host，可选择 untilize |
| getDeviceTensors(tensor) | 获取多设备 Tensor 的各设备 shard |
| toLayout(tensor, device, layout, retain) | 转换数据布局或 host/device 存储位置 |
| hasLayout(tensor, layout) | 判断 Tensor 是否符合指定布局 |
| getTensorLayout(tensor) | 获取当前 Tensor 布局 |
| getLayout(binary, programIndex, inputIndex) | 从 executable 获取指定输入所需布局 |

其中 `getLayout()` 常用于提交前准备输入：

```text
Layout layout = getLayout(binary, programIndex, inputIndex);
Tensor deviceInput = toLayout(hostInput, device, layout);
```

## 数据复制

提供两个主要重载：

```text
void memcpy(void *dst, Tensor src, optional<DataType> targetType);
void memcpy(Tensor dst, Tensor src);
```

| 接口 | 典型用途 |
| --- | --- |
| memcpy(void*, Tensor) | Tensor → 普通 host buffer |
| memcpy(Tensor, Tensor) | Host↔Device、Device↔Device 或 Host↔Host |

底层行为由当前 runtime 决定：

```text
TTNN     → ttnn::memcpy
TTMetal  → ttmetal::memcpy
Distributed → distributed::memcpy
```

分派实现见 [runtime.cpp (line 1039)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/runtime.cpp:1039)。

## 同步接口

| 接口 | 作用 |
| --- | --- |
| wait(Event) | 等待一个 runtime event |
| wait(Tensor, cqId) | 等待产生该 Tensor 的异步操作完成 |
| wait(vector<Tensor>, cqId) | 等待多个 Tensor 完成 |

它不像 Modular `DeviceContext` 那样把 stream 作为核心公开对象，而是主要通过 Tensor/Event 和 command queue ID 表达依赖。

## 显式释放

| 接口 | 作用 |
| --- | --- |
| deallocateTensor(tensor, force) | 释放单个 Tensor 的 host/device 存储 |
| deallocateBuffers(device) | 清空设备 allocator 管理的 buffers |
| releaseTrace(device, binaryId, programId) | 释放 TTNN trace 占用的资源 |

`deallocateTensor()` 会根据当前 runtime 分发，见 [runtime.cpp (line 1058)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/runtime.cpp:1058)。

## Fabric 配置

| 接口 | 作用 |
| --- | --- |
| computeMeshFabricConfig(systemDesc, meshShape) | 根据系统拓扑计算 Mesh fabric 配置 |
| setFabricConfig(config) | 配置芯片间 fabric 通信 |

主要用于多芯片 collective、点对点通信等场景。

## Executable 执行接口

最核心的上层执行接口是：

```text
std::vector<Tensor> submit(
    Device device,
    Binary executable,
    uint32_t programIndex,
    std::vector<Tensor> &inputs);
```

作用是：

1. 选择 executable 中第 `programIndex` 个 program；
2. 绑定输入 Tensor；
3. 分派到当前 runtime；
4. 执行 program；
5. 返回输出 Tensor。

分派代码正是你标出的 [runtime.cpp (line 1281)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/runtime.cpp:1281)：

```text
TTNN executable
    → tt::runtime::ttnn::submit(...)

TTMetal executable
    → tt::runtime::ttmetal::submit(...)

Distributed runtime
    → tt::runtime::distributed::submit(...)
```

因此 `runtime.cpp::submit()` 只是统一入口；TTMetal 真正的 executable 解释执行位于 [ttmetal/runtime.cpp (line 1027)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/runtime.cpp:1027)。

## Executable 与回调调试接口

| 接口 | 作用 |
| --- | --- |
| walkProgram(binary, programIndex, callback) | 遍历 executable 中的操作 |
| getOpDebugString(opContext) | 获取 op 的调试文本 |
| getOpLocInfo(opContext) | 获取 op 对应的 MLIR/source location |
| getOpOutputTensor(op, programContext) | 获取某个 op 的输出 Tensor |
| getOpOutputRefs(op) | 获取 op 输出的 FlatBuffer TensorRef |
| getOpInputRefs(op) | 获取 op 输入的 TensorRef |
| getTensorRefShape(ref) | 从 TensorRef 获取 shape |
| getTensorRefDataType(ref) | 从 TensorRef 获取 dtype |
| retrieveTensorFromPool(context, ref, untilize) | 从执行期 tensor pool 获取 Tensor |
| updateTensorInPool(context, ref, tensor) | 替换 tensor pool 中的 payload |
| getProgramIndex(context) | 获取当前执行的 program index |
| invokeCpuOp(context, op, inputs) | 单独执行被 hoist 到 CPU 的操作 |

其中部分接口当前只支持 TTNN；调用 TTMetal 可能进入 `fatalNotImplemented()`。

## Tensor 文件接口

| 接口 | 作用 |
| --- | --- |
| dumpTensor(tensor, filePath) | 把 Tensor 保存成二进制文件 |
| loadTensor(filePath, device) | 从二进制文件加载 Tensor，可选择放到设备上 |

## 与 Modular `DeviceContext` 的核心差异

Modular 把下面能力集中进一个有状态对象：

```text
DeviceContext
├── device
├── stream
├── allocation
├── copy
├── compile/load function
└── enqueue
```

tt-runtime 则采用“opaque handle + namespace function”：

```text
tt::runtime
├── Device handle
├── Tensor handle
├── Event handle
├── Binary handle
├── openMeshDevice()
├── memcpy()/toLayout()
├── wait()
└── submit()
```

因此最接近 Modular `DeviceContext` 的最小核心 API 是：

```text
Device openMeshDevice(...);
Tensor createEmptyTensor(...);
Tensor toLayout(...);
void memcpy(...);
std::vector<Tensor> submit(...);
void wait(...);
void deallocateTensor(...);
void closeMeshDevice(...);
```

它们共同组成 tt-runtime 面向上层的设备执行上下文，只是没有封装为单独的 `DeviceContext` 类。
