# device_context对接的AsyncRT\_\* C ABI

> 来源：[飞书知识库原文](https://terapines.feishu.cn/wiki/AjiVwOcIHiwE9CkUuXecbjDCnsg)  
> 迁移版本：revision 20；飞书最后更新时间：2026-09-08T07:18:35Z  
> 本地同步日期：2026-09-08

注释 1 `device_context.mojo` 当前共引用了 **78 个不同的 `AsyncRT_*` C ABI 符号**。它们通过 Mojo 的 `external_call[...]` 进入预编译的 native AsyncRT。

注意：并非每个 ABI 都直接调用 CUDA/HIP；retain/release、错误字符串、引用计数等可能只在 runtime 内部处理。

## Context 创建和生命周期

| C ABI | 用途 |
| --- | --- |
| AsyncRT_DeviceContext_create | 创建指定设备的 context |
| AsyncRT_DeviceContext_retain | 增加 context 引用计数 |
| AsyncRT_DeviceContext_release | 释放 context 引用 |
| AsyncRT_DeviceContextScope_create | 将 context 压入当前线程作用域 |
| AsyncRT_DeviceContextScope_release | 恢复先前的 context |
| AsyncRT_DeviceContext_setAsCurrent | 设置当前设备 context |

对应 CUDA 时，大致与设备选择/context 管理有关，但不保证一一对应某个 CUDA API。

## 设备发现和信息查询

| C ABI | 用途 |
| --- | --- |
| AsyncRT_DeviceContext_numberOfDevices | 查询某后端的设备数量 |
| AsyncRT_DeviceContext_deviceName | 查询设备名称 |
| AsyncRT_DeviceContext_deviceApi | 查询设备 API，如 cuda、hip、cpu |
| AsyncRT_DeviceContext_id | 查询设备 ID |
| AsyncRT_DeviceContext_archName | 查询架构名称 |
| AsyncRT_DeviceContext_computeCapability | 查询 compute capability |
| AsyncRT_DeviceContext_getApiVersion | 查询后端 API 版本 |
| AsyncRT_DeviceContext_getAttribute | 查询设备属性 |
| AsyncRT_DeviceContext_getMemoryInfo | 查询空闲和总内存 |
| AsyncRT_DeviceContext_maxSingleAllocationSize | 查询最大单次 allocation |
| AsyncRT_DeviceContext_isCompatible | 检查设备是否兼容 |
| AsyncRT_DeviceContext_runHealthcheck | 执行设备健康检查 |
| AsyncRT_DeviceContext_isHostUnified | 查询 host/device 是否共享物理内存 |
| AsyncRT_DeviceContext_supportsMulticast | 查询是否支持 multicast |

这组最终可能落到类似设备枚举、设备属性和显存信息查询的驱动 API。

## 多 GPU 和 P2P

| C ABI | 用途 |
| --- | --- |
| AsyncRT_DeviceContext_canAccess | 查询一个设备能否访问另一个设备 |
| AsyncRT_DeviceContext_enablePeerAccess | 开启两个设备之间的 P2P |
| AsyncRT_DeviceContext_enableAllPeerAccess | 为所有可用设备开启 P2P |
| AsyncRT_DeviceContext_allPeerAccessEnabled | 查询是否已全面开启 P2P |

对应 NVIDIA 时，概念上接近 CUDA peer-access API。

## Buffer 创建和生命周期

| C ABI | 用途 |
| --- | --- |
| AsyncRT_DeviceContext_createBuffer_async | 异步申请设备 buffer |
| AsyncRT_DeviceContext_createHostBuffer | 创建 host buffer |
| AsyncRT_DeviceContext_createBuffer_owning | 用已有指针构造 owning/non-owning buffer |
| AsyncRT_DeviceBuffer_createSubBuffer | 创建共享底层内存的子 buffer |
| AsyncRT_DeviceBuffer_retain | 增加 buffer 引用计数 |
| AsyncRT_DeviceBuffer_release | 释放 buffer 引用 |
| AsyncRT_DeviceBuffer_release_ptr | 释放通过裸指针管理的 buffer 引用 |
| AsyncRT_DeviceBuffer_reassignOwnershipTo | 将 buffer 所有权转移给另一个 context |
| AsyncRT_DeviceBuffer_context | 查询 buffer 所属 context |
| AsyncRT_DeviceBuffer_bytesize | 查询 buffer 字节数 |
| AsyncRT_DeviceBuffer_hostPtr | 获取可用的 host 映射指针 |

其中最可能触及驱动内存管理的是：

```text
AsyncRT_DeviceContext_createBuffer_async
AsyncRT_DeviceBuffer_release
```

但它们会先经过 MAX memory manager/VMM，不一定每次直接产生 `cuMemAlloc` 或 `hipMalloc`。

## 数据复制与内存填充

| C ABI | 用途 |
| --- | --- |
| AsyncRT_DeviceContext_HtoD_async | Host → Device 异步复制 |
| AsyncRT_DeviceContext_DtoH_async | Device → Host 异步复制 |
| AsyncRT_DeviceContext_DtoD_async | Device → Device 异步复制 |
| AsyncRT_DeviceContext_DtoD_async_no_cross_stream_sync | D2D 复制，但不自动加入跨 stream 同步 |
| AsyncRT_DeviceContext_setMemory_async | 异步 memset/fill |

这组比较直接地对应：

```text
CUDA memcpy/memset async
HIP memcpy/memset async
Metal blit/command encoder
```

## Kernel 加载和启动

| C ABI | 用途 |
| --- | --- |
| AsyncRT_DeviceContext_loadFunction | 从已编译产物加载设备函数 |
| AsyncRT_DeviceContext_enqueueFunctionDirect | 在 context 默认 stream 上启动 kernel |
| AsyncRT_DeviceStream_enqueueFunctionDirect | 在指定 stream 上启动 kernel |
| AsyncRT_DeviceFunction_retain | 增加设备函数引用计数 |
| AsyncRT_DeviceFunction_release | 释放设备函数 |
| AsyncRT_DeviceFunction_getAttribute | 查询 kernel 属性 |
| AsyncRT_DeviceFunction_copyToConstantMemory | 向 kernel/module 的 constant memory 复制数据 |
| AsyncRT_occupancyMaxActiveBlocksPerMultiprocessor | 计算每个 SM 可同时驻留的最大 block 数 |

最核心的硬件执行接口是：

```text
AsyncRT_DeviceContext_enqueueFunctionDirect
AsyncRT_DeviceStream_enqueueFunctionDirect
```

它们最终大致对应 CUDA/HIP/Metal 的 kernel launch。

## Host 函数提交

| C ABI | 用途 |
| --- | --- |
| AsyncRT_DeviceContext_enqueueHostFunction | 向执行队列提交一个 host 函数 |
| AsyncRT_DeviceContext_enqueueHostFunctionRange | 提交处理某个范围的 host 函数 |
| AsyncRT_DeviceStream_enqueueHostFunc | 在指定 stream 上安排 host callback |
| AsyncRT_DeviceStream_enqueueWaitOnHostValue | 让 stream 等待某个 host-visible value |

这些用于把 CPU callback、设备工作和异步依赖串起来。

## Stream 创建和生命周期

| C ABI | 用途 |
| --- | --- |
| AsyncRT_DeviceContext_stream | 获取 context 的 stream |
| AsyncRT_DeviceContext_createStream | 创建新 stream |
| AsyncRT_DeviceContext_createExternalStream | 包装外部提供的 stream |
| AsyncRT_DeviceContext_selectStream | 选择 context 内的某条 stream |
| AsyncRT_DeviceContext_numStreams | 查询 stream 数量 |
| AsyncRT_DeviceContext_streamPriorityRange | 查询支持的 stream 优先级范围 |
| AsyncRT_DeviceStream_retain | 增加 stream 引用计数 |
| AsyncRT_DeviceStream_release | 释放 stream |
| AsyncRT_DeviceStream_synchronize | 等待指定 stream 完成 |
| AsyncRT_DeviceContext_synchronize | 等待 context 上的工作完成 |

这组会对接 CUDA stream、HIP stream 或 Metal command queue 一类对象。

## Event 与异步依赖

| C ABI | 用途 |
| --- | --- |
| AsyncRT_DeviceContext_eventCreate | 创建 event |
| AsyncRT_DeviceContext_enqueue_event | 将 event 加入 context 执行队列 |
| AsyncRT_DeviceStream_eventRecord | 在 stream 上记录 event |
| AsyncRT_DeviceStream_waitForEvent | 让 stream 等待 event |
| AsyncRT_DeviceContext_enqueue_wait_for_context | 让一个 context 等待另一个 context |
| AsyncRT_DeviceEvent_retain | 增加 event 引用计数 |
| AsyncRT_DeviceEvent_release | 释放 event |
| AsyncRT_DeviceEvent_synchronize | 在 host 侧等待 event 完成 |

对应 CUDA/HIP 时，概念上就是 event record、stream wait event 和 event synchronize。

## Timer

| C ABI | 用途 |
| --- | --- |
| AsyncRT_DeviceContext_startTimer | 开始设备计时 |
| AsyncRT_DeviceContext_stopTimer | 停止设备计时并取得时间 |
| AsyncRT_DeviceTimer_release | 释放 timer |

通常会基于设备 event 计算 GPU 执行耗时。

## Completion flag

| C ABI | 用途 |
| --- | --- |
| AsyncRT_CompletionFlag_devicePtr | 获取 completion flag 的设备地址 |

用于 CPU/GPU 之间或异步任务之间表示“已完成”状态。

## Multicast buffer

| C ABI | 用途 |
| --- | --- |
| AsyncRT_DeviceMulticastBuffer_allocate | 在多个设备间创建 multicast 内存对象 |
| AsyncRT_DeviceMulticastBuffer_unicastBufferFor | 获取某设备对应的普通单播 view |
| AsyncRT_DeviceMulticastBuffer_multicastBufferFor | 获取某设备对应的 multicast view |

主要用于支持相关硬件能力的多 GPU 通信。

## 辅助函数

| C ABI | 用途 |
| --- | --- |
| AsyncRT_DeviceContext_strfree | 释放 AsyncRT 返回的 C 字符串 |

它不操作硬件，只处理 native runtime 返回字符串的内存。

最核心的硬件对接面可以进一步压缩为：

```text
设备初始化
  AsyncRT_DeviceContext_create

显存
  AsyncRT_DeviceContext_createBuffer_async
  AsyncRT_DeviceBuffer_release

复制
  AsyncRT_DeviceContext_HtoD_async
  AsyncRT_DeviceContext_DtoH_async
  AsyncRT_DeviceContext_DtoD_async
  AsyncRT_DeviceContext_setMemory_async

Kernel
  AsyncRT_DeviceContext_loadFunction
  AsyncRT_DeviceContext_enqueueFunctionDirect
  AsyncRT_DeviceStream_enqueueFunctionDirect

并发与同步
  AsyncRT_DeviceContext_createStream
  AsyncRT_DeviceStream_eventRecord
  AsyncRT_DeviceStream_waitForEvent
  AsyncRT_DeviceStream_synchronize
  AsyncRT_DeviceContext_synchronize
```

这 15 个左右的 ABI，构成了 `device_context.mojo` 对接底层 GPU runtime/driver 的主干；其余大多是对象生命周期、设备查询、多 GPU 和辅助功能。
