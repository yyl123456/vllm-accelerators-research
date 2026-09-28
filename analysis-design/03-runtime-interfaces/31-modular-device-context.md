# device_context.mojo

> 来源：[飞书知识库原文](https://terapines.feishu.cn/wiki/T3cGwcWruiiEizkbVqWc9coznuf)  
> 迁移版本：revision 15；飞书最后更新时间：2026-09-08T06:44:10Z  
> 本地同步日期：2026-09-08

```text
device_context.mojo
       ↓ external_call
AsyncRT_* C ABI
       ↓ 闭源/预编译 runtime
CUDA / HIP / Metal
```



modular/max/mojo/max/gpu/host/device_context.mojo

这个文件主要对外暴露 5 类对象。真正需要先看的，是 `DeviceContext`，其余是它返回或接收的对象。

## 1. `DeviceContext`：设备操作总入口

位置：[device_context.mojo (line 3867)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:3867)

### 设备信息

```text
ctx.name()
ctx.api()
ctx.id()
ctx.arch_name()
ctx.compute_capability()
ctx.get_api_version()
ctx.get_attribute(...)
ctx.get_memory_info()
ctx.max_single_alloc_size()
ctx.is_compatible()
ctx.run_healthcheck()
```

用途：

- `name()`：设备名称；
- `api()`：后端类型，例如 CUDA、HIP；
- `arch_name()`：GPU 架构；
- `compute_capability()`：NVIDIA compute capability；
- `get_memory_info()`：空闲和总显存；
- `max_single_alloc_size()`：允许的最大单次 allocation。

### 内存分配

```text
ctx.enqueue_create_buffer[dtype](size)
ctx.create_buffer_sync[dtype](size)
ctx.enqueue_create_host_buffer[dtype](size)
```

位置：[device_context.mojo (line 4270)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:4270)

区别：

- `enqueue_create_buffer()`：异步创建设备 buffer；
- `create_buffer_sync()`：同步创建设备 buffer；
- `enqueue_create_host_buffer()`：创建 host buffer，通常用于 CPU 或主机与设备的数据交换。

典型调用链：

```text
enqueue_create_buffer()
    ↓
AsyncRT_DeviceContext_createBuffer_async
    ↓
MAX memory manager
    ↓
CUDA/HIP/VMM allocation
```

### 数据复制

文件中有很多 `enqueue_copy()` 重载，可以合并理解为：

```text
ctx.enqueue_copy(dst_device, src_device)
ctx.enqueue_copy(dst_device, src_host)
ctx.enqueue_copy(dst_host, src_device)
ctx.enqueue_copy(dst_host, src_host)
```

位置从 [device_context.mojo (line 5373)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:5373) 开始。

这些覆盖：

```text
Host   → Device
Device → Host
Device → Device
Host   → Host
```

另外还有：

```text
ctx.enqueue_copy_no_cross_stream_sync(...)
```

它用于明确跳过某些跨 stream 同步，属于偏底层的高级接口。

### 内存填充

```text
ctx.enqueue_memset(buffer, value)
```

位置：[device_context.mojo (line 5862)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:5862)

用途类似：

```text
cudaMemsetAsync
hipMemsetAsync
```

但先经过 AsyncRT/设备后端。

### 编译和加载 kernel

```text
ctx.compile_function[func](...)
ctx.load_function(...)
```

位置：

- [compile_function (line 4361)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:4361)
- [load_function (line 4514)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:4514)

可以理解为：

```text
compile_function
    Mojo kernel → 设备代码 → DeviceFunction

load_function
    已有设备代码/模块 → DeviceFunction
```

### 启动 kernel

```text
ctx.enqueue_function(
    function,
    arguments...,
    grid_dim=...,
    block_dim=...,
    shared_mem_bytes=...,
)
```

位置从 [device_context.mojo (line 4581)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:4581) 开始。

它有多个重载，但核心都是：

```text
DeviceFunction
+ kernel 参数
+ grid dimension
+ block dimension
+ dynamic shared memory
          ↓
      kernel launch
```

这是最接近下面这些调用的接口：

```text
cuLaunchKernel
hipModuleLaunchKernel
Metal command encoder dispatch
```

### CPU 函数执行

```text
ctx.enqueue_cpu_function(...)
ctx.enqueue_cpu_range(...)
```

位置：

- [enqueue_cpu_function (line 5039)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:5039)
- [enqueue_cpu_range (line 5077)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:5077)

它们用于把 CPU 工作加入同一套异步执行体系。

### Stream

```text
ctx.stream()
ctx.create_stream(priority=...)
ctx.create_external_stream(...)
ctx.stream_priority_range()
ctx.num_streams()
ctx.select_stream(stream_id)
```

位置从 [device_context.mojo (line 5960)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:5960) 开始。

用途：

- 获取默认 stream；
- 创建新 stream；
- 包装外部 CUDA/HIP stream；
- 查询优先级；
- 选择 context 内的某条 stream。

### 同步与依赖

```text
ctx.synchronize()
ctx.enqueue_wait_for(other_context)
ctx.create_event(...)
```

用途：

- `synchronize()`：等待该 context 上的任务完成；
- `enqueue_wait_for()`：让一个 context/stream 等待另一个；
- `create_event()`：创建设备 event。

### 多 GPU

```text
ctx.can_access(peer)
ctx.enable_peer_access(peer)
ctx.supports_multicast()
ctx.is_host_unified()

DeviceContext.number_of_devices(...)
DeviceContext.enable_all_peer_access()
DeviceContext.all_peer_access_enabled()
```

用途：

- 判断 GPU 之间能否直接访问；
- 开启 P2P；
- 查询 multicast；
- 查询 unified memory；
- 获取设备数量。

### 性能测量

```text
ctx.execution_time(...)
ctx.execution_time_iter(...)
```

用于通过设备 timer/event 测量 kernel 或一组异步任务的执行时间。

---

## 2. `DeviceBuffer`：设备内存

位置：[device_context.mojo (line 1446)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:1446)

主要函数：

```text
buffer.create_sub_buffer[offset, size]
buffer.enqueue_copy_to(dst)
buffer.enqueue_copy_from(src)
buffer.enqueue_fill(value)
buffer.context()
buffer.map_to_host(...)
buffer.device_ptr()
buffer.unsafe_ptr()
buffer.take_ptr()
buffer.reassign_ownership_to(ctx)
```

最重要的几个：

- `create_sub_buffer()`：创建共享底层 allocation 的子区域；
- `enqueue_copy_to/from()`：复制数据；
- `enqueue_fill()`：填充值；
- `device_ptr()`：取得设备地址；
- `map_to_host()`：尝试把设备内存映射到 host；
- `context()`：取得所属设备 context。

释放通常由对象生命周期触发，最终进入：

```text
AsyncRT_DeviceBuffer_release
```

## 3. `HostBuffer`：主机内存

位置：[device_context.mojo (line 366)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:366)

接口和 `DeviceBuffer` 很相似：

```text
buffer.create_sub_buffer(...)
buffer.enqueue_copy_to(...)
buffer.enqueue_copy_from(...)
buffer.enqueue_fill(...)
buffer.unsafe_ptr()
buffer.context()
buffer.reassign_ownership_to(...)
```

它代表 host-visible 内存，可能是：

- 普通 CPU 内存；
- pinned host memory；
- 与设备共享或映射的内存。

## 4. `DeviceStream` 和 `DeviceEvent`

### `DeviceStream`

位置：[device_context.mojo (line 2260)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:2260)

主要接口：

```text
stream.enqueue(...)
stream.enqueue_function(...)
stream.synchronize()
stream.record_event(event)
stream.enqueue_wait_for(event)
stream.enqueue_host_func(...)
stream.wait_for_host_value(...)
```

对应关系大致是：

```text
DeviceStream       ≈ CUDA/HIP stream
record_event       ≈ cudaEventRecord
enqueue_wait_for   ≈ cudaStreamWaitEvent
synchronize        ≈ cudaStreamSynchronize
```

### `DeviceEvent`

位置：[device_context.mojo (line 2751)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:2751)

主要接口：

```text
event.synchronize()
```

event 通常由 `DeviceContext.create_event()` 创建，然后交给 stream 记录或等待。

## 5. `DeviceFunction`：已经编译的 kernel

位置：[device_context.mojo (line 2860)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:2860)

主要接口：

```text
function.get_attribute(...)
function.occupancy_max_active_blocks_per_multiprocessor(...)
function.dump_rep(...)
```

用途：

- 查询 kernel 属性；
- 计算 occupancy；
- 查看编译后的表示。

另外还有：

```text
DeviceExternalFunction
```

位置：[device_context.mojo (line 3652)](/home/yyl/workspace/llmss/engine/modular/max/mojo/max/gpu/host/device_context.mojo:3652)

它主要用来包装外部提供的设备函数。

## 最值得先看的 8 个接口

如果目标只是理解“Modular 怎么对接 CUDA/HIP”，先看这些就够了：

```text
DeviceContext.enqueue_create_buffer()
DeviceBuffer.create_sub_buffer()
DeviceContext.enqueue_copy()
DeviceContext.enqueue_memset()
DeviceContext.compile_function()
DeviceContext.load_function()
DeviceContext.enqueue_function()
DeviceContext.synchronize()
```

对应四种底层行为：

```text
显存管理：create_buffer / create_sub_buffer
数据传输：enqueue_copy / enqueue_memset
程序管理：compile_function / load_function
设备执行：enqueue_function / synchronize
```

它们内部再通过 `external_call["AsyncRT_*"]` 进入 Modular 的 native runtime。
