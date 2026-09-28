# runtime对接后端

> 来源：[飞书知识库原文](https://terapines.feishu.cn/wiki/O0u9wvsIwi8QmTkaEiEc2unCndg)  
> 迁移版本：revision 10；飞书最后更新时间：2026-09-08T07:29:53Z  
> 本地同步日期：2026-09-08

[runtime.cpp::submit() (line 1281)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/runtime.cpp:1281) 自己直接调用的后端接口只有三个：

```text
tt::runtime::ttnn::submit(...)
tt::runtime::ttmetal::submit(...)
tt::runtime::distributed::submit(...)
```

具体调用哪一个，由当前的 `DeviceRuntime` 和 `HostRuntime` 决定。

## 总分派关系

```text
tt::runtime::submit()
│
├─ Local + TTNN
│    └─ tt::runtime::ttnn::submit()
│
├─ Local + TTMetal
│    └─ tt::runtime::ttmetal::submit()
│
└─ Distributed
     └─ tt::runtime::distributed::submit()
```

源码分派位置：

[runtime.cpp (line 1281)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/runtime.cpp:1281)

```text
return DISPATCH_TO_CURRENT_RUNTIME(
    RetType,
    [&] {
        return tt::runtime::ttnn::submit(...);
    },
    [&] {
        return tt::runtime::ttmetal::submit(...);
    },
    [&] {
        return tt::runtime::distributed::submit(...);
    });
```

下面分别展开。

---

## TTMetal 后端调用链

### 2.1 第一级：进入 TTMetal runtime

直接调用：

```text
tt::runtime::ttmetal::submit(
    deviceHandle,
    executableHandle,
    programIndex,
    inputs);
```

实现位于：

[ttmetal/runtime.cpp (line 1027)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/runtime.cpp:1027)

它首先调用或访问：

| 接口 | 所属层 | 作用 |
| --- | --- | --- |
| getBinary(executableHandle) | tt-runtime | 将通用 Binary handle 转成 TTMetalBinary FlatBuffer |
| fbb.programs()->Get(programIndex) | FlatBuffer schema | 取出目标 Program |
| deviceHandle.as<MeshDevice>() | tt-runtime handle | 取出底层 tt-metal MeshDevice |
| MeshDevice::create_submesh(...) | tt-metal | 必要时创建 program 对应的子 Mesh |
| executeMeshDeviceProgram(...) | tt-runtime TTMetal executor | 解释并执行 DeviceProgram |

主要逻辑：

```text
Binary handle
  → TTMetalBinary
  → Program[programIndex]
  → DeviceProgram[]
  → executeMeshDeviceProgram()
```

### 2.2 第二级：创建命令解释器

`executeMeshDeviceProgram()` 创建：

```text
MCQExecutor executor(...);
executor.execute(commandQueue);
```

见 [executor.cpp (line 665)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:665)。

命令队列通过：

```text
meshDevice->mesh_command_queue(queueId)
```

获取实际 tt-metal `MeshCommandQueue`，然后逐条解释 FlatBuffer command，见 [executor.cpp (line 156)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:156)。

支持的 executable command 包括：

| FlatBuffer command | runtime 动作 |
| --- | --- |
| HostAllocCommand | 创建 host buffer |
| ReturnCommand | 收集程序输出 |
| CreateBufferCommand | 建立固定地址的设备 Buffer |
| DeallocateBufferCommand | 释放 Buffer wrapper |
| EnqueueWriteBufferCommand | Host → Device |
| EnqueueReadBufferCommand | Device → Host |
| EnqueueProgramCommand | 创建并提交设备 kernel |
| EnqueueRecordEventCommand | 记录 event |
| EnqueueWaitForEventCommand | command queue 等待 event |
| EventSynchronizeCommand | host 等待 event |
| MemrefCopyCommand | host memref copy |
| CpuCommand | 调用 CPU 动态库函数 |
| FinishCommand | 等待 command queue 完成 |
| MeshShardCommand | host 侧分片/合并 |
| CreateGlobalSemaphoreCommand | 创建全局 semaphore |
| ResetGlobalSemaphoreCommand | 重置全局 semaphore |
| CreateLocalSemaphoreCommand | 记录本地 semaphore 初值 |

完整 switch 见 [executor.cpp (line 173)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:173)。

### 2.3 Buffer 相关 tt-metal 接口

#### 创建固定地址 Buffer

```text
CreateBufferCommand
 → createMeshBufferFromBufferRef()
 → distributed::MeshBuffer::create(..., compiledAddress)
 → Buffer::create(device, address, ...)
```

runtime 调用点：

[executor.cpp (line 521)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:521)

主要后端接口：

```text
tt_metal::distributed::MeshBuffer::create(
    meshBufferConfig,
    deviceLocalConfig,
    meshDevice,
    address);
```

这里的 `address` 来自 executable，不由 tt-metal 动态选择。

#### 释放 Buffer

```text
meshBuffer->deallocate();
```

见 [executor.cpp (line 529)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:529)。

对于编译器固定地址建立的 non-owning buffer，这主要销毁运行时对象，不把地址交给动态 allocator 重新管理。

### 2.4 数据传输接口

#### Host → Device

```text
writeHostTensorToMeshBuffer(
    meshCommandQueue,
    hostBuffer,
    meshBuffer,
    blocking);
```

调用点见 [executor.cpp (line 501)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:501)。

它进一步使用 tt-metal Mesh Command Queue 的写 buffer 接口。

#### Device → Host

```text
readHostTensorFromMeshBuffer(
    meshCommandQueue,
    meshBuffer,
    hostBuffer,
    blocking);
```

见 [executor.cpp (line 511)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:511)。

这些操作会进入 tt-metal dispatch/UMD 数据传输路径。

### 2.5 Kernel 创建接口

处理 `EnqueueProgramCommand` 时调用：

```text
tt_metal::CreateProgram()
```

创建一个 host-side `tt_metal::Program`，见 [executor.cpp (line 365)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:365)。

对 executable 中的每个 kernel 调用：

```text
createKernel(...)
```

`createKernel()` 根据 kernel 类型，最终使用 tt-metal 的 kernel 创建能力，配置：

- Compute kernel
- NoC/Data Movement kernel
- core range
- NOC 0/1
- math fidelity
- FP32 accumulation
- unpack/pack 配置
- 编译优化等级
- kernel source

调用位置见 [executor.cpp (line 385)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:385)。

### 2.6 Kernel 参数接口

公共 runtime arguments：

```text
tt_metal::SetCommonRuntimeArgs(
    program,
    kernelHandle,
    commonArgs);
```

每个 core 的 runtime arguments：

```text
tt_metal::SetRuntimeArgs(
    program,
    kernelHandle,
    coreRange,
    runtimeArgs);
```

见 [executor.cpp (line 395)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:395)。

参数内容可能包括：

- Buffer 地址
- CB 地址或 port
- semaphore 地址
- tensor accessor 参数
- shape/stride
- 用户 runtime scalar

这里是编译期 BufferRef 最终变成 kernel 实际设备地址的地方。

### 2.7 Circular Buffer 接口

对 executable 中的 CB 描述调用：

```text
tt_metal::CircularBufferConfig config = ...;

tt_metal::CreateCircularBuffer(
    program,
    coreRangeSet,
    config);
```

见 [executor.cpp (line 443)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:443)。

如果 circular buffer 与编译期规划的全局 buffer 关联，配置中会使用对应固定地址。

### 2.8 Semaphore 接口

局部 semaphore：

```text
tt_metal::CreateSemaphore(
    program,
    coreRangeSet,
    initialValue);
```

全局 semaphore：

```text
tt_metal::experimental::CreateGlobalSemaphore(...)
```

另外 runtime 支持把 semaphore 地址填入 kernel runtime args。

### 2.9 Event 与同步接口

| runtime 调用 | tt-metal 后端接口 |
| --- | --- |
| 记录 event | mcq->enqueue_record_event() |
| queue 等待 event | mcq->enqueue_wait_for_event(event) |
| host 同步 event | distributed::EventSynchronize(event) |
| command queue 完成 | distributed::Finish(mcq) |

对应实现见 [executor.cpp (line 540)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:540)。

### 2.10 最终提交接口

所有 Buffer、kernel、runtime args、CB、semaphore 准备好后，调用：

```text
distributed::EnqueueMeshWorkload(
    *meshCommandQueue,
    meshWorkload,
    blocking);
```

见 [executor.cpp (line 494)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttmetal/executor.cpp:494)。

这是 TTMetal submit 路径中最核心的后端执行接口：

```text
tt-runtime
 → tt-metal MeshCommandQueue
 → fast/slow dispatch
 → UMD/KMD
 → device
```

所以 TTMetal 分支的关键后端 API 可浓缩为：

```text
MeshDevice::create_submesh()
MeshDevice::mesh_command_queue()

MeshBuffer::create()
MeshBuffer::deallocate()

CreateProgram()
CreateKernel(...)
SetCommonRuntimeArgs()
SetRuntimeArgs()
CreateCircularBuffer()
CreateSemaphore()
CreateGlobalSemaphore(...)

EnqueueRead/WriteBuffer(...)
EnqueueMeshWorkload(...)
EventSynchronize(...)
Finish(...)
```

---

## TTNN 后端调用链

### 3.1 第一级：进入 TTNN runtime

```text
tt::runtime::ttnn::submit(...)
```

实现位于：

[ttnn/runtime.cpp (line 2504)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttnn/runtime.cpp:2504)。

它创建：

```text
ProgramExecutor(
    deviceHandle,
    executableHandle,
    programIndex,
    inputs);
```

然后调用：

```text
executor->execute();
executor->gatherOutputTensors();
```

### 3.2 ProgramExecutor 解释 TTNN operations

与 TTMetal command stream 不同，TTNN executable 中保存的是 TTNN operation。`ProgramExecutor` 遍历每个 operation，并分发到：

```text
runtime/lib/ttnn/operations/<category>/<op>.cpp
```

分派入口在：

[program_executor.cpp](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/ttnn/program_executor.cpp)

典型映射如下：

| executable op | runtime wrapper | TTNN 后端接口 |
| --- | --- | --- |
| EmptyOp | operations::creation::run() | ttnn::empty() |
| ToDeviceOp | operations::layout::run() | ttnn::to_device() |
| FromDeviceOp | operations::layout::run() | ttnn::from_device() |
| ToLayoutOp | layout wrapper | ttnn::to_layout() |
| ToMemoryConfigOp | layout wrapper | ttnn::to_memory_config() |
| MatmulOp | matmul wrapper | ttnn::matmul() |
| LinearOp | matmul wrapper | ttnn::linear() |
| Conv2dOp | convolution wrapper | ttnn::conv2d() |
| AddOp | eltwise wrapper | ttnn::add() |
| MultiplyOp | eltwise wrapper | ttnn::multiply() |
| SoftmaxOp | normalization wrapper | ttnn::softmax() |
| AllGatherOp | CCL wrapper | ttnn::all_gather() |
| DeallocateOp | deletion wrapper | ttnn::deallocate() |

TTNN runtime wrappers实际调用的后端接口覆盖：

- creation：`empty`、`zeros`、`full`、`arange`
- layout：`to_device`、`from_device`、`to_layout`、`to_memory_config`
- eltwise：`add`、`subtract`、`multiply`、`divide`、比较和 bitwise
- matmul：`matmul`、`linear`、`sparse_matmul`
- convolution：`conv1d`、`conv2d`、`conv3d`、`conv_transpose2d`
- reduction：`sum`、`mean`、`max`、`min`、`prod`
- normalization：`softmax`、`layer_norm`、`rms_norm`、`group_norm`
- data movement：`reshape`、`permute`、`transpose`、`slice`、`concat`
- collectives：`all_gather`、`all_reduce`、`reduce_scatter`
- memory：`create_device_tensor`、`deallocate`

这些 `ttnn::*` 操作内部会继续调用 tt-metal 完成：

- 设备内存动态分配；
- program factory/cache；
- kernel 创建；
- runtime args 设置；
- command queue 提交。

即 TTNN 分支的边界是：

```text
tt-runtime ProgramExecutor
 → ttnn C++ operation API
 → tt-metal
 → driver/device
```

它不像 TTMetal 分支那样由 tt-runtime 自己直接构造每个 `tt_metal::Program`。

---

## Distributed 后端调用链

第一级调用：

```text
tt::runtime::distributed::submit(...)
```

见 [distributed/runtime.cpp (line 181)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/distributed/runtime.cpp:181)。

它调用：

```text
ControllerSingleton::get().submit(...)
```

然后 `Controller::submit()`：

1. 创建 FlatBuffer command builder；
2. 生成输出 Tensor handles；
3. 调用 `CommandFactory::buildSubmitCommand(...)`；
4. 调用 `pushToCommandAndResponseQueues(...)`；
5. 将 submit command 发给 worker。

实现见 [controller.cpp (line 598)](/home/yyl/workspace/llmss/engine/vllm/tt-mlir/runtime/lib/distributed/controller/controller.cpp:598)。

worker 收到命令后，最终重新调用本地：

```text
tt::runtime::submit(...)
```

然后落到 worker 上的 TTNN 或 TTMetal 后端。

调用链为：

```text
上层
 → distributed::submit()
 → Controller::submit()
 → buildSubmitCommand()
 → controller/worker command queue
 → worker CommandExecutor
 → worker 本地 tt::runtime::submit()
 → TTNN 或 TTMetal
```

---

## 最终汇总

`runtime.cpp::submit()` 的直接调用只有：

```text
ttnn::submit
ttmetal::submit
distributed::submit
```

继续展开后：

```text
TTNN:
ProgramExecutor
 → operations::*::run
 → ttnn::{matmul, conv2d, empty, deallocate, ...}
 → tt-metal
```

```text
TTMetal:
executeMeshDeviceProgram
 → MCQExecutor
 → MeshBuffer::create
 → CreateProgram/CreateKernel
 → SetRuntimeArgs/CreateCircularBuffer
 → EnqueueMeshWorkload
 → tt-metal dispatch
```

```text
Distributed:
Controller::submit
 → SubmitCommand
 → worker
 → worker 本地 TTNN/TTMetal submit
```

如果只关心“绕开 TTNN、直接对接 tt-metal”的静态编译路径，那么核心代码就是：

```text
runtime.cpp::submit
 → ttmetal::submit
 → executeMeshDeviceProgram
 → MCQExecutor
 → tt_metal::EnqueueMeshWorkload
```
