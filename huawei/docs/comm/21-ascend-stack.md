# ascend详情

> 来源：[飞书知识库原文](https://terapines.feishu.cn/wiki/AXSsweATsikVUqkHEWDckwQgnLb)  
> 迁移版本：revision 36；飞书最后更新时间：2026-09-03T05:21:35Z  
> 本地同步日期：2026-09-08

可以按“**独立项目/仓库**”来拆。最核心的主链不是“vLLM → CANN 一个大包”，而是下面这些项目逐层接起来：

```text
┌───────────────────────────────────────┐
                    【模型 / Serving 层】

┌───────────────────────────────────────┐
│ 1. vLLM                              │
│ Serving / Scheduler / KV / Model     │
└────────────────┬──────────────────────┘
                 │ Hardware Plugin API
                 ▼
┌───────────────────────────────────────┐
│ 2. vLLM-Ascend                       │
│ Ascend Platform / Worker / Attention │
│ KV / Graph / Custom Op               │
└────────────────┬──────────────────────┘
                 ▼
┌───────────────────────────────────────┐
│ 3. PyTorch                           │
│ Tensor / ATen / Dispatcher / FX      │
└────────────────┬──────────────────────┘
                 ▼
┌───────────────────────────────────────┐
│ 4. torch_npu                         │
│ PyTorch NPU Backend                  │
│ Device / Allocator / Stream / Event  │
│ 算子适配 / NPUGraph / HCCL入口       │
└───────────────┬───────────────────────┘
                │
        ┌───────┴────────────────┐
        ▼                        ▼
┌───────────────────┐    ┌──────────────────┐
│ 5a. CANN Ops      │    │ 5b. HCCL         │
│ ACLNN / OP API    │    │ Collective通信   │
│ 算子Host侧实现    │    │ AR/AG/RS/A2A     │
└─────────┬─────────┘    └────────┬─────────┘
          │                       │
          └───────────┬───────────┘
                      ▼

                    【运行时层】

┌───────────────────────────────────────┐
│ 6. CANN Runtime                      │
│ Device / Memory / Stream / Event     │
│ BinaryLoad / GetFunction             │
│ Kernel Launch / Task / ACLGraph      │
│ ModelRI                              │
└────────────────┬──────────────────────┘
                 │ Driver HAL
                 ▼
┌───────────────────────────────────────┐
│ 7. CANN Driver                       │
│ SQ/CQ / TRS / TS Agent / SVM        │
│ Resource / Queue / Device Driver     │
└────────────────┬──────────────────────┘
                 ▼
┌───────────────────────────────────────┐
│ 8. Firmware / Task Scheduler         │
│ 消费SQE / 调度计算、DMA、通信任务    │
└────────────────┬──────────────────────┘
                 ▼
┌───────────────────────────────────────┐
│ 9. NPU Hardware                      │
│ AI Core / Cube / Vector / MTE / HBM  │
│ DMA / Interconnect                   │
└───────────────────────────────────────┘

```

其中 **TorchAir/GE 是图编译执行的旁路分支**，不是所有 vLLM 请求都必须经过。

```text


             【Kernel 编译链：旁路进入 Runtime】

┌───────────────────────────────────────┐
│ A. Ascend C / 自定义 Kernel 源码     │
└────────────────┬──────────────────────┘
                 ▼
┌───────────────────────────────────────┐
│ B. Ascend C SDK / asc-devkit         │
│ Kernel开发API / intrinsic / 编程模型 │
└────────────────┬──────────────────────┘
                 ▼
┌───────────────────────────────────────┐
│ C. CANN / Ascend Kernel Compiler     │
│ 编译、优化、代码生成                 │
└────────────────┬──────────────────────┘
                 ▼
        Kernel Binary / ELF
                 │
                 ▼
         CANN Runtime
      aclrtBinaryLoad()
      BinaryGetFunction()
                 │
                 ▼
             funcHandle
                 │
                 ▼
          aclrtLaunchKernel()
                 │
                 ▼
             Task / SQE



           【图编译链：可选，不是所有执行必经】

PyTorch FX / torch.compile
          │
          ▼
      TorchAir
       ┌──┴────────────┐
       ▼               ▼
  NPUGraph_EX          GE
  FX优化/融合       Graph Engine
       │          图优化/编译/内存规划
       ▼               │
   ACLGraph             │
       └───────┬────────┘
               ▼
          CANN Runtime
               ▼
          CANN Driver
               ▼
              NPU
```

---

# 1. `vllm-project/vllm`

### 它是什么

**通用大模型 Serving Engine。**

它本身不是 Ascend 项目，CUDA、ROCm、Ascend 等后端共享它的上层逻辑。

主要负责：

| 功能 | vLLM 做什么 |
| --- | --- |
| 请求调度 | continuous batching、chunked prefill |
| KV Cache | block 分配、prefix cache、淘汰 |
| Prefill/Decode | 决定当前 step 跑哪些 request/token |
| 模型执行框架 | ModelRunner / Worker |
| 并行 | TP/DP/PP/EP 等框架 |
| Sampling | logits、TopK、TopP 等 |
| Serving | OpenAI API server 等 |
| Hardware Plugin | 给 Ascend 等设备留扩展点 |

它不会自己知道：

```text
Ascend SQE 长什么样
CANN Stream 怎么创建
NPU kernel 怎么 launch
```

这些交给插件和下层。

---

# 2. `vllm-project/vllm-ascend`

这是 **vLLM 和 Ascend 之间最重要的项目**。

官方定义就是：

> vLLM Ascend 是 vLLM 的 Ascend NPU hardware plugin，通过 hardware-pluggable interface 将 Ascend 支持与上游 vLLM 解耦。([GitHub](https://github.com/vllm-project/vllm-ascend?utm_source=chatgpt.com))

它实际通过 Python entry point 注册：

```Python
"vllm.platform_plugins": [
    "ascend = vllm_ascend:register"
]
```

还有 KV connector、model loader、model 等插件入口。(GitHub)

### 它负责什么

可以理解成：

> **把 vLLM 的抽象翻译成 Ascend 能高性能执行的实现。**

例如：

| vLLM 抽象 | vLLM-Ascend 做什么 |
| --- | --- |
| Platform | 定义 Ascend Platform |
| Worker | Ascend Worker |
| ModelRunner | Ascend 执行逻辑 |
| Attention backend | Ascend Attention / MLA backend |
| KV Cache | Ascend KV layout/metadata |
| Custom op | Ascend 高性能算子 |
| Graph | ACLGraph / NPUGraph 相关适配 |
| Distributed | HCCL 适配 |
| Device capability | A2/A3/950 等硬件能力判断 |

这里有一个很重要的事实：

**vLLM-Ascend 不一定所有操作都经过 `torch_npu`。**

它自己的 C++ extension 明确直接链接：

```text
torch_npu
ascendcl
opapi
tiling_api
...
```

代码还会直接检测：

```C++
aclrtMemcpyBatchAsync
```

这样的 CANN Runtime API。(GitHub)

所以实际存在两种：

```text
vLLM-Ascend
    ↓
torch_npu
    ↓
CANN
```

以及：

```text
vLLM-Ascend custom C++
    ↓
ACL / OP API
    ↓
CANN
```

---

# 3. `pytorch/pytorch`

这个就是官方 PyTorch。

在这条链里主要负责：

```text
Tensor
nn.Module
ATen
Dispatcher
torch.compile
FX / Dynamo
Autograd（训练）
```

比如模型里：

```Python
x = torch.matmul(a, b)
```

首先还是：

```text
Python
 ↓
PyTorch
 ↓
ATen / Dispatcher
```

PyTorch Dispatcher 根据：

```text
Tensor device = npu
```

选择 NPU backend。

TorchNPU 官方架构文档也直接把 PyTorch Core 描述为负责：

> autograd、nn.Module、Optimizer、DataLoader、**Dispatcher 算子分发**等基础设施。([GitHub](https://github.com/Ascend/pytorch/blob/master/docs/zh/user_guide/product_overview.md?utm_source=chatgpt.com))

然后才进入 `torch_npu`。

---

# 4. `Ascend/pytorch`：`torch_npu`

仓库名容易产生误解：

```text
项目仓：
Ascend/pytorch

Python包：
torch_npu
```

官方项目定义就是：

> Ascend Extension for PyTorch，使 PyTorch 可以使用 Ascend NPU。([GitHub](https://github.com/Ascend/pytorch/blob/master/README.md?utm_source=chatgpt.com))

它就是我们之前一直讲的：

> **PyTorch Device Backend / Adapter。**

### 它主要负责

```text
PyTorch
   │
   ▼
torch_npu
├── NPU Device
├── Tensor backend
├── ATen算子适配
├── NPUCachingAllocator
├── Stream
├── Event
├── Device
├── HCCL
├── NPUGraph
├── torch.compile backend
└── C++ extension
```

TorchNPU 官方现在自己给出的分层就是：

```text
PyTorch Core
    ↓
TorchNPU Python
    ↓
torch_npu._C
    ↓
CANN
```

其中 C++ 层包括：

> Tensor 基础设施、内存分配器、算子执行框架、HCCL、Inductor backend 等。([GitHub](https://github.com/Ascend/pytorch/blob/master/docs/zh/user_guide/product_overview.md?utm_source=chatgpt.com))

所以普通：

```Python
torch.matmul(npu_tensor)
```

大致：

```text
PyTorch Dispatcher
        ↓
torch_npu
        ↓
对应 NPU 算子
        ↓
CANN
```

---

# 5. `cann-ops` / `ops-nn` / 高性能算子库

这里不要和 Runtime 混在一起。

它们负责的是：

> **“算什么”。**

Runtime 负责：

> **“怎么把这个计算任务发给设备”。**

比如 `cann-ops` 官方定义就是：

> 基于 Ascend 硬件的基础算子仓库，主要使用 Ascend C 开发算子。([Gitee](https://gitee.com/ascend/cann-ops/blob/master/QuickStart.md?skip_mobile=true&utm_source=chatgpt.com))

里面有：

```text
activation
conv
matmul
norm
quant
pooling
...
```

一个算子通常包括：

```text
op_host/
    ↓
tiling / shape / metadata

op_kernel/
    ↓
AscendC kernel
```

所以例如：

```text
torch.matmul
    ↓
torch_npu
    ↓
aclnn / opapi
    ↓
MatMul算子实现
    ↓
CANN Runtime launch
```

它不是 Driver。

---

# 6. `cann/hccl`

这是另外一个独立项目：

> **Huawei Collective Communication Library。**

负责：

```text
AllReduce
AllGather
ReduceScatter
AllToAll
Broadcast
P2P
...
```

支持：

```text
HCCS
RoCE
PCIe
```

而且现在同时支持：

```text
单算子模式
+
图模式
```

官方仓已经明确这些功能。(GitCode)

所以多卡：

```text
vLLM TP
   ↓
torch.distributed / vLLM communicator
   ↓
torch_npu
   ↓
HCCL
   ↓
CANN / Driver
   ↓
HCCS/RoCE/PCIe
```

HCCL 是一条**通信旁路**，不是所有 compute kernel 都经过 HCCL。

---

# 7. `cann/runtime`

这是你现在最值得研究的项目。

官方定义非常直接：

> CANN Runtime 提供 Ascend NPU 运行时用户编程接口和核心实现，包括设备管理、Stream、Event、内存管理、任务调度等。([GitCode](https://gitcode.com/cann/runtime/blob/master/README.md?utm_source=chatgpt.com))

它的职责大概：

```text
CANN Runtime
│
├── Device
├── Context
├── Stream
├── Event
├── Memory
├── Kernel
├── Task
├── Launch
├── Engine
├── Notify
├── ACLGraph
└── Driver Adapter
```

官方架构目录就是这样组织的。(GitCode)

### Runtime 最大的职责

上面给它：

```text
“在 stream X 上执行 kernel K”
```

Runtime 把它组织成：

```text
Kernel
 ↓
Task
 ↓
SQE
 ↓
Driver
```

另外 Runtime 管：

```text
aclrtMalloc
aclrtMemcpy
aclrtCreateStream
aclrtCreateEvent
aclrtLaunchKernel
...
```

以及我们刚才研究的：

```text
ACLGraph
CaptureModel
ModelRI
```

官方 Runtime 样例甚至直接覆盖：

```text
Device
Stream
Event
Memory
Kernel execution
ACL Graph
ModelRI
```

(GitCode)

---

# 8. `cann/driver`

https://gitcode.com/cann/driver/blob/master/README.md?utm_source=chatgpt.com

这就是 Runtime 再往下一层。

官方项目定位：

> CANN 的驱动模块，提供基础驱动、资源管理和调度等能力，使能 Ascend 芯片。包括 DCMI、HAL、SDK-driver。([GitCode](https://gitcode.com/cann/driver/blob/master/README.md?utm_source=chatgpt.com))

结构已经非常有意义：

![图片展示了`sdk_driver`的结构。上方有`DCMI`和`ascend_hal`。中间是`sdk_driver`，包含设备管理、资源管理、通信管理、基础库&驱动等部分。设备管理有DMS、FMS等；资源管理有SVM、TRS等；通信管理有HDC、VNIC、VPC等；基础库&驱动有comm、RoCE、pbl等。最下方是`kernel_adapt`。该图与上下文关系紧密，是对`sdk_driver`结构的直观呈现，帮助理解其组成部分及其层级关系。](https://feishu.cn/file/PibAbKEfnobJijxT54pckT6vnMU)

```Bash
├── build.sh                                       # 项目工程编译脚本
├── cmake                                          # 工程编译目录
├── CMakeLists.txt                                 # 项目工程CMakeLists入口
├── CONTRIBUTING.md                                # 社区贡献指导
├── docs                                           # 说明文档
├── examples                                       # 接口使用样例
├── pkg_inc                                        # 本仓对外提供的头文件
├── LICENSES                                       # 本仓涉及协议目录
├── OAT.xml                                        # 配置脚本，代码仓工具使用，用于检查License是否规范
├── README.md
├── scripts                                        # 本仓脚本目录
│   ├── package                                    # 构建打包相关脚本
│   ├── ut                                         # UT生成cpp覆盖率脚本
├── SECURITY.md                                    # 项目安全声明文件
├── Third_Party_Open_Source_Software_Notice        # 本仓引用的第三方开源软件声明
├── src                                            # Driver包源码
│   ├── ascend_hal                                 # HAL层源码文件夹
│   │   ├── bbox                                   # 黑匣子（Black Box，系统临终遗言）
│   │   ├── buff                                   # 进程间共享内存管理
│   │   ├── build                                  # ascend_hal动态库编译脚本
│   │   ├── comm                                   # Communication 主机侧<->设备侧通信层
│   │   ├── dmc                                    # DMC（Device Maintenance Components）设备维护组件
│   │   │   ├── device_monitor                     # DSMI消息通路
│   │   │   ├── dsmi                               # DSMI（Device System Management Interface）设备系统管理接口
│   │   │   ├── logdrv                             # Log日志
│   │   │   ├── prof                               # Profiling性能采集
|   |   |   ├── prof_sample                        # Profiling Host侧采集注册
│   │   │   └── verify_tool                        # 设备侧镜像校验工具
│   │   ├── dms                                    # DMS（Device Management System）设备管理系统
│   │   ├── dpa                                    # DPA（Device Public Adapter）设备公共适配层
│   │   ├── esched                                 # 事件调度（Event Schedule）
│   │   ├── hdc                                    # 主机-设备通信（Host-Device Communication）
│   │   ├── inc                                    # HAL层内部公共头文件目录
│   │   ├── mmpa                                   # MMAP（Medium Multiple Platform Adaptive）基础系统接口库
│   │   ├── msnpureport                            # 设备侧维测信息导出工具
│   │   ├── pbl                                    # PBL（Public Base Lib）基础公共库
│   │   │   ├── uda                                # UDA（Unified Device Access）统一设备接入
│   │   │   ├── urd                                # URD（User Request Distribute）用户请求转发
│   │   │   ├── commlib                            # 公共函数库
│   │   │   ├── queryfeature                       # 用于兼容性适配的软件特性查询
|   |   |   └── ubmm                               # UB Memory Adapter
│   │   ├── queue                                  # 消息队列信息管理
│   │   ├── roce                                   # RoCE（RDMA over Converged Ethernet）
│   │   ├── svm                                    # 共享虚拟内存（Shared Virtual Memory）
│   │   └── trs                                    # 任务资源调度（Task Resource Schedule）
│   ├── custom                                     # 定制化特性源码库
│   │   ├── cmake                                  # CMake编译配置目录
│   │   ├── dev_prod                               # 设备定制管理目录
│   │   ├── include                                # 公共头文件导出目录
│   │   ├── lqdrv                                  # 灵渠PCIe故障检测
│   │   ├── ndr                                    # NPU RDMA直通特性
│   │   ├── network                                # DCMI网络接口实现
│   │   └── ops_debug                              # 算子诊断目录
│   └── sdk_driver                                 # SDK层源码文件夹
│       ├── buff                                   # 进程间共享内存管理
│       ├── comm                                   # Communication 主机侧<->设备侧通信层
│       ├── dmc                                    # DMC（Device Maintenance Components）设备维护组件
│       ├── dms                                    # DMS（Device Management System）设备管理系统
│       ├── dpa                                    # DPA（Device Public Adapter）设备公共适配层
│       ├── dvpp                                   # DVPP（Digital Vision Pre-Processing）数字视觉预处理模块
│       ├── esched                                 # 事件调度（Event Schedule）
│       ├── fms                                    # FMS（Fault Management System）故障管理系统
│       ├── hdc                                    # 主机-设备通信（Host-Device Communication）
│       ├── inc                                    # SDK层内部公共头文件目录
│       ├── kernel_adapt                           # SDK驱动代码与内核源码适配层
│       ├── pbl                                    # PBL（Public Base Lib）基础公共库
│       ├── platform                               # 芯片资源（中断、预留内存等）存储库
│       ├── queue                                  # 消息队列信息管理
|       ├── seclib                                 # 公共安全函数库（Secure Library）
│       ├── svm                                    # 共享虚拟内存（Shared Virtual Memory）
│       ├── ts_agent                               # TS（Task Schedule）代理驱动源码
│       ├── trsdrv                                 # TRS（Task Resource Schedule）软件sqcq通信、mailbox消息特性
│       │   ├── trs                                # 任务资源调度（Task Resource Schedule）
│       │   └── trsbase                            # 任务资源调度（Task Resource Schedule）基础层
│       ├── vascend                                # 昇腾算力切分特性
│       ├── vmng                                   # 设备虚拟化管理（Virtual Machine Manager）
│       ├── vnic                                   # VNIC（Virtual Network Interface Card）虚拟网卡
│       └── vpc                                    # VPC（Virtual Physical Communication）物理机与虚拟机通信
└── test                                           # UT用例文件目录

```



官方 README 对：

```text
ts_agent
trsdrv
```

定义分别就是：

```text
TS = Task Schedule 代理驱动

TRS =
Task Resource Schedule
software SQ/CQ communication
mailbox message
```

(GitCode)

---

# Runtime → Driver 的边界现在已经很明确

我们刚才查到的 HAL：

```C
halSqTaskSend(...)
halCqReportRecv(...)

halStreamTaskFill(...)

halSqSwitchStreamBatch(...)
```

就是：

```text
CANN Runtime
       ↓
Driver HAL
       ↓
CANN Driver
```

的真实接口。

官方注释分别写着：

```text
SQ task send
CQ report recv
stream task fill
SQ bind stream
```

(GitCode)

所以这一段现在可以比较确定地画：

```text
CANN Runtime
     │
     │ Task / SQE
     ▼
Driver HAL
     │
     ├─ halSqTaskSend
     ├─ halCqReportRecv
     ├─ halStreamTaskFill
     └─ halSqSwitchStreamBatch
     │
     ▼
CANN Driver
     │
     ├─ TRS
     ├─ SQ/CQ
     ├─ TS Agent
     ├─ mailbox
     └─ SVM
     │
     ▼
Firmware / TS
     │
     ▼
NPU
```

---

# 10. `TorchAir` 是什么位置

这是一个**可选图模式项目**，不要放在 Eager 主链里面。

项目：

```text
Ascend/torchair
```

官方定义：

> TorchAir 是 `torch_npu` 的图模式扩展库，为 `torch.compile` 提供 Ascend backend。([GitHub](https://github.com/Ascend/torchair/blob/master/docs/zh/overview.md?utm_source=chatgpt.com))

它现在有两条主要路线：

```text
PyTorch FX
   ↓
TorchAir
   ├──────────────┐
   ▼              ▼
npugraph_ex       Ascend IR
   │              │
ACLGraph          GE
   │              │
Runtime           Runtime
```

官方文档明确区分：

### `npugraph_ex`

```text
FX优化
 ↓
ACLGraph Capture/Replay
 ↓
CANN Runtime
```

(GitHub)

### GE

```text
FX
 ↓
Ascend IR
 ↓
GE compile
 ↓
GE execute
```

(GitHub)

---

# 11. `cann/ge`

现在 GE 也已经是独立开源项目。

**GE = Graph Engine。**

官方定义：

> 面向 Ascend 的图编译器和执行器，负责图优化、多 Stream 并行、内存复用、模型下沉和图执行。([GitCode](https://gitcode.com/cann/ge/tree/develop?utm_source=chatgpt.com))

所以它的位置：

```text
PyTorch
 ↓
torch.compile
 ↓
TorchAir
 ↓
GE
 ↓
CANN Runtime
 ↓
CANN Driver
```

这里和 ACLGraph 不一样：

```text
ACLGraph
主要是capture/replay已有task

GE
是真正的图编译器/图执行器
```

GE 官方项目目录本身都有：

```text
compiler/
runtime/
parser/
graph_metadef/
```

(GitCode)

---

# 最后，把所有项目放到一张图里

这是目前我认为最清楚的项目边界：

```text
                       Application
                           │
                           ▼
┌────────────────────────────────────────────┐
│                 vLLM                       │
│ Scheduler / KV / Model / Serving / Parallel│
└──────────────────────┬─────────────────────┘
                       │ plugin
                       ▼
┌────────────────────────────────────────────┐
│              vLLM-Ascend                   │
│ Platform / Worker / Attention / KV / Graph │
│ Ascend custom kernels / communication      │
└─────────────┬────────────────┬─────────────┘
              │                │
       PyTorch Tensor          │ direct custom op
              │                │
              ▼                │
┌──────────────────────┐       │
│       PyTorch        │       │
│ ATen / Dispatcher    │       │
│ Dynamo / FX          │       │
└──────────┬───────────┘       │
           ▼                   │
┌──────────────────────┐       │
│      torch_npu       │◄──────┘
│ PyTorch NPU backend  │
└───┬──────────┬───────┘
    │          │
    │          └───────────────┐
    │                          │
    ▼                          ▼
┌──────────────┐        ┌──────────────┐
│ CANN ops     │        │  TorchAir    │
│ ops-nn/adv   │        │ torch.compile│
└──────┬───────┘        └──────┬───────┘
       │                 ┌─────┴─────┐
       │                 ▼           ▼
       │             ACLGraph       GE
       │                 │           │
       └────────────┬────┴───────────┘
                    ▼
┌────────────────────────────────────────────┐
│              CANN Runtime                  │
│ Device / Memory / Stream / Event / Kernel  │
│ Task / SQE / ACLGraph / ModelRI            │
└──────────────────────┬─────────────────────┘
                       │ HAL
                       ▼
┌────────────────────────────────────────────┐
│               CANN Driver                  │
│ HAL / SQ / CQ / TRS / TS Agent / SVM       │
│ Resource / Queue / Mailbox / Device Driver │
└──────────────────────┬─────────────────────┘
                       ▼
                 Firmware / TS
                       ▼
                  Ascend NPU
```

通信再旁挂：

```text
torch_npu / vLLM-Ascend
          ↓
         HCCL
          ↓
  HCCS / RoCE / PCIe
          ↓
 Driver / Hardware
```

---

## 对你现在做硬件最重要的是这几个项目

如果你的目标是**自己设计一个能支撑 vLLM Eager + Graph Replay 的 NPU 软件/硬件接口**，研究优先级可以直接定成：

1. **`cann/runtime`**：看 Framework/Runtime 最终需要哪些执行能力。([GitCode](https://gitcode.com/cann/runtime/blob/master/README.md?utm_source=chatgpt.com))
2. **`cann/driver`**：看 Runtime→Driver/HW 接口，尤其 `SQ/CQ/TRS/TS/SVM`。([GitCode](https://gitcode.com/cann/driver/blob/master/README.md?utm_source=chatgpt.com))
3. **`Ascend/pytorch` / torch_npu**：看 PyTorch Device backend 需要 Runtime 提供什么。([GitHub](https://github.com/Ascend/pytorch/blob/master/docs/zh/user_guide/product_overview.md?utm_source=chatgpt.com))
4. **`vllm-ascend`**：看真正 Serving 场景还需要 Attention/KV/Graph/通信哪些特殊能力。([GitHub](https://github.com/vllm-project/vllm-ascend?utm_source=chatgpt.com))
5. **`cann-ops` + HCCL**：分别补齐计算 kernel 和多芯片通信。([Gitee](https://gitee.com/ascend/cann-ops/blob/master/QuickStart.md?skip_mobile=true&utm_source=chatgpt.com))

真正和你们**硬件接口定义最直接相关的边界**其实是：

```text
torch_npu / vLLM-Ascend
          ↓
      CANN Runtime
════════════════════════   ← 软件 Runtime
      Driver HAL
          ↓
      CANN Driver
════════════════════════   ← OS/Driver
      SQ/CQ + TS
          ↓
       Hardware
```

下一步如果继续往下挖，最有价值的就是直接从 **`aclrtLaunchKernel → Runtime Task → SQE → halSqTaskSend → TRS/SQ → TS`** 追一条真实源码调用链，这样可以直接反推出你们需要实现的 Driver/HW 接口。
