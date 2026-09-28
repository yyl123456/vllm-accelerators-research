# 昇腾计算图捕获与重放（Capture & Replay）架构深度剖析

## 1. 概述与核心价值

在以大语言模型（LLM）推理为代表的高吞吐、低延迟计算负载中，自回归解码（Auto-regressive Decode）阶段呈现出 **“计算时间极短（Microsecond 级）、小 Batch、Kernel 发射密集”** 的典型特征。
在传统的 PyTorch Eager 动态执行模式下，Host CPU 需要为每一个算子依次完成 Python 字节码解析、Dispatcher 查找、参数打包、校验并下发给底层驱动，导致 **CPU 调度与下发延迟（Launch Latency）远大于 NPU 设备端的实际计算耗时（Execution Time）**，出现严重的 **Launch-Bound（CPU 发射瓶颈）**。

为了彻底消除 Host 侧开销，昇腾软件栈借鉴并深度重构了类似 CUDA Graph 的硬件级执行流录制与重放机制——**ACLGraph / NPUGraph**。

### 1.1 Capture & Replay 与 Eager 模式的核心差异与优势

| 维度 | 原生 Eager 模式 | Capture & Replay (NPUGraph) 模式 |
| :--- | :--- | :--- |
| **下发机制** | Host 端 Python/C++ 逐个算子解释并同步/异步 Launch 到驱动 | **一次性全硬件录制，后续一键重放（Single Doorbell Launch）** |
| **CPU 参与度** | 持续 100% 满载打满 CPU，每个 Step 发射数十上百次 | **首次录制后 CPU 彻底休眠/解放**，仅需极轻量触发重放 |
| **下发开销 (Launch Overhead)**| 每个算子 3~10 微秒，一个 Step 累计延迟可达数百微秒甚至毫秒级 | **整个计算图发射延迟降至 1~2 微秒**（纯单次 MMIO 开销） |
| **内存管理** | 运行时动态从 `NPUCachingAllocator` 申请与释放，有切分合并开销 | **静态专用内存池（Private MemPool）**，拓扑地址完全固定，零动态分配 |
| **多流与硬件流水** | 依赖 Host 线程通过 Event 协调多流同步，存在抖动 | **流间依赖（Stream Event）完全固化至硬件执行拓扑**，硬件直接流转 |
| **硬件优化空间** | NPU 只能看到局部单个算子，无法做全局流水编排 | 底层 CANN 可对整张图进行 **SuperKernel 融合与全局流水优化** |

---

## 2. 用户视角：如何选择与开启 Capture & Replay

用户在不同业务层次，有三种典型的开启与选择路径：

### 2.1 路径 A：声明式整图编译与优化（`torch.compile`）
面向标准 PyTorch 模型开发与通用推理，最简便直观：

```python
import torch
import torch_npu

model = MyModel().npu()

# 方式 1：Inductor 算子融合 + NPUGraph 录制重放（最强组合，推荐）
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="reduce-overhead"  # 核心开关：指示 Inductor 在生成代码外层自动包装 NPUGraph
)

# 方式 2：纯 ACLGraph 原生算子录制重放（不重新编译 Kernel，只消除 CPU 开销）
compiled_model_pure = torch.compile(
    model,
    backend="npugraphs"
)
```

### 2.2 路径 B：显式手动录制 API（`torch_npu.npu.NPUGraph`）
面向对执行流有极致掌控欲的推理引擎开发者（类似 CUDA Graph 的直接手动控制）：

```python
import torch
import torch_npu

# 1. 准备静态输入输出缓冲区（地址必须在 Capture 与 Replay 间保持完全恒定）
static_input = torch.randn(1, 4096, device="npu")
static_output = torch.empty(1, 4096, device="npu")

# 2. Warmup 消除首次分配与底层初始化干扰
s = torch_npu.npu.Stream()
s.wait_stream(torch_npu.npu.current_stream())
with torch_npu.npu.stream(s):
    for _ in range(3):
        static_output = model(static_input)
torch_npu.npu.current_stream().wait_stream(s)

# 3. 开启图录制 (Capture)
g = torch_npu.npu.NPUGraph()
with torch_npu.npu.graph(g, stream=s):
    static_output = model(static_input)

# 4. 后续推理执行：仅需拷贝输入并触发重放 (Replay)
static_input.copy_(new_input)
g.replay()  # 零 CPU Launch 开销！
```

### 2.3 路径 C：大模型服务化框架参数（vLLM-Ascend）
在部署大语言模型推理时，`vllm-ascend` 默认且深度集成了强化版录制重放后端 `npugraph_ex`：

```bash
vllm serve /path/to/model \
    --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true}}'
```

---

## 3. 架构全景与跨项目职责边界

Capture & Replay 绝非单一模块的功能，而是**纵跨四大项目的系统工程**。各项目的职责分工与边界如下：

```text
==================================================================================================
                        Capture & Replay 跨项目架构全景与协作边界
==================================================================================================

[ 项目 1：应用与服务化层 (vllm-ascend / PyTorch 前端) ]
  ├── 用户入口: torch.compile(..., mode="reduce-overhead") 或 model_runner.capture_model()
  ├── 静态内存固定: 为 KV Cache、Weights、Input/Output 固定持久物理地址
  └── 自定义融合 Pass: vllm_ascend/compilation/passes/ (注入大模型融合规则)
                                │
                                ▼ (通过标准 FX Graph 传递)
[ 项目 2：图模式扩展与编译器层 (torchair / npugraph_ex) ]
  ├── 核心文件: torchair/npugraph_ex/npugraph_ex/npu_fx_compiler.py
  ├── 算子融合: 运行 Pattern Passes (如 Add + RMSNorm + Quant) 生成粗粒度大算子
  └── 静态代码生成: fx2acl_converter.py (生成包含 assert_size_stride 与静态调用的代码)
                                │
                                ▼ (调用底层设备流与图生命周期接口)
[ 项目 3：Ascend 设备适配层 (torch_npu) ]
  ├── 图状态机管理: torch_npu/torch_npu/csrc/core/npu/NPUGraph.cpp (NPUGraph 类)
  ├── 专属私有内存池: NPUCachingAllocator::beginAllocateToPool (录制期内存隔离与保护)
  ├── 任务队列协调: 检查 TASK_QUEUE_ENABLE，防止异步下发打乱硬件录制时序
  └── 随机数与状态保护: captured_generator_states_ (维护 RNG 种子，保证重放数学正确性)
                                │
                                ▼ (通过 C API 彻底交出控制权，下沉调用驱动)
[ 项目 4：底层运行时与硬件驱动 (CANN Runtime - libascendcl.so) ]
  ├── 硬件流捕获: AclmdlRICaptureBegin / AclmdlRICaptureEnd (将 SQ 指令包固化)
  ├── 拓扑固化与优化: AclskOptimize (可选的 SuperKernel 算子间极致硬连线)
  └── 硬件执行发射: AclmdlRIExecuteAsync (向硬件调度器发射静态图句柄)
```

---

## 4. 深度运行机制与端到端调用路线追踪

### 4.1 录制阶段（Capture Phase）：时序与源码深度分析

录制并不是把 Python 代码保存下来，而是**在底层让硬件流进入“监听/抓包模式”，将发往 NPU 的所有硬件任务（SQ 描述符）拦截并固化为一个拓扑 DAG**。

```text
[User / Python]            [torch_npu C++: NPUGraph]          [CANN: libascendcl.so]
      │                                │                                │
      │ g.capture_begin()              │                                │
      ├───────────────────────────────>│                                │
      │                                │-- 1. 校验流与配置 (Line 243-274) │
      │                                │      (非默认流、关闭异步队列等)     │
      │                                │                                │
      │                                │-- 2. 内存池隔离 (Line 296-314)   │
      │                                │      beginAllocateToPool()     │
      │                                │      (录制期间显存永不归还系统)    │
      │                                │                                │
      │                                │-- 3. 下沉开启底层硬件捕获       │
      │                                │      AclmdlRICaptureBegin()   │
      │                                ├───────────────────────────────>│
      │                                │                                │ (硬件 Stream 进入
      │                                │<───────────────────────────────┤  Capture 抓包状态)
      │                                │-- 4. 获取捕获句柄与 ID          │
      │                                │      AclmdlRICaptureGetInfo()  │
      │<───────────────────────────────┤                                │
      │                                │                                │
      │ === 运行业务前向 ===           │                                │
      │ y = model(x) (多个算子下发)    │                                │
      │ (Eager 快速发射指令包)          │                                │
      │                                │                                ├── 拦截每一个 Task 包
      │                                │                                ├── 记录内存输入输出物理地址
      │                                │                                └── 记录流与事件依赖拓扑
      │                                │                                │
      │ g.capture_end()                │                                │
      ├───────────────────────────────>│                                │
      │                                │-- 5. 下沉结束硬件捕获           │
      │                                │      AclmdlRICaptureEnd()      │
      │                                ├───────────────────────────────>│
      │                                │                                │ (固化生成 aclmdlRI
      │                                │<───────────────────────────────┤  静态可执行拓扑)
      │                                │-- 6. 解除内存池锁定 (Line 369)   │
      │                                │      endAllocateToPool()       │
      │                                │-- 7. 标记 has_graph_exec_ = true│
      │<───────────────────────────────┤                                │
```

#### 关键源码依据追踪：
1. **环境与配置防呆（`NPUGraph.cpp:243-274`）**：
   - 捕获必须在**非默认流（Non-default Stream）**上执行；
   - 必须关闭多线程抢占队列（`TASK_QUEUE_ENABLE != 2`），保证下发时序绝对线性和确定性。
2. **静态内存池隔离保护（`NPUGraph.cpp:306`）**：
   - 调用 `c10_npu::NPUCachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, filter)`；
   - **机制本质**：录制期间算子申请的所有临时 Buffer，其物理内存块全部被钉死（Pin）在专用 Pool 中，**显存绝对不能被操作系统或普通分配器回收**，保证在未来 Replay 重放时物理地址永久有效。
3. **底层 CANN 捕获触发（`NPUGraph.cpp:318`）**：
   - 调用 `c10_npu::acl::AclmdlRICaptureBegin(capture_stream_, capture_mode)`；
   - CANN 驱动接管该 Stream，后续该 Stream 上的所有算子 launch 不会真正敲门铃执行，而是被写入静态描述符链表。
4. **结束并固化硬件模型（`NPUGraph.cpp:356-378`）**：
   - 调用 `c10_npu::acl::AclmdlRICaptureEnd(capture_stream_, &model_ri)`；
   - CANN 返回固化后的静态图句柄 `model_ri`（`aclmdlRI`），此时拓扑、参数、内存偏移全部冻结。

---

### 4.2 重放阶段（Replay Phase）：极速发射与执行

在实际推理的死循环中，计算图已经变为一个不可再修改的硬件静态包：

```text
[User / Python]            [torch_npu C++: NPUGraph]          [CANN: libascendcl.so]
      │                                │                                │
      │ 1. 拷贝新数据到静态输入        │                                │
      │    static_input.copy_(x)       │                                │
      │                                │                                │
      │ 2. g.replay()                  │                                │
      ├───────────────────────────────>│                                │
      │                                │-- 3. 推进随机数状态 (Line 399) │
      │                                │      replay_prologue()         │
      │                                │                                │
      │                                │-- 4. 极速发射静态图 (Line 404)  │
      │                                │      AclmdlRIExecuteAsync()    │
      │                                ├───────────────────────────────>│
      │                                │                                │ (单次 MMIO 敲门铃)
      │                                │<───────────────────────────────┤
      │<───────────────────────────────┤                                │
      │ (CPU 立刻返回，继续下一步)     │                                ├── NPU 硬件全自动按拓扑
      │                                │                                │   连续执行所有 Kernel
      │                                │                                └── 结果直接写入静态输出
```

#### 关键源码依据追踪：
- **异步发射入口（`NPUGraph.cpp:392-408`）**：
  ```cpp
  void NPUGraph::replay() {
    TORCH_CHECK(has_graph_exec_, "Called NPUGraph::replay without a preceding successful capture.");
    ...
    auto stream = c10_npu::getCurrentNPUStream();
    // 核心下沉点：一行 C API 将整张图的所有 Kernel 异步投递进硬件
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIExecuteAsync(model_ri_, stream));
  }
  ```
- **延迟对比**：
  如果一个网络有 200 个算子，Eager 模式下 CPU 需要循环执行 200 次 C++ 函数调用、200 次驱动交互；
  而在 `replay()` 中，**CPU 仅执行了 1 次 `AclmdlRIExecuteAsync`，耗时约 1~2 微秒，随后 CPU 即可直接退出该函数处理其他事务，硬件端按预设好的拓扑自行流水推进。**

---

## 5. 局限性、边界条件与工程避坑指南

尽管 Capture & Replay 能带来数倍的性能飞跃，但由于其**硬件固化**的底层本质，它引入了极强的刚性约束：

1. **绝对不支持动态输入尺寸（Dynamic Shape 不兼容）**：
   - **原因**：CANN 底层捕获的算子参数包含固定的 Tensor 形状和 Stride，硬件切块（Tiling）参数在 Capture 结束时已固化。
   - **解决策略**：结合 `torch_npu` 的 `shape_handling` 特性对输入进行分档（Bucket），或在大模型 Decode 阶段使用 Padding 固定 Batch Size。
2. **严禁在 Capture 作用域内出现 CPU-NPU 同步操作**：
   - 在录制期间，**绝对不能调用 `tensor.item()`、`tensor.cpu()`、`print(tensor)` 或 `torch.npu.synchronize()`**。
   - 否则会导致捕获流中断并抛出 `ACL_ERROR_RT_STREAM_CAPTURE_UNSUPPORTED` 致命错误。
3. **输入/输出必须使用固定内存（Static Buffers）**：
   - 重放时，Kernel 依然读取录制时绑定的那块物理显存。
   - 用户必须通过原地拷贝（`static_input.copy_(new_input)`）将新数据送入，**绝不能重新给变量赋一个新初始化的 Tensor（改变了 `data_ptr` 会导致模型读不到新输入）**。
4. **多线程并发限制**：
   - 默认的 `capture_mode = "global"` 会阻止其他线程在该 Device 上分配普通显存，防止录制期间内存池拓扑被破坏。
