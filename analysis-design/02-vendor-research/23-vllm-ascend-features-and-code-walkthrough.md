# 华为 vLLM-Ascend 核心功能特性与底层执行逻辑技术分析

## 1. 项目定位与全景架构

`vllm-ascend` 是针对华为昇腾（Ascend NPU）硬件生态开发的 **Out-of-Tree (OOT)** 硬件加速插件。其核心目标是在**完全不侵入/极少改动上游 vLLM 核心框架代码**的前提下，通过 Python 动态插件机制、猴子补丁（Monkey Patch）、定制化 Worker/ModelRunner、ACL 静态图引擎以及针对达芬奇架构（DaVinci AI Core）优化的算子库，将 vLLM 的大模型推理请求无缝下沉至昇腾计算硬件。

### 1.1 系统软硬件全景架构图

```text
+-----------------------------------------------------------------------------------------+
|                                Upstream vLLM Ecosystem                                  |
|  [LLM Serving API / AsyncEngineCore] <---> [Scheduler / KV Cache Block Manager]        |
+--------------------------------------------+--------------------------------------------+
                                             |
                                    (OOT Plugin Hooks)
                                             |
+--------------------------------------------v--------------------------------------------+
|                          vllm-ascend (Hardware Plugin Layer)                            |
|                                                                                         |
|  +---------------------------+  +--------------------------------+  +----------------+  |
|  |       NPUPlatform         |  |   Worker & Runner Hierarchy    |  |  ACLGraph Eng  |  |
|  | - Plugin Register         |  | - NPUWorker                    |  | - Capture      |  |
|  | - HardwareProfile (A2/A3) |  | - NPUModelRunner (V1/V2)       |  | - Multi-shape  |  |
|  | - Environment / Allocator |  | - AscendMetadata & Padding     |  | - Fast Replay  |  |
|  +---------------------------+  +--------------------------------+  +----------------+  |
|                                                                                         |
|  +---------------------------+  +--------------------------------+  +----------------+  |
|  |     Ascend Attention      |  |     Ascend Ops & Patches       |  | Dist & MoE/EPLB|  |
|  | - attention_v1 (FIA/FA3)  |  | - RMSNorm / RotaryEmbedding    |  | - MC2 Comm     |  |
|  | - mla_v1 / dsa_v1         |  | - Monkey Patch (DeepSeek, etc.)|  | - All2All-V    |  |
|  | - NZ / Paged KV Cache     |  | - Custom Linear (TP/Column/Row)|  | - EPLB Worker  |  |
|  +---------------------------+  +--------------------------------+  +----------------+  |
+--------------------------------------------+--------------------------------------------+
                                             |
                   (ATen Dispatch / PrivateUse1 / Custom C++ / PyHCCL)
                                             |
+--------------------------------------------v--------------------------------------------+
|                            torch_npu (PyTorch Device Backend)                           |
|  - PrivateUse1 Backend Registration (`torch.device("npu")`)                             |
|  - NPU Allocator (Expandable Segments / Caching Allocator)                              |
|  - ATen Op Dispatch & Custom NPU Extensions (`torch_npu.npu_*`)                         |
|  - HCCL Distributed ProcessGroup Backend (`torch.distributed.init_process_group`)       |
+--------------------------------------------+--------------------------------------------+
                                             |
                                 (CANN Driver / Runtime API)
                                             |
+--------------------------------------------v--------------------------------------------+
|                             Huawei CANN Runtime & Drivers                               |
|  - ACL Runtime (`libascendcl.so`, `libacl_op_compiler.so`)                              |
|  - HCCL Communicator (`libhccl.so`)                                                     |
|  - ACLNN / Op-Plugin (ATen to Task Engine translation)                                  |
|  - Ascend Driver & Device HAL (Ascend 910B / 910C / 310P)                               |
+-----------------------------------------------------------------------------------------+
                                             |
                                    (Physical Hardware)
                                             |
+--------------------------------------------v--------------------------------------------+
|                  Ascend AI Core (Cube Unit + Vector Unit) & HBM                         |
+-----------------------------------------------------------------------------------------+
```

---

## 2. 核心功能矩阵清单

| 模块名称 | 功能概述 | 核心技术价值 | 关键源码目录与入口 |
| :--- | :--- | :--- | :--- |
| **插件生命周期与配置** | OOT 平台注册、设备自适应 Profile 匹配、显存分配环境变量控制 | 无缝介入 vLLM 启动流，规避显存碎片与死锁 | `vllm_ascend/platform.py`<br>`vllm_ascend/device/hardware_profile.py` |
| **单步推理执行引擎** | Worker 与 ModelRunner 调度、Batch 打包、Pad 规整与多流发射 | 消除主机端调度瓶颈，支持异构模型高吞吐前向 | `vllm_ascend/worker/worker.py`<br>`vllm_ascend/worker/model_runner_v1.py` |
| **ACLGraph 图执行** | NPU 静态图捕获、分段（Piecewise）回放、参数流更新 | 完全绕过 Python 调度与 Runtime 任务重下发，大幅降低 Decode 延迟 | `vllm_ascend/compilation/acl_graph.py`<br>`vllm_ascend/worker/v2/aclgraph_utils.py` |
| **融合 Attention 与 KV** | 融合注意力（FIA/FA3/MLA）、NZ 物理格式转换、分页 Block 表管理 | 发挥 Cube 矩阵乘与 Vector 矢量单元吞吐，榨干片上带宽 | `vllm_ascend/attention/attention_v1.py`<br>`vllm_ascend/attention/mla_v1.py` |
| **算子替换与 Monkey Patch** | 替换 RMSNorm、SwiGLU、RoPE，动态打补丁支持 DeepSeek/Qwen 特殊结构 | 无需重写上游模型骨干网络，直接享有 NPU 极速算子 | `vllm_ascend/ops/`<br>`vllm_ascend/patch/` |
| **分布式通信与动态 MoE** | HCCL 封装、MC2 融合通信、All2All-V、EPLB 动态负载均衡 | 彻底消除跨机跨卡 AllToAll 通信气泡，动态消除热点专家倾斜 | `vllm_ascend/ops/fused_moe/`<br>`vllm_ascend/eplb/`<br>`vllm_ascend/distributed/` |

---

## 3. 插件注册与生命周期管理（Platform & Worker 初始化）

### 3.1 OOT 插件注册入口
上游 vLLM 依托 Python 的 `entry_points` 机制发现第三方平台扩展。在 `huawei/vllm-ascend/setup.py`（Line 519）：
```python
"vllm.platform_plugins": ["ascend = vllm_ascend:register"],
```
在引擎启动时，上游 `vllm.platforms.load_plugins()` 触发 `vllm_ascend/__init__.py:register()`（Line 68-71），返回字符串：
```python
def register():
    return "vllm_ascend.platform.NPUPlatform"
```
从而将 `NPUPlatform` 注入为全局唯一的 `current_platform`。

### 3.2 硬件能力画像：`HardwareProfile`
在 `vllm_ascend/device/hardware_profile.py` 中，定义了昇腾芯片系列的能力画像机制：
- **能力枚举**：`HardwareCapability`（Line 16-60），包含 `PAGED_ATTENTION`、`AUTO_ENABLE_CUSTOM_OPS`、`MC2_HIERARCHY_COMM`、`CANN_MEGAMOE` 等数十项细粒度硬件支持位。
- **设备类型分流**：`_HARDWARE_PROFILES`（Line 158-205）将 Ascend 芯片分为：
  - **A2 系列（如 910B）**：`CPUBindingMode.TOPO_AFFINITY`、`DeviceAddressingMode.DIRECT`、默认 Worker `vllm_ascend.worker.worker.NPUWorker`。
  - **A3 系列（如 910C）**：`CPUBindingMode.GLOBAL_SLICE`、`DeviceAddressingMode.DUAL_CHIP_CARD`、使能 `CANN_MEGAMOE`。
  - **310P 系列**：`default_worker_cls="vllm_ascend._310p.worker_310p.NPUWorker310"`、强制 `WeightLayoutPolicy.FORCE_NZ`。

### 3.3 接管 Worker 类别与内存分配策略
在 `vllm_ascend/platform.py` 中：
1. **配置校验与修改入口**：`NPUPlatform.check_and_update_config()`（Line 441-496）是配置接入的心脏，它依次完成环境变量注入、模型配置修复、编译模式适配以及 Worker/Scheduler 分发。
2. **Worker 类绑定**：在 `_setup_worker_and_scheduler()`（Line 1230-1254）中：
   ```python
   if parallel_config and parallel_config.worker_cls == "auto":
       hardware_profile = get_current_hardware_profile()
       parallel_config.worker_cls = hardware_profile.default_worker_cls
   ```
   将 `parallel_config.worker_cls` 明确绑定为 `vllm_ascend.worker.worker.NPUWorker`。
3. **NPU 内存分配器配置**：在 `_set_pytorch_npu_alloc_env()`（Line 1322-1342）：
   ```python
   if vllm_config.model_config and not vllm_config.model_config.enable_sleep_mode:
       npu_alloc_configs = os.getenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
       if ("expandable_segments" not in npu_alloc_configs ...):
           npu_alloc_configs += ",expandable_segments:True"
       os.environ["PYTORCH_NPU_ALLOC_CONF"] = npu_alloc_configs
   ```
   强制为 `torch_npu` 注入 `expandable_segments:True`（虚拟内存段扩展，类似 CUDA 的虚拟内存寻址），从根源上阻断反复申请释放大块连续显存造成的虚拟内存碎片 OOM。

---

## 4. 单步推理执行全景剖析（`execute_model` 逐行级透视）

单步推理的核心链路体现为：`SchedulerOutput` 抵达 -> Worker 协调流水线 -> ModelRunner 组织 Token 与元数据 -> 静态图/Eager 判定 -> 前向计算 -> 采样与通信。

### 4.1 `NPUWorker.execute_model()` 流程（`vllm_ascend/worker/worker.py` Line 643-710）

```text
[SchedulerOutput 传入]
         |
         v
+-------------------------------------------------------------+
| 1. 等待上一轮异步 PP 发送完成: self._pp_send_work.wait()    | (Line 652-655)
+-------------------------------------------------------------+
         |
         v
+-------------------------------------------------------------+
| 2. PP 非首卡异步接收激活值: irecv_tensor_dict()              | (Line 659-672)
|    - 封装为 AsyncIntermediateTensors                        |
+-------------------------------------------------------------+
         |
         v
+-------------------------------------------------------------+
| 3. 调用下层 ModelRunner: model_runner.execute_model(...)    | (Line 677)
+-------------------------------------------------------------+
         |
         v
+-------------------------------------------------------------+
| 4. PP 非末卡异步发送激活值: isend_tensor_dict(...)           | (Line 681-691)
|    - 记录 handle 到 self._pp_send_work                      |
+-------------------------------------------------------------+
         |
         v
[返回 ModelRunnerOutput 或 None]
```

### 4.2 `NPUModelRunner.execute_model()` 逐阶段透视（`vllm_ascend/worker/model_runner_v1.py` Line 2038-2476）

#### 阶段一：批处理状态更新与内存同步（Line 2098-2134）
- 获取本轮调度的 Token 总数 `total_num_scheduled_tokens`。
- 在 `self.synchronize_input_prep()` 上下文内调用 `self._update_states(scheduler_output)`（Line 2121），将 CPU 维护的请求状态、KV 块映射、序列长度写入预分配的设备固定内存（Pinned / Device Tensor）。

#### 阶段二：输入打包装配（Line 2167-2176）
- 调用 `self._prepare_inputs(scheduler_output, num_scheduled_tokens_np)`：提取待计算的 Token ID、位置编码、计算 `logits_indices`（哪些位置需要计算 Logits 输出），并构建 Speculative Decoding 元数据。

#### 阶段三：静态图模式与 Token Padding 判定（Line 2189-2204）
- 调用 `self._determine_batch_execution_and_padding(...)`：
  - 根据输入 Token 总数及最大请求长度，从预捕获的静态图 Shape 桶（Capture Sizes）中查找最匹配档位。
  - 若命中，返回 `cudagraph_mode = CUDAGraphMode.FULL` 或 `PIECEWISE`，以及规整后的 `batch_desc.num_tokens`（例如实际有 13 个 Token，向上对齐 Pad 到 16）。

#### 阶段四：Attention Metadata 构建与 FIA 对齐（Line 2285-2315）
- 若处于静态图模式或使能序列并行（SP），调用 `self._pad_query_start_loc_for_fia(...)`（Line 2293）：
  - 针对昇腾专用的 `FusedInferAttentionScore` (FIA) 算子，重写 `query_start_loc`，将末端填充无效请求，使其严格对齐硬件算子的 Shape 要求。
- 调用 `self._build_attention_metadata(...)`（Line 2302）：组装 `AscendMetadata`，包括 `slot_mapping`（物理 Block 寻址偏移）、`block_tables`、`actual_seq_lengths_q`、`actual_seq_lengths_kv`。

#### 阶段五：前向上下文注入与模型执行（Line 2358-2407）
- 进入 `set_ascend_forward_context(...)` 上下文（Line 2360），通过 Python 全局上下文将 `attn_metadata`、`num_tokens_padded`、`batch_descriptor` 传递给底层各层神经网络。
- 执行 `self._model_forward(...)`（Line 2401）：
  - 如果被 `ACLGraphWrapper` 包裹，则通过当前 `batch_descriptor` 查找捕获好的图实例并执行 `replay()`；若为 Eager 模式，则执行 PyTorch 模型前向。

#### 阶段六：后处理与 Logits 计算（Line 2407-2468）
- 若为流水线并行非末卡，打包 `IntermediateTensors` 返回（Line 2415-2421）。
- 若为末卡，通过 `sample_hidden_states = hidden_states[logits_indices]` 提取目标 Token，并调用 `self.model.compute_logits(...)`（Line 2432）。
- 将结果暂存入 `self.execute_model_state`，等待随后的 `sample_tokens()` 被调用执行采样（Line 2455-2468）。

---

## 5. 三大关键技术深度解析

### 5.1 静态图机制：ACLGraph（类似 CUDAGraph）

在昇腾平台上，Host（CPU）下发 Task 到 Device（NPU）存在一定的驱动与硬件指令发射开销。在 Decode 阶段，每步仅计算 1 个 Token，算子执行耗时往往在几十微秒级别，Host 端调度（Python 解释器开销、ATen 派发、CANN Runtime Launch）会成为主要瓶颈。

#### 1. 架构封装：`ACLGraphWrapper`
源码位于 `vllm_ascend/compilation/acl_graph.py`（Line 60-268）：
- `ACLGraphWrapper` 包裹具体的模型前向函数（`runnable`）。
- 内部维护 `self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry]`（Line 114），每个不同的 Batch 规整尺寸对应一个独立的图实例。

#### 2. Capture 阶段流程（Line 153-241）
- 调用 `aclgraph = torch.npu.NPUGraph()`。
- 使用 `with torch.npu.graph(aclgraph, pool=self.graph_pool):` 上下文。
- **私有显存池（Graph Pool）**：通过 `self.graph_pool = current_platform.get_global_graph_pool()`，所有捕获的图共享同一片设备显存池，确保图与图之间不产生显存膨胀。
- **禁用 GC 优化**：在分段捕获中，通过 `patch("gc.collect", lambda: None)`（Line 175）避免跨层频繁触发垃圾回收，显著加速预热过程。
- **弱引用保护**：捕获完毕后，对输出及 Workspace 采用 `weak_ref_tensors()`（Line 202, 227），解除 Python 端的强引用占用，其底层物理内存完全托付给 NPUGraph 管理。

#### 3. 分段（Piecewise）与多档 Padding
- **为什么需要分段（Piecewise）**：全图捕获（Full Graph）需要模型内没有任何动态控制流（Data-dependent shape）。但部分复杂模型（如 MoE 的 Token Dispatch、部分含动态掩码的算子）无法做全图捕获。`vllm-ascend` 支持 `CUDAGraphMode.PIECEWISE`，将模型拆分成多个 Layer 级的子图，中间穿插 Eager 调度。
- **为什么需要 Shape Padding**：静态图要求 Tensor 的物理维度与内存地址绝对固定。调度器将离散的 Token 数量规整（Bucket Padding）到固定的捕获点（如 1, 2, 4, 8, 16...），输入 Tensor 地址复用静态 Buffer。

#### 4. Replay 阶段（Line 252-268）
- 在重放前，如果处于异步调度，调用 `torch.npu.current_stream().synchronize()` 确保上一轮图重放执行完毕。
- 执行 `entry.aclgraph.replay()`：**完全绕过 Python 与 PyTorch ATen 调度层**，直接在底层 CANN 硬件队列中重新激发已录制的硬件任务流，执行耗时接近物理极限。

---

### 5.2 Attention 算子与 KV Cache 管理机制

#### 1. 硬件特性与算子对齐
昇腾 AI Core 采用达芬奇架构，内部区分 **Cube 矩阵乘单元**（擅长 16x16 矩阵微块乘法）与 **Vector 矢量单元**（擅长 Softmax、RMSNorm、转置等元素级计算）。
标准的 Attention 计算包含 $Q \times K^T \rightarrow \text{Softmax} \rightarrow \times V$。在 Eager 模式下分立执行会导致频繁的 HBM 往返搬运。

在 `vllm_ascend/attention/attention_v1.py` 中：
- **Prefill 阶段**：直接调用底层的融合算子 `torch_npu.npu_fusion_attention`（Line 1577）：
  ```python
  return torch_npu.npu_fusion_attention(
      query=query, key=key, value=value,
      head_num=self.num_heads, input_layout="TND",
      scale=self.scale, actual_seq_qlen=actual_seq_qlen, ...
  )[0]
  ```
  采用 `TND`（Total tokens, Num heads, Head Dim）紧凑内存布局，避免 Batch 维度的无效内存空洞。
- **Decode 阶段**：调用 `DeviceOperator.npu_fused_infer_attention_score`（Line 1486），即昇腾专用的 **FIA（Fused Infer Attention）** 算子。该算子直接将 Paged KV Cache 的内存地址表、Block Tables、Slot Mapping 传入底层 Kernel，在 AI Core 内部流式完成 Cache 读取与 Attention 融合计算。

#### 2. KV Cache 物理存储布局与 NZ 格式
- 在传统 GPU 上，KV Cache 普遍按 `[num_blocks, block_size, num_heads, head_dim]` 的连续行优先格式（ND 格式）排布。
- 在昇腾硬件上，Cube 单元计算要求数据按照 **5 维分形格式（Fractal NZ / FZ）** 对齐（16 字节对齐）。若保持 ND 格式，Cube 计算前必须动态做转置与格式重排，严重浪费片上带宽。
- 在 `vllm_ascend/attention/attention_v1.py:do_kv_cache_update()`（Line 1588-1608）中：
  ```python
  DeviceOperator.reshape_and_cache(
      key=key, value=value,
      key_cache=self.key_cache, value_cache=self.value_cache,
      slot_mapping=slot_mapping,
  )
  ```
  该底层算子不仅完成将当前步的 K/V 写入分页内存槽位，还在写回 HBM 的过程中**就地完成了向 NZ 格式的物理重排**，后续 FIA 算子读取时即为原生高效对齐格式。

#### 3. DeepSeek MLA / DSA 支持
- 在 `vllm_ascend/attention/mla_v1.py` 和 `dsa_v1.py` 中，针对 DeepSeek 的低秩吸收特性，`vllm-ascend` 实现了把 $W_{UK}$ 权重吸收到 Key 投影中，使 KV Cache 只需存储极低维度的压缩潜变量（Compressed Latent Vector $c_t^{KV}$ 与解耦 RoPE $k_t^R$），显存开销降低 80% 以上。

---

### 5.3 算子库适配与 Patch 机制

为了在零修改 HuggingFace / 上游 vLLM 官方模型定义的前提下榨干 NPU 性能，`vllm-ascend` 广泛运用了 **OOT PluggableLayer 注册** 与 **Monkey Patch（猴子补丁）**。

#### 1. 核心算子深度定制（`vllm_ascend/ops/`）
- **RMSNorm**（`vllm_ascend/ops/layernorm.py:AscendRMSNorm` Line 30）：继承自上游 `RMSNorm`，底层使用 `torch_npu.npu_rms_norm` 替代 CUDA Kernel，实现单指令周期完成方差统计与归一化缩放。
- **Linear 算子族**（`vllm_ascend/ops/linear.py`）：实现 `AscendQKVParallelLinear`、`AscendRowParallelLinear` 等，在权重矩阵乘（MatMul）中根据硬件配置注入转置优化与 AllReduce 融合标志。
- **SwiGLU 激活**（`vllm_ascend/ops/activation.py:AscendSiluAndMul` Line 35）：调用 `torch_npu.npu_silu` 融合 Kernel。

#### 2. Monkey Patch 拦截机制（`vllm_ascend/patch/`）
在进程启动初期（`vllm_ascend/__init__.py:_ensure_global_patch()` Line 51-65），调用 `adapt_patch(is_global_patch=True)`，对上游关键类进行重构：
- **模型类 Forward 替换**：以 DeepSeek 为例（`vllm_ascend/patch/worker/patch_deepseek_v2.py:293`），将上游 `DeepseekV2Model.forward` 替换为 `_patched_forward`，使得模型在执行 Transformer 层时，直接调用 Ascend 版本的 MLA 与 Fused MoE 模块。
- **分布式 ProcessGroup 拦截**：在 `vllm_ascend/patch/worker/patch_distributed.py`（Line 46-75），劫持了 PyTorch 的 `init_process_group`，将通信后端强制替换为昇腾专属的 `hccl` 后端，并设置 ProcessGroup 资源池。

---

### 5.4 分布式并行与卡间通信（HCCL）

在超大规模集群推理中，张量并行（TP）、流水线并行（PP）与专家并行（EP）的卡间通信是决定端到端吞吐的生命线。

#### 1. MoE 通信架构与 MC2 融合
在开源实现中，MoE 的路由与分发通常是：`Router -> TopK -> CPU/GPU 排序 -> AllToAll 通信 -> Expert GEMM -> AllToAll 反向通信`。这一过程存在两次跨卡传输，通信开销极大。

`vllm-ascend` 在 `vllm_ascend/ops/fused_moe/token_dispatcher.py` 中实现了 `TokenDispatcherWithMC2`（Line 109）：
- **获取底层 HCCL 通信句柄**（Line 115-116）：
  ```python
  backend = device_group._get_backend(torch.device("npu"))
  self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
  ```
  直接绕过高级封装，抓取华为底层 `libhccl.so` 维系的物理通信器名称。
- **硬件级 Dispatch 融合**（Line 232）：
  ```python
  output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs_mc2)
  ```
  该算子在昇腾底层将 **Token 排序重排（Permute）、量化（FP8/MXFP）、卡间 AllToAll 传输** 三个阶段融合成单个 NPU 硬件任务，数据直接在各卡的私有通信缓冲区之间流转。
- **硬件级 Combine 融合**（Line 326）：
  ```python
  combined_output = torch_npu.npu_moe_distribute_combine_v2(**kwargs_mc2)
  ```
  同样将逆向 AllToAll 通信、加权求和（TopK Weight Combine）、逆重排（Unpermute）融合执行。

#### 2. EPLB（专家动态负载均衡器）
在大模型 MoE 推理中，长尾分布会导致某些“明星专家”被过频激活，而其他专家处于闲置状态，造成严重计算气泡。
`vllm-ascend` 在 `vllm_ascend/eplb/` 下构建了成体系的动态平衡子系统：
- **热度监控**：在 `EplbUpdator.forward_end()`（Line 129）中，统计每步推理各专家的被选中频率（Load Info）。
- **后台平衡规划**：在独立的后台进程 `EplbWorker.worker_process()`（Line 365）中，利用 `FlashLB`（Line 521）或 `SwiftBalancer`（Line 30）算法，基于滑动窗口与极值方差规划最优冗余副本。
- **卡间权重动态置换（D2D Transfer）**：`EplbDeviceTransferLoader.asyn_expert_weight_transfer()`（Line 85）在推理间隙通过 NPU 间直接通信（P2P/HCCL），将热点专家权重复制到空闲 NPU 的冗余槽位中，动态更新 `logical_to_physical_map`，使得后续推理的 Dispatch 算子能够将流量均匀分流。

---

## 6. 项目边界与跨项目调用清单

`vllm-ascend` 严格扮演承上启下的“胶水与驱动层”。其所有能力最终必须跨出自身代码边界，调用下游支持库。

```text
                  +-----------------------------------+
                  |        huawei/vllm-ascend         |
                  +-----------------+-----------------+
                                    |
         +--------------------------+--------------------------+
         |                                                     |
         v                                                     v
+-----------------------------------+         +-----------------------------------+
|             torch_npu             |         |         PyTorch Distributed       |
| - torch_npu.npu_fusion_attention  |         | - torch.distributed.all_reduce    |
| - torch_npu.npu_fused_infer_...   |         | - torch.distributed.all_to_all_...|
| - torch_npu.npu_rms_norm          |         | - backend: "hccl"                 |
| - torch_npu.npu_moe_distribute_...|         +-----------------+-----------------+
| - torch.npu.NPUGraph              |                           |
| - torch.npu.graph / replay        |                           |
+-----------------+-----------------+                           |
                  |                                             |
                  +---------------------+-----------------------+
                                        |
                                        v
                      +-----------------------------------+
                      |      Huawei CANN Open/Closed      |
                      | - libascendcl.so (ACL Runtime)    |
                      | - libhccl.so (HCCL Communication) |
                      | - libacl_op_compiler.so (ACLNN)   |
                      | - Ascend Driver (HAL / DevIO)     |
                      +-----------------------------------+
```

### 6.1 核心下游调用 API 清单

| 被调项目/库 | 调用形式 / 具体 API | 在 vllm-ascend 中的调用点与功能 |
| :--- | :--- | :--- |
| **`torch_npu`** | `torch_npu.npu_fusion_attention` | `attention_v1.py:1577`：Prefill 阶段全量注意力融合算子 |
| **`torch_npu`** | `DeviceOperator.npu_fused_infer_attention_score` | `attention_v1.py:1486`：Decode 阶段 Paged FIA 注意力融合算子 |
| **`torch_npu`** | `DeviceOperator.reshape_and_cache` | `attention_v1.py:1602`：KV Cache 槽位搬运与 Fractal-NZ 格式重排 |
| **`torch_npu`** | `torch_npu.npu_moe_distribute_dispatch_v2` | `token_dispatcher.py:232`：MoE 专家分发、重排、卡间通信一体化融合 |
| **`torch_npu`** | `torch_npu.npu_moe_distribute_combine_v2` | `token_dispatcher.py:326`：MoE 专家输出聚合与逆重排一体化融合 |
| **`torch_npu`** | `torch.npu.NPUGraph` / `torch.npu.graph` | `acl_graph.py:165, 187`：静态图录制与物理显存池绑定 |
| **`torch.distributed`**| `dist.all_reduce`, `dist.all_gather`, `dist.irecv` | `worker.py:664`、`patch_distributed.py:57`：跨卡跨节点流水线/张量通信 |
| **`HCCL`** | `backend.get_hccl_comm_name()` | `token_dispatcher.py:116`：提取底层 HCCL 通信域句柄直接传给 NPU 算子 |
| **`CANN Runtime`** | `expandable_segments:True` 环境变量 | `platform.py:1340`：底层虚拟内存分页分配器，避免碎片化 OOM |
