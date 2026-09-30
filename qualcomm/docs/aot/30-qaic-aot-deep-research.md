# 高通 Cloud AI (QAIC) AOT 静态编译路线深度调研与机制解析

> 调研日期：2026-09-29  
> 源码基线：
> - `qualcomm/third_party/vllm-qaic` (`3212cc670130b7e5290b429f781dc978d7ecf430`)
> - `qualcomm/third_party/efficient-transformers` (`e127a6b741d8b71e81134cc76c7e24cea7210cdc`)
> - `qualcomm/third_party/cloud-ai-sdk` (Models & Specialization Templates)  
> 权威参考目录：[`20260903-vllm-ai-infra-research/11-qaic/`](../../20260903-vllm-ai-infra-research/11-qaic/)

---

## 1. 核心回答与概念释疑

### 1.1 “vLLM Block Table 在此退化为一张请求槽位分配表”究竟是什么意思？

在标准 vLLM（如针对 NVIDIA GPU、Ascend NPU）中，PagedAttention 的精髓在于**动态细粒度分页内存管理**：
- 显存被划分为大小为 16 或 32 个 token 的逻辑块（Block）；
- 随着请求生成 token，BlockManager 动态按需分配新的物理 Block，并通过 `block_table`（二维映射数组：`[req_id, block_number]`）维护逻辑块到不连续物理显存块的映射；
- 算子底层通过 PagedAttention Kernel 逐块索引 KV Cache。

**而在高通 AOT 静态编译模式下，该机制发生了根本性退化**：
1. **`block_size` 强设为 `max_model_len`**：
   在 [`vllm_qaic/platform_base.py:313`](../../third_party/vllm-qaic/vllm_qaic/platform_base.py#L313)，系统直接执行：
   ```python
   cache_config.block_size = model_config.max_model_len
   ```
   这意味着每个“Block”的长度直接等于模型的最大上下文长度（例如 4096 或 8192）。一个请求从始至终只消耗且必须消耗**整整一个 Block**。
2. **强制 Block 数量等于并发上限加 1**：
   在 [`vllm_qaic/worker/worker.py:642`](../../third_party/vllm-qaic/vllm_qaic/worker/worker.py#L642)：
   ```python
   assert num_gpu_blocks == self.scheduler_config.max_num_seqs + 1
   ```
   系统总共分配的“Block”数量，严格等于设定的最大并发序列数加 1。
3. **`block_table` 直接提取为槽位索引**：
   在 [`vllm_qaic/worker/model_runner.py:814-816`](../../third_party/vllm-qaic/vllm_qaic/worker/model_runner.py#L814-L816)：
   ```python
   self.batch_indices = (
       self.input_batch.block_table[0].get_numpy_array()[:num_reqs, 0] - 1
   )
   ```
   此时，`block_table` 的第一列直接减 1，得到的就是当前请求独占的硬件物理上下文槽位 ID（`batch_index`，取值 `0 ~ max_num_seqs - 1`）。

**结论**：高通 AOT 路线只是借用了 vLLM 的 `BlockManager` 接口来做请求并发槽位（Slot）的占位计数器，其底层完全没有 PagedAttention 的动态页式扩展，而是退化为最原始的**静态连续槽位预分配机制**。

---

### 1.2 ONNX 模型图里面会有这个东西吗？

**答案是：有，而且必须以显式输入（Graph Input）和专用自定义算子的形式存在于 ONNX 计算图中！**

在原生 HuggingFace Transformer 导出的 ONNX 图中，输入只有 `input_ids`、`attention_mask`、`position_ids` 和 `past_key_values`。但在高通 AOT 管线中：

1. **ONNX Graph Inputs 中显式增加了 `batch_index`**：
   在 [`QEfficient/exporter/export_utils.py:72-88`](../../third_party/efficient-transformers/QEfficient/exporter/export_utils.py#L72-L88)：
   ```python
   dynamic_axis_past_key = "full_batch_size" if "batch_index" in input_names else "batch_size"
   ...
   elif iname == "batch_index":
       dynamic_axes[iname] = {0: "batch_size"}
   ```
   导出时，`batch_index` 被正式声明为 ONNX 图的顶级输入张量之一（维度为 `[batch_size]` 或 `[batch_size, 1]`，Dtype 为 INT32）。同时，`past_key_values` 的第 0 维动态轴名字从 `batch_size` 变更为 `full_batch_size`（即设备端常驻的最大并发槽位数）。

2. **图内注入了 Qualcomm 专属的 Scatter/Gather ONNX 算子**：
   在 [`QEfficient/customop/ctx_scatter_gather_cb.py:17-38`](../../third_party/efficient-transformers/QEfficient/customop/ctx_scatter_gather_cb.py#L17-L38)，QEfficient 定义了 domain 为 `com.qualcomm.cloud` 的 ONNX 自定义算子：
   - **`CtxScatterCB`**：
     入参为 `(data, batch_index, position_ids, updates)`。利用 `batch_index` 定位到第几个槽位，利用 `position_ids` 定位到槽位内的 sequence 偏移，将当前 Step 计算出的新 KV 写入到整段静态 KV Buffer 中：
     ```python
     indices = ops.Concat(batch_idx, head_idx, ctx_idx, axis=3)
     return ops.ScatterND(data, indices, updates)
     ```
   - **`CtxGatherCB` / `CtxGatherBlockedKVCB`**：
     入参为 `(data, batch_index, ctx_indices)`。根据当前请求分配的 `batch_index`，从常驻的全局静态 KV Buffer 中把历史 Key/Value 切片抽取出来，送入 Attention 矩阵乘法计算。

3. **编译器固化**：
   高通闭源编译器 `qaic-compile` 读取到带有 `batch_index` 与 `CtxScatterCB/GatherCB` 的 ONNX 图后，将其编译映射到硬件 NSP 计算核与片上 SRAM/DDR 固定的连续地址上。

---

## 2. 高通 AOT 路线软件栈端到端架构解析

### 2.1 整体架构与两路线判定
高通在同一个适配层 `vllm-qaic` 中支持 AOT 与 PyTorch Eager 两条路径，但代码中存在鲜明的排他性与默认偏好：
- 在 [`vllm_qaic/platform_base.py:64-70`](../../third_party/vllm-qaic/vllm_qaic/platform_base.py#L64-L70)：
  ```python
  _torch_qaic_installed: bool = importlib.util.find_spec("torch_qaic") is not None
  is_aot = not _torch_qaic_installed
  worker_cls_name = "QaicWorkerAoT" if is_aot else "QaicWorkerPyt"
  ```
  只要当前 Python 环境没有安装闭源的 `torch_qaic` 轮子包，系统便强制且默认走 `QaicWorkerAoT` 与 `QaicModelRunnerAoT`。

### 2.2 离线编译与产物容器 (QPC)
1. **模型导出与编译触发**：
   在 [`vllm_qaic/model_loader/qaic.py:1398-1424`](../../third_party/vllm-qaic/vllm_qaic/model_loader/qaic.py#L1398-L1424)，若启动服务时未提供现成的 QPC 路径，系统将调用 `QEfficient` 的 `get_hf_model` 构造改写后的模型，并触发 `compile()`。
2. **编译参数 Hash 与缓存复用**：
   在 [`QEfficient/base/modeling_qeff.py:1269-1286`](../../third_party/efficient-transformers/QEfficient/base/modeling_qeff.py#L1269-L1286)，编译器命令行入参、`specializations.json`（静态分桶档位）、`custom_io.yaml`（Host/Device 精度）、`mdp_ts_*.json`（多卡 Tensor-Slice 切分）计算哈希值，生成独立目录 `qpc_path-<compile_hash>/qpc`。如果目录下已存在 `programqpc.bin`，则秒级加载跳过编译。
3. **QPC 元数据契约**：
   在 [`vllm_qaic/model_loader/qaic_session_np.py:123-140`](../../third_party/vllm-qaic/vllm_qaic/model_loader/qaic_session_np.py#L123-L140)，`qaicrt.Qpc.getIoDescriptor()` 反序列化 `aicapi.IoDesc`：
   - 提取 `iodesc.allowed_shapes`：限定了系统合法执行的所有输入张量维数与 Dtype；
   - 提取 `iodesc.selected_set.bindings`：确定每个输入输出变量的内存索引。

### 2.3 运行时交互与异步完成契约
1. **纯 CPU/NumPy 数据流**：
   在 [`vllm_qaic/worker/model_runner.py:1055-1275`](../../third_party/vllm-qaic/vllm_qaic/worker/model_runner.py#L1055-L1275)，`QaicModelRunnerAoT` 接收到 vLLM 的 `SchedulerOutput` 后，将 token 与位置拆分为 Prefill 和 Decode：
   - 不足预编译分块的，按固定长度 padding；
   - 将组织好的连续 NumPy 内存传递给 `QAICInferenceSession`。
2. **入队与完成等待**：
   在 [`vllm_qaic/model_loader/qaic_session_np.py:546-583`](../../third_party/vllm-qaic/vllm_qaic/model_loader/qaic_session_np.py#L546-L583)：
   - `np_run()` 负责将连续内存绑定至预分配的 `qaicrt.ExecObj`，并推入 `queue.enqueue()`；
   - **严格生命周期约束**：源码明确强调 `np buffer` 的存活时间必须长于 `waitForCompletion()`，否则由于底层 DMA 异步传输会导致内存非法访问（SEGFAULT）；
   - 在 `complete_inf()`（阻塞调用 `waitForCompletion()`）返回后，硬件计算完成，此时才能安全提取 Logits 并归还 ExecObj。

---

## 3. 开闭源边界与工业落地局限

### 3.1 开闭源边界一览
- **开源部分 (开发人员可修改)**：
  - `vllm-qaic`：调度映射、Padding 填充、Session 管理、Logits 提取；
  - `QEfficient`：模型图改写、自定义 ONNX 算子生成（`CtxScatterCB/GatherCB`）、分桶模板与编译调用脚本；
  - `cloud-ai-sdk (models)`：各模型家族的 Specialization 配置与启动示例。
- **完全闭源黑盒 (开发人员无法干涉)**：
  - `qaic-compile`：图编译优化、常量折叠、算子融合、NSP 核心与片上 SRAM 内存物理分配；
  - `qaicrt`：C++/Python 运行时驱动交互、DMA 传输管理。

### 3.2 工业落地与高并发 Serving 局限
1. **KV 内存浪费极高**：因为 `block_size = max_model_len`，哪怕一个请求只生成 1 个 token，也要独占 4K/8K 的全部 KV 内存，导致单卡并发容量（`max_num_seqs`）非常受限。
2. **算力 Padding 浪费**：由于静态分桶与固定 Decode Batch 限制，未被请求填满的槽位必须填充 `-1` 并送入芯片完整执行，造成有效算力损失。
3. **完全无弹性能力**：无法在线应对超出编译 profile 范围的超长序列请求，任何序列长度和分桶调整都需重新编译 QPC（耗时数十分钟）。
4. **高级特性缺失**：在源码中，AOT 路线明确禁用了 Prefix Caching（公共前缀复用）与权重热更新（`reload_weights` 为空操作）。
