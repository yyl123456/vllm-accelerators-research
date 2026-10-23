# TorchTPU（torch_tpu）开源后端架构与 vLLM 接入分析

> 调研日期：2026-10-23
> 源码基线：`google/third_party/torch_tpu` @ `d7a3c1cc73a115376a058c7664986f5e34ca8362`（2026-10-22，shallow clone）
> 官方发布形态：PyPI `torch_tpu==0.2.0`，文档站 <https://torchtpu.readthedocs.io/>
> 对应权威编号：`20260903-131` ~ `20260903-135`（13-tpu 章节）
> 证据等级：`SOURCE_IMPLEMENTED × STATIC_REVIEWED`（本工作区无 TPU 实机，无 `DEVICE_SMOKE`）

---

## 0. 结论速览

| 维度 | 事实 | 证据 |
|---|---|---|
| 项目性质 | Google 官方开源的 **out-of-tree PyTorch 后端**，注册为 `PrivateUse1`，运行时重命名为 `tpu` | `csrc/common/device_type.h:27-45`；`src/torch_tpu/_loader.py:256` |
| 与 PyTorch/XLA 关系 | 官方 FAQ 明确：**PyTorch/XLA 已废弃（deprecated），由 TorchTPU 取代**，且"zero JAX runtime dependencies" | `docs/faq.md:7-27` |
| 接入形态 | 走 PyTorch 设备自动加载（autoload）入口点 `torch.backends`，用户代码几乎零改动 | `build_files/entry_points.bzl:17-19`；`src/torch_tpu/_loader.py:387` |
| 执行范式 | ATen 算子先落成 **DeferredOp（延迟算子）**，由后台线程成图、编译、下发 PjRt；**不是**解释执行，也**不是** CUDA Graph 式 replay | `csrc/eager/device_buffer.h:415`；`csrc/eager/materialize.cc:291-360` |
| 编译栈 | StableHLO → OpenXLA（`-O1` FastCompile / `-O2` FastRuntime）→ PjRt 可执行文件，**不使用 TorchInductor** | `csrc/common/compilation_spec.h:33-37`；`docs/how_torchtpu_works.md` |
| 动态 shape | **Bounded Dynamism**：按 128 对齐分桶（默认 1 桶，最多 8 桶），桶内复用同一 executable | `src/torch_tpu/_internal/compile/dynamic/dynamic_compiler.py:123-151` |
| vLLM 接入 | 由**独立仓库** `vllm-project/vllm-torchtpu` 提供 Platform 插件；本仓库只提供 backend + compiler | `.github/workflows/postsubmit.yml:231-237`；`docker/README.md:1-23` |
| 代码体量 | C++/头文件 ≈ 21.6 万行，Python ≈ 4.3 万行，155 个算子目录，200 个测试文件，27 篇官方文档 | 本地统计（`find/wc`） |

---

## 1. 项目定位：三条 Torch-on-TPU 路线的分叉

官方 FAQ 给出三个明确边界（`docs/faq.md:5-27`）：

1. **TorchTPU vs PyTorch/XLA**：PyTorch/XLA 历史实现"重度依赖 JAX runtime 组件"；TorchTPU 是原生 PyTorch 后端，**无 JAX runtime 依赖**。
2. **TorchTPU vs TorchAX（torch_xla2）**：TorchAX 在 **Python 层**拦截算子并降到 XLA（常经 JAX）；TorchTPU 在 **ATen / C++ dispatcher 层**接入。
3. **PyTorch/XLA 的去向**：已废弃并被 TorchTPU 取代（`docs/faq.md:21-23`）。

> ⚡ **【上下文冲突校准】**：本工作区 `google/docs/aot/20-vllm-torchtpu-xla.md`（2026-09-08）当时断言"TorchTPU 官方仓库尚未公开、Google 私有源码"，并列出 5 项"无法从源码确认"。该断言已被本次开源证伪，相关空缺在本文第 2~6 节逐条闭合。

与 vLLM 的关系需要区分两个仓库：

- `vllm-project/tpu-inference`（本地固定 `fd338000`）：上游 README 已声明改名为 **vLLM JAX（vllm-jax）**，主线是 JAX 模型 + TorchAX 桥接；
- `vllm-project/vllm-torchtpu`：承接 **PyTorch 原生 serving** 路径，torch_tpu 是其编译/执行底座。

---

## 2. 端到端链路（源码锚点）

### 2.1 进程内注册

- 入口点：`build_files/entry_points.bzl:17-19` 声明 `torch.backends` 组下的 `torch_tpu = torch_tpu._loader:load`，由 PyTorch 的 backend autoload 机制在 `import torch` 时拉起。
- `_loader.py:387` 的 `load()` 依次完成：设备模块注册 → `rename_privateuse1_backend("tpu")`（`:256`）→ `generate_methods_for_privateuse1_backend()`（生成 `Tensor.is_tpu` / `.tpu()` 等）→ 分布式后端 `tpu_dist` 注册。
- 物理设备与逻辑设备解耦：`csrc/common/device_type.h:27-45` 注释明确"PyTorch 永远看到 PrivateUse1，TorchTPU 可用于任何带 PjRt 插件的 XLA 设备类型"，当前仅枚举 `kTPU`。

### 2.2 ATen 派发 → DeferredOp

- 每个算子内核用 `TT_KERNEL(op_name, param_keys, args, ...)` 宏包裹（`csrc/ops/macros/kernel.h:77`），自动完成 Python 上下文捕获、**参数 cache key** 计算与错误归因。
- `DispatchOp<kArity, kNumOutputs>`（`csrc/eager/op_dispatcher.h:366-430`）把 `op_builder`（StableHLO 构建闭包）、输入 `DeviceBufferRef`、输出 `Shape`、参数 cache key 打包成 `DeferredOp`（`csrc/eager/device_buffer.h:415`）。
- 关键设计：返回给 PyTorch 的 `at::Tensor`，其 `StorageImpl` 的 `data_ptr` **指向 CPU 侧的元数据**，而非 TPU 显存；这层间接是"CUDA 式 API ↔ XLA 函数式语义"的转换点（`docs/torchtpu_memory_model.md`）。
- 算子覆盖：`csrc/ops/` 下 155 个算子目录（binary/conv/attention/norm/quant 等），官方宣称 450 个核心 ATen 算子（前向+反向）（`docs/intro_to_torchtpu.md`）。

### 2.3 Stream、物化与执行线程

- 每个 DeferredOp 入当前 `torch.Stream` 队列；**进程内单个 materialize 线程 + 单个 execute 线程**（`csrc/eager/materialize.cc:291-293`）。
- `MaterializeLoop`（`:319`）：出队 → `ProcessMaterializationTask`（`:124`）做图分析、切分、查/编译 → 产出 `ExecutionTask` 入执行队列。
- `ExecuteLoop`（`:337`）：串行 `task.Run()`，失败只记日志，不中断线程。
- 线程被显式禁止访问 thread-local 上下文状态（`DisallowThisThreadToAccessContextState`，`materialize.cc:319-326`），所有配置在派发线程解析后随任务下传。

### 2.4 图切分与编译档位

- `Traversal`（`csrc/eager/traversal.h:108`）表示一段可编译的 DeferredOp 图；`SplitTraversal`（`csrc/eager/split_traversal.h:33-50`）按"物化启发式 + `OpSplitMode`"切成单输出子图，并保证拓扑序。
- 编译档位仅两档（`csrc/common/compilation_spec.h:33-37`）：`kFastCompile`（`-O1`，编译快、执行慢）与 `kFastRuntime`（`-O2`，编译慢、执行快）。
- 档位选择并非全局固定：例如卷积有专门启发式，按 batch/宽度/MACs 在 O1 与 O2 间切换（`csrc/eager/materialization_heuristics.h` 注释给出 35x~106x 的实测区间）。

### 2.5 多级编译缓存

- `CompilationCache::GetOrCompile`（`csrc/common/compilation_cache.h:284`）统一入口：命中则直接取 executable future，未命中则入编译线程池异步编译。
- 三级缓存（`docs/how_torchtpu_works.md` + `csrc/common/env_vars.h:203,207`）：
  - Tier-1 进程内内存缓存，始终开启；
  - Tier-2 主机共享内存（`TORCH_TPU_TIER2_COMPILATION_CACHE`），分布式程序默认开启；
  - Tier-3 远端文件系统（`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`，如 GCS），默认关闭。
- Cache key 由"图结构 + 输入/输出张量形状 + 编译选项 + 硬件/libtpu/PjRt 版本指纹"构成（`docs/how_torchtpu_works.md` 末段）。

### 2.6 PjRt 下发

- `Execute`（`csrc/pjrt/pjrt_execute.cc:245`）→ `PjRtLoadedExecutable::Execute`（`:303`），并以 `strict_shape_checking = true` 校验（`:250`）。
- 执行是异步的：返回 `PjRtBuffer` future，CPU 侧通过 `Stream.synchronize` / `Event` / D2H 拷贝等待。

---

## 3. 四种执行模式

| 模式 | 开关 | 构图粒度 | 编译档位 | 生产适用性 |
|---|---|---|---|---|
| **Strict eager（默认）** | 默认 | 单算子 | O1 | 兼容性最好；每 op 一次编译，靠 Tier-1 缓存复用 |
| **Debug eager（launch blocking）** | `TORCH_TPU_LAUNCH_BLOCKING=1`（`env_vars.h:184`） | 单算子 | O1 | 仅调试，每个 op 主机阻塞同步 |
| **Optimized eager（Defer & Fuse）** | `TORCH_TPU_DEFER_AND_FUSE=1`（`env_vars.h:99-100`） | 跨 Python 调用的算子段 | O2 | 官方称较默认 eager 提升最多 50%；依赖"重复派发检测"触发物化 |
| **`torch.compile(backend="tpu")`** | `registry.register_backend(name="tpu")`（`src/torch_tpu/_internal/compile/__init__.py:31`） | 整个 FX graph | O2 | 性能上限最高；一次 trace + 一次编译，运行期零图分析 |

要点：

1. 四种模式共用同一套 DeferredOp / 缓存 / PjRt 执行基座，差异只在"何时物化"与"编译档位"。
2. `EagerMode` 通过 Python 上下文管理器按线程生效（`src/torch_tpu/_internal/execution_mode/__init__.py:30-37`）。
3. `torch.compile` 路径不经过 Inductor，而是把 FX graph 整体降到 StableHLO 后交 XLA（`docs/how_torchtpu_works.md` 的 "torch.compile support" 节）。

---

## 4. 动态 shape：Bounded Dynamism

- 机制：Dynamo 把变化维度记为 `SymInt`，TorchTPU 求其上界区间，按 **128 对齐**切桶（默认 1 桶，最多 8 桶），每桶编译一个 executable；运行期按实际尺寸选桶，XLA 把输入 pad 到桶上界并用代数单位元填充（`docs/dynamic_shapes.md`）。
- 源码：`dynamic_compiler.py:123-151`（`bucket_size=128`、`num_buckets` 上限 8）、`:268-273`（在 `bucket_size` 的整数倍处生成桶边界）。
- 动机用例直指 LLM 解码：KV cache 每步 +1，静态编译会导致"每 token 一次全量 XLA 重编译"（`docs/dynamic_shapes.md` "Motivating use case"）。

> ⚠️ **【文档与实现不一致】**：`_backend.py:15-23` 的模块 docstring 仍写"dynamic shapes are not supported"，与 `dynamic_shapes.md` 及 `dynamic_compiler.py` 的实现相矛盾。以代码与用户文档为准，判定该 docstring 陈旧。

---

## 5. vLLM 推理接入视角

### 5.1 插件契约

- `vllm-torchtpu` 作为 vLLM Platform 插件，需要 `dispatch_key = "PrivateUse1"`（因为设备已被重命名）、`simple_compile_backend = "tpu"`，并通过 `torch.compile(backend="tpu")` 复用本仓库编译器。
- 该契约在 2026-09 的 `20-vllm-torchtpu-xla.md` 中只能靠外部仓库符号推断；现在 `_backend.py` / `compile/__init__.py:31` 已给出实现侧确认。

### 5.2 与连续批处理的结构性摩擦

| 摩擦点 | 机制 | 影响 |
|---|---|---|
| D2H 同步 | 采样/控制流会 `synchronize` 并 drain 整条 stream（`docs/torchtpu_execution_model.md` "Pitfalls"） | 每 step 的 logits→CPU 采样会把流水线打断 |
| 单物化线程 | `MaterializeLoop` 全局唯一（`materialize.cc:291`） | 多请求并发时主机侧串行，成为吞吐上限 |
| Defer&Fuse 触发条件 | 依赖"检测到重复派发"才物化（`env_vars.h` 的 `...DETECT_REPEATED_OPS`） | batch 组成抖动时退化为逐 op 编译 |
| 形状漂移 | 每超出桶上界一次即新增编译（最多 8 桶） | 长尾长度分布会触发额外编译与内存占用 |

### 5.3 KV cache 与 Attention

- 官方给出两条对照路线（`docs/case_studies_muse_glimmer_30b_inference.md`）：预分配 **StaticCache + `dynamic=False`**（静态编译、需原地写）vs 增长式 **DynamicCache + `dynamic=None`**（桶内复用）。
- Paged Attention 不在核心算子表内，而是通过 **Pallas 自定义内核**提供（`docs/custom_kernels.md:36` 列出 Paged attention / Gated DeltaNet / MLA 等变体）。
- 官方 feature_set 将 "Inference (multi-host, P/D disagg)" 标为 production-ready、vLLM/SGLang 标为 validated（`docs/feature_set.md`）。

---

## 6. 与既有知识库结论的校准

| 既有结论（来源） | 现状 | 判定 |
|---|---|---|
| "TorchTPU 仓库未公开，内部实现无法确认"（`google/docs/aot/20-vllm-torchtpu-xla.md`） | 已开源，SHA `d7a3c1cc` | **实质冲突，前者过期** |
| "TPU 不支持逐算子即时派发，纯 Eager 仅 Debug"（`13-tpu/20260903-131`） | TorchTPU 提供 strict eager：逐 op 构图 + O1 即时编译 + 缓存，异步执行 | **需细化**：不能"无图"执行，但可以"逐 op 建图并编译"；并非解释执行 |
| "PyTorch 前端经 TorchAX → JAX 编译"（`13-tpu/20260903-131/132`） | 对 `tpu-inference@fd338000` 仍成立；但该仓库已改名 vllm-jax，PyTorch 原生路径迁移到 vllm-torchtpu + torch_tpu | **语境错位**：路线已分叉，非事实错误 |
| "Shape Bucketing 由 tpu-inference 的 `compilation_manager` 实现" | TorchTPU 把分桶下沉到编译器（bounded dynamism），框架层不再需要手工预编译桶 | **架构演进**，两层可能并存 |

---

## 7. 开闭源边界

| 层 | 状态 | 说明 |
|---|---|---|
| Python/C++ 后端、ATen 内核、DeferredOp、缓存、PjRt 调用、`tpu_dist` | **开源**（Apache-2.0，`LICENSE`） | 本文全部锚点所在 |
| XLA 编译器 TPU 后端（HLO→TPU 机器码） | 开源仓库 `openxla/xla`，但 TPU codegen 细节不透明 | 本工作区 `google/third_party/xla` |
| `libtpu`（runtime/驱动交互） | **闭源二进制**（PyPI 分发） | 本仓库仅 `libtpu.configure_library_path()`（`src/torch_tpu/__init__.py:27-33`） |
| TPU ISA / 固件 / 驱动 | **闭源** | 官方文档明确 ISA 不公开（`docs/how_torchtpu_works.md`） |
| `vllm-torchtpu` 插件本体 | 独立仓库，本工作区**未固定 SHA** | 接入细节属 `UNKNOWN`，需后续单独固定 |

---

## 8. 工程局限与风险（客观）

1. **主机侧串行**：单 materialize + 单 execute 线程，是设计上的简化，也是 serving 吞吐的硬约束。
2. **编译抖动**：桶外 shape 必然重编译；8 桶上限意味着极长上下文或极端 batch 仍会溢出。
3. **一致性与死锁**：SPMD 需各 rank 编译同一 HLO，仓库提供 `TORCH_TPU_SPMD_HANDSHAKE_STAGE`（compile/dispatch 两阶段，`csrc/distributed/handshake.h`）做对齐；动态控制流不一致仍可能挂死。
4. **调试可见性**：异步执行使错误延迟暴露，需 launch blocking 或 `xprof`（`docs/performance_profiling_with_xprof.md`）。
5. **无实机验证**：本工作区未在 TPU VM 上运行，性能与数值结论均为官方声明（`VENDOR_CLAIM`），非实测。

---

## 9. 下一步

- [ ] 固定 `vllm-project/vllm-torchtpu` 的 SHA，核查 Platform/Runner/KV 链路（对应 `20260903-132/133`）。
- [ ] 在 `tools/claims_index.jsonl` 增补 TorchTPU 相关 claim（需先确定 owner 文档归属）。
- [ ] 若有 TPU 环境，补 `DEVICE_SMOKE`：strict eager vs defer&fuse vs torch.compile 的编译次数/时延对比。
