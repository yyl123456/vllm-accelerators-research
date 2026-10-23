# Modular MAX 源码开放增量盘点（2026-10-22 "source available" 发布）

> 调研日期：2026-10-23
> 源码基线：`modular/third_party/modular`（`da0db31d422e19ad5952d8d106b6da1d6899cf0a`，2026-10-23）
> 上一基线：`e700d92fe1ba720701d5b9d233c791590199bfb1`（2026-09-30，Mojo 1.2.0.dev / MAX 26.7.0.dev）
> 发布版本：Mojo `1.3.0.dev2026102305` / MAX `26.8.0.dev2026102305`
> 官方公告：*MAX is now source available and open to everyone*（2026-10-22，<https://www.modular.com/blog/max-license>）

---

## 1. 结论先行

上一基线（2026-09-30）时，仓库里 MAX 侧只有 Python 框架、管线（pipelines）、Serving 与 Mojo kernel 库；**图编译器、C++ 运行时、驱动/HAL、KV Cache 的 Rust 实现均不在公开树中**。本次更新（2026-10-22 合入）把这三层补齐：

| 新增目录 | 内容 | 规模（新增行 / 文件） | 许可证 |
|---|---|---|---|
| `GraphCompiler/` | MAX 图编译器（MAX IR → MEF 产物） | +221,900 / 1,426 | MAX License |
| `MLRT/` | Modular Low-level Runtime：Core / GraphRT / Driver | +76,577 / 277 | MAX License |
| `max/driver/` | Mojo 侧设备与 machine 层、HAL 胶水 | +17,097 / 86 | MAX License |
| `Target/` | HAL 官方硬件插件（CUDA / HIP / Metal） | +14,129 / 107 | MAX License |
| `SDK/` | HAL 的 ABI 安全 C API 头文件（`M_driver_*.h`） | +1,970 / 15 | MAX License |
| `rust_kv/` | KV Cache Rust crates（kv-core / kv-tier-connector / profiler） | +12,750 / 50 | Apache-2.0 + LLVM Exception |
| `Kernels/` | MAX License 覆盖的 kernel 侧胶水（WeightsRegistry / Tracing / MojoString C API） | +839 / 10 | MAX License |
| `max/lib/` | 随 MAX License 迁移的库目录 | +2,732 / 18 | MAX License |

按目录计合计约 **34.8 万行 / 1,989 个文件**；该合并提交整体为 **2,266 文件 / +379,743 行**（差额来自 `max/`、`Mojo/`、`bazel/` 内的配套调整）。这是 2026 年 8 月 Mojo 编译器开源之后，Modular 在 MAX 侧最大的一次源码开放。

**关键限定**：官方措辞是 **source available**，不是 OSI 认可的开源。新增的 MAX 主体代码走 `Licenses/MAX_LICENSE`（2026-10-21 修订），只授予**目标代码形式**的分发权；源码再分发仅允许"为向上游贡献而 fork"（§1.3）。Mojo 与部分 MAX 组件仍是 Apache-2.0 + LLVM Exception，逐文件头注释为准。

---

## 2. 更新操作与仓库增量

```bash
git -C modular/third_party/modular fetch --all --prune
git -C modular/third_party/modular pull --ff-only    # e700d92fe -> da0db31d4
```

- 新增提交：**2,016** 个 first-parent 提交（2026-09-30 → 2026-10-23），全量 diff：**8,049 文件，+1,006,476 / -296,997**。
- 开放事件本身是一个合并提交：
  `d0462d003e4d9a5e4309c5a76beec3b8c1eaa788`（2026-10-22，PR #7370，标题 `[MAX] Source-available component history`），单次 **2,266 文件 / +379,743 / -1,220**。提交标题中的 "history" 表明搬入的不只是快照，还包含这些组件的完整开发历史。
- 同一批 `[OSS]` 提交把发布方式从"预编译 wheel"改为"从本仓库源码构建"（`762423501`、`51101d602`），并新增 `bazel/public-patches/` 的公开补丁集，源码树与发行二进制的对应关系因此可验证。
- README 重写，逐组件标注开放时间：Mojo 编译器（2026-08）、Mojo stdlib（2024-03）、MAX GPU 抽象与 kernel 库（2025-05）、MAX Python 框架（2025-11）、**MAX graph compiler / runtime / driver（本次）**。

---

## 3. 新开放组件的实际职责（源码锚点）

### 3.1 GraphCompiler：MAX 图编译器

- 定位：框架无关的高层算子与图优化层，"MAX graphs → MEF artifacts for GraphRT"（`GraphCompiler/AGENTS.md`）。
- 目录结构：`lib/{MODialect,MAPDialect,MOGGDialect,RMODialect,MOSHDialect,MGPDialect,Conversion,FrameworkFrontend,MEFModel,Support,Driver}`，即从 MO（强类型算子）→ MAP → MOGG（生成 Mojo）的多级 lowering。
- 代表性 pass：`MODialect/Transforms/` 下 `ConstantFold.cpp`、`PatternFusion.cpp`、`InferLayouts.cpp`、`FlattenElementwise.cpp`、`MergeAdjacentParallels.cpp`、`NanCheckPass.cpp` 等。
- 设计文档同步开放：`GraphCompiler/docs/ArchitectureSpecification.md`（1,238 行，12 条原则 / 24 项质量属性 / 8 条约束 / 8 个 ADR）、`Passes.md`、`CachingInTheGraphCompiler.md`、`Runtime.md`。
- 明确的架构边界（同样来自该规范）：MEF 是与 GraphRT 的唯一边界，且"每个顶层图一个 MEF"；编译产物缓存命中直接跳过编译；设备数量必须由图中显式标注，编译器不做自动切分。

### 3.2 MLRT：MAX 低层运行时

`MLRT/docs/README.md` 给出三个子库：

- **Core**：IO（GPUDirect Storage，实验性）、Tensor/Buffer 生命周期、MojoValue（C++ ↔ Mojo 对象安全传递）。
- **GraphRT**：预编译 dataflow 图的执行层，原语以 C/C++ 函数按名注册，只依赖 AsyncRT 与 MLIR。
- **Driver**：跨厂商设备发现、内存管理、kernel 执行，并含 **NIXL** 多节点通信层；`MLRT/docs/Driver/README.md`。

其上是仓库原有的 `AsyncRT`（线程池/并行 CPU 计算），本次未变。

### 3.3 Driver + Target + SDK：HAL 三层

- `SDK/include/HAL/` 共 12 个头：`M_driver_api.h`、`M_driver_device.h`、`M_driver_memory.h`、`M_driver_queue.h`、`M_driver_event.h`、`M_driver_function.h`、`M_driver_peer.h`、`M_driver_tma.h`、`M_driver_bundle.h`、`M_driver_context.h`、`M_driver_common.h`。统一的 `M_DRIVER_STATUS_*` 状态协议，属于 ABI 安全的 C 接口。
- `Target/`：官方 HAL 插件（CUDA / HIP / Metal）。`Target/README.md` 规定插件**只允许依赖 `//SDK`**，引用计数、缓存等"跨硬件共性逻辑"必须留在 HAL 之上；并明确**合作方/保密插件不得进入 monorepo**，必须在受控内部仓库开发。
- `max/driver/`：Mojo 侧 `machine` / device 抽象（`max/driver/src/max_license/machine/*.mojo`），把设备能力暴露给 Mojo 运行时。

### 3.4 rust_kv：KV Cache 的 Rust 实现

`rust_kv/README.md` 自述全部 crate 为 **Apache-2.0 + LLVM Exception**，并在公开仓库从源码构建：

- `kv-core`：纯 Host 侧 KV 记账（无 GPU/异步/Python 依赖），含前缀命中规则、块哈希、Jenga 几何规划与块池；
- `kv-tier-connector`：分层 KV 连接器（host/disk tier），运行时 `dlopen` CUDA/HIP driver，因此无需 GPU toolkit 即可编译；
- `kv-core-pyo3` / `kv-tier-connector-pyo3`：Python 绑定（`max._kv_core`、`kv_tier_connector`）；
- `max-profiler`：NVTX/rocTX 标注 shim 的绑定。

该目录同时暴露了闭源边界：**"the private MAX Serve image bundles it as the `kv-tier-connector` wheel, whose packaging lives in `kv-tier-connector-pyo3/internal` and is not part of the open-source export."**

### 3.5 Kernels / max/lib

- `Kernels/lib/max_license/`：`WeightsRegistry`、`Tracing`、`MojoString` C API 等 MAX License 覆盖的胶水代码，被单独从 Apache 许可的 kernel 树中隔离出来。
- `max/lib/max_license/`：同类隔离目录。

---

## 4. 许可证边界（必须显式区分）

官方公告标题与 README 均使用 **source available**，而非 open source：

1. **MAX License（`Licenses/MAX_LICENSE`，Last Modified 2026-10-21）**
   - §1.1：授予复制、准备衍生作品、**以目标代码形式**公开展示/分发/再许可的权利；
   - §1.2：可随"有实质额外功能"的应用以目标代码再分发；
   - §1.3：仅允许为向上游贡献而 fork 并公开修改后的源码，**不允许源码再分发**；
   - §1.4(d)：禁止把 MAX 用作训练/微调数据或 AI 系统输入，以产出"重新实现或替代 MAX"的软件（允许用 AI 工具阅读、分析、改进 MAX）；
   - §2.1：可能采集遥测数据；§3.3：以托管服务形式商业部署时须展示 Modular 商标。
2. **Apache-2.0 + LLVM Exception**：Mojo 全部代码（`Licenses/README.md`），以及 `rust_kv` 全部 crate、部分 MAX 组件（逐文件声明）。
3. 2026-10-22 的 `e9a89164c` 修正了 `Licenses/APACHE_LICENSE` 顶部横幅：由"整个仓库 Apache"改为"仓库的一部分"，与根 `LICENSE`、README 的三处表述对齐。

> 研究含义：引用这些源码做架构分析没有问题；但"可自由再分发源码/用于训练替代实现"的假设不成立。

---

## 5. 仍未开放 / 显式空缺（UNKNOWN）

- **Mach**（闭源 Serving/产品层）：仓库无 `mach/` 目录；`rust_kv/README.md` 明确提到 `max-profiler` 从 `mach/crates/max-profiler` 迁出，以便连接器"不依赖闭源代码"。
- **合作方/保密 HAL 插件**：`Target/README.md` 明文禁止进入 monorepo。
- **`kv-tier-connector` 的打包流程**：位于 `kv-tier-connector-pyo3/internal`，不在开源导出范围。
- **Mojo 前端之外的编译器内部**：Mojo 编译器源码自 2026-08 起已在 `Mojo/lib/Compiler`、`Mojo/lib/KGENToLLVM` 等目录，但代码生成后端与闭源 LLVM 补丁的边界仍需逐文件核对（本次未展开，标记 `UNKNOWN`）。
- **性能结论**：本次仅做源码与文档静态审阅，未在真实加速器上运行，不构成性能或数值结论。

---

## 6. 对本研究库的影响

- 路线 A（AOT 静态编译）此前对 Modular 只能依据 Python 层与文档推断；现在可直接以 `GraphCompiler/` 的 pass 管线、MEF 产物边界、编译缓存机制作为一手证据，与 QAIC 的 QPC、TPU 的 XLA 编译缓存做同层对比。
- `SDK/include/HAL/M_driver_*.h` 是一份真实厂商的"硬件抽象 C ABI"范本，可直接用于第 14 节硬件抽象建议的对照。
- `rust_kv` 与 Python 侧 `jenga_block_pool.py` 构成"KV 池化 + 分层卸载"的完整证据链，可与 `modular/docs/01-modular-prefix-kv-cache-jenga.md` 交叉验证。

---

## 7. 复核命令

```bash
cd modular/third_party/modular
git log -1 --format='%H %ad %s' --date=iso
git show --stat d0462d003 | tail -3
git diff --shortstat e700d92fe1ba720701d5b9d233c791590199bfb1..HEAD
git ls-tree --name-only HEAD
```
