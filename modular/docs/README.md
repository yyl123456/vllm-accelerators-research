# Modular (MAX / Mojo) 软件栈调研现状与知识库

> **资料角色声明**：
> 本目录归档 Modular Platform（MAX 推理栈 + Mojo 语言）的源码级调研。
> 与华为/高通/Google 等"vLLM 后端插件"不同，Modular 是**自带图编译器与运行时的独立推理栈**，是路线 A（AOT 图编译）在 vLLM 生态之外的直接对照样本。
> **固定源码 Revision**：`modular/third_party/modular/` 当前为 `da0db31d422e19ad5952d8d106b6da1d6899cf0a`（2026-10-23，Mojo 1.3.0.dev / MAX 26.8.0.dev）。

---

## 官方路线侧重与成熟度说明

- **AOT 图编译 + 预编译产物缓存路线（官方唯一交付主线，侧重程度：★★★★★）**：
  - MAX 的 Serving 路径固定为：Python 框架/管线构图 → `GraphCompiler` 优化并 lowering → 生成 **MEF** 产物 → `MLRT/GraphRT` 加载执行；MEF 缓存命中时跳过编译（`GraphCompiler/docs/ArchitectureSpecification.md`，QA-S2）。
  - 运行时通过 `SDK/include/HAL/M_driver_*.h` 的 C ABI 对接 `Target/` 下的 CUDA/HIP/Metal 插件，与 QAIC 的 AOT 交付形态、TPU 的 XLA 编译缓存属同一类工程结构。
- **动态/即时路径（辅助，侧重程度：★★☆☆☆）**：
  - 保留 eager 与自定义算子路径（规范中 QA-E1 要求 custom ops 在 eager 与 compiled 两种模式下均可用），但不是服务化交付主干。
- **闭源边界**：Mach 产品层、合作方 HAL 插件、部分打包流程不开源；MAX 主体代码为 **source-available（MAX License）**，仅 Mojo 与 `rust_kv` 等为 Apache-2.0 + LLVM Exception。

---

## 文档索引

| 编号 | 文档 | 主题 | 基线 |
|---|---|---|---|
| 01 | [Prefix KV Cache 的 Jenga 块池机制](01-modular-prefix-kv-cache-jenga.md) | 混合 KV 类型下的碎片化治理：LCM 几何对齐、Parked/Claimed 状态机、跨 Cache 征用 | `e700d92f` |
| 02 | [MAX 源码开放增量盘点（2026-10-22）](02-max-source-available-2026-10.md) | 新开放的 GraphCompiler / MLRT / Driver / HAL SDK / rust_kv，许可证边界与闭源缺口 | `da0db31d` |

---

## 已证实事实 vs 显式空缺

### 已证实（源码 + 官方文档静态审阅）
- [x] 图编译器、运行时、驱动与 HAL C ABI 已进入公开树，且带完整开发历史（PR #7370，2,266 文件 / +379,743 行）。
- [x] 发行方式改为**从源码构建**（`[OSS]` 提交 `762423501`、`51101d602`），公开树与 wheel 可对应。
- [x] MAX 主体的许可证为 MAX License（目标代码分发；源码仅限贡献性 fork），并非 OSI 开源。
- [x] KV Cache 的 Rust 实现（`rust_kv`）为 Apache-2.0，覆盖 Jenga 几何、前缀命中与分层卸载。

### 空缺 / 待复核（UNKNOWN）
- [ ] Mach 闭源产品层与 MAX Serve 私有镜像的内部实现。
- [ ] 合作方/保密 HAL 插件的接口实现细节（`Target/README.md` 明确排除在 monorepo 外）。
- [ ] 编译器后端与 LLVM 补丁的开闭源切分（未逐文件核对）。
- [ ] 任何性能/数值结论：当前均为静态审阅，无真实硬件实测。
