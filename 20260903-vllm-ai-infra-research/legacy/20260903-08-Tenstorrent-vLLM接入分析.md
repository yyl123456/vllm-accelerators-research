# Tenstorrent vLLM 接入分析

## 核心结论

Tenstorrent 路径是 upstream vLLM + 独立 `vllm-tt-plugin` + TT-Metal/TTNN 模型 adapter，而不是长期维护的 vLLM fork。plugin 使用标准 entry point，但在 scheduler、engine、worker、model runner、sampling、device mapping 和 KV capacity 上拥有较多控制权。

## 软件栈

```text
vLLM API / request protocol
        ↓
TT Platform + TT Scheduler/Engine/Worker/ModelRunner
        ↓
tt-metal model generator / reusable LLM runtime components
        ↓
TTNN / TT-Metal programs, trace, command queues
        ↓
Tensix compute + distributed SRAM + NoC + GDDR + Ethernet mesh
```

## Plugin 激活与版本

`SOURCE_IMPLEMENTED`：

- `vllm.general_plugins`: `tt_model_registry`；
- `vllm.platform_plugins`: `tt`；
- 只有 `ttnn` 可 import 时 `platform_plugin()` 才返回 `TTPlatform`；
- 安装脚本固定 `vllm==0.26.0`，用 `VLLM_TARGET_DEVICE=empty` 从源码安装，torch 由 tt-metal 环境拥有；
- 当前 vLLM 基线已比该 pin 更新，因此不能假定本地 main 可直接组合。

## 硬件架构

Wormhole n300 含两颗 ASIC，合计 24GB GDDR6、576GB/s、192MB 分布式 SRAM、128 个可用 Tensix cores，并以 PCIe、片内 200G 和跨卡 200G 连接。[官方卡规格](https://docs.tenstorrent.com/aibs/wormhole/index.html)

Tensix 的 FPU/SFPU 通过 unpacker/packer 和固定 tile register 工作；RISC-V data-movement kernels 通过 NoC 从远端 SRAM 或 DRAM 显式搬运数据。[compute/dataflow](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html)；[memory/NoC](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.html)

其优化单位不是单一 CUDA-like kernel，而是 tile layout、circular buffer、reader/writer/compute pipeline、core grid、NoC route 和 mesh program。

## 调度差异

仓库 `docs/SCHEDULING.md` 明确：

- 一个 TT step 为纯 prefill 或纯 decode，不混合两者；
- 支持 chunked prefill 取决于 tt-metal generator 是否能恢复 prefill；
- block-output model 不能使用同样的 split prompt 语义；
- async 主要优化 steady decode 的 host readback/sampling overlap；
- lane-DP 在单进程合并多个 lane；标准 DP 则是各 rank 独立 engine/scheduler/submesh。

这是硬件/模型 program 约束反馈到 scheduler 的清晰案例。它保留广义 continuous batching，但牺牲同一个 step 内的 mixed batch 灵活性。

## KV capacity

TT 不按 GPU profile 的方式发现空闲显存。worker 从模型 class 的 `max_tokens_all_users` 等能力推导 block 数，并把容量传回 vLLM KV planner。

因此：

- capacity 与具体模型、设备和 tt-metal adapter 强绑定；
- `max_model_len=-1` 可能把几乎整个 pool 给单请求，导致并发约为 1；
- serving 应按 `pool / target_concurrency` 显式选择上下文；
- registration、model adapter、release ModelSpec 是逐级增强但不同的支持证据。

## Trace、sampling 与异步

TT 支持模型能力声明，例如 async decode、prefix cache、sample-on-device、chunked prefill。device sampling 可以减少 D2H/host 工作，但必须保持 RNG、grammar、penalty、logprob 和 intermediate prefill 的一致性。

源码对 intermediate chunk 使用 host sampling/clone RNG，避免本不应产 token 的行推进 device RNG。这说明 sampling location 是请求状态协议的一部分，不只是性能优化。

async decode 的 deferred output/event 表示完成 readback/finalization，而非只表示提交。仓库文档宣称队列深度通常为 2；是否在所有模型/topology 下真实 overlap 仍需 trace 测量。

当前 revision 还显式处理了 async placeholder 与 preemption/强制 prefix-cache reset 的边界：未消费的 in-flight output 会阻止请求过早 resume，强制 reset 则向 runner 传递需要丢弃的 stale frame 计数。这是 `SOURCE_IMPLEMENTED` 的修复证据，但并不等于所有模型和设备上的 cancel/preemption 已经 `RUNTIME_VALIDATED`。

## 拓扑

一个 n300 是两颗 Wormhole；四张同主机 n300 为八颗芯片，可映射成 T3K `(1,8)`，前提是跨卡 Ethernet 完整并被 `tt-topology` 识别。若分布在多个 host，不能从 chip count 推断支持。

`TTPlatform.device_id_to_physical_device_id()` 还把一个 DP rank 映射为逗号分隔的设备 group string，偏离上游“单个 int physical ID”语义。plugin 自己记录了这一偏差；所有 Ray/topology/diagnostic consumer 都需要专项验证。

## 主要风险

- vLLM pin、plugin、tt-metal model 和 release image 必须严格匹配；
- model-owned scheduler/runner 能表达硬件能力，但增加上游升级面；
- 自定义 device ID namespace 可能遇到假设整数 ID 的共享路径；
- 模型支持具有 device/mesh/context/concurrency 限制；
- registration 不能证明实际 topology、数值正确性或性能；
- trace cache、fabric 和 program 初始化需要明确 warmup/失效策略。

## 证据结论

- plugin 架构、scheduler 和 capacity 路径：`SOURCE_IMPLEMENTED`。
- 公开 ModelSpec/支持等级：`RELEASE_CONTRACT`，必须重新绑定目标 release。
- 四张 n300 → 八芯片/T3K：硬件数量兼容为 `INFERENCE`；真实 mesh 为 `UNKNOWN`。
- correctness、性能、多 host、故障恢复：当前 `UNKNOWN`。
