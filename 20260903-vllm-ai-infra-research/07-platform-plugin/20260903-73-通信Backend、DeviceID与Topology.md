# 通信 Backend、Device ID 与 Topology

## 文档契约

- **唯一问题**：平台如何把logical rank映射到physical device和通信拓扑。
- **In scope**：visible-device env、local/physical ID、Ray resources、dist backend、communicator registration、topology。
- **Out of scope**：rank group算法（60）、互联硬件原理（93）、vendor具体通信库。
- **证据基线**：vLLM `bb363db9...` Platform interface、distributed utils/device communicators。

容器/launcher可重映射visible IDs；local rank、framework device index、physical PCI/BDF或vendor ID不是同一概念。固定revision提供assigned physical GPU IDs的幂等设置，说明映射必须在worker初始化时冻结；不同值二次写入应失败。

Platform声明device control env和Ray resource key，Executor负责placement与local rank，Worker绑定device，ParallelState建立groups，device communicator选择collective实现。任何一层自行再次解释visible list都可能双重映射。

topology至少含device↔host NUMA、peer links、PCIe root、NIC affinity和跨节点fabric。logical TP group应尽量匹配高带宽域，但插件不能从连续device IDs推断物理邻近。

通信backend需声明supported collectives、dtype、async completion、stream语义、timeout/error。注册成功不代表all-to-all或heterogeneous topology可用。验证保存rank/device/topology表，执行collective correctness/timeout，覆盖visibility reorder、Ray placement、multi-host NIC选择和process restart。

## 映射链与不变式

```text
physical identity/BDF -> container-visible ID -> framework local device
 -> launcher local rank -> vLLM global/axis rank -> communicator member
```

`platforms/interface.py` 持有 resource/env/communicator hooks；`v1/executor/vllm_net_devices.py` 与 executors处理 assignments；`distributed/parallel_state.py` 消费 rank。映射必须单射到当前 epoch，并可逆查 physical device。

topology probe 只描述观察时状态；partition、reset、reschedule 后必须换 epoch。`SOURCE_IMPLEMENTED`：上述映射面；`RECOMMENDATION`：rank artifact 保存 BDF/NUMA/link/NIC 与 groups；`UNKNOWN`：vendor peer/fault semantics。
