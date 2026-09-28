# Stream、Event、Queue 与 Device Completion

## 文档契约

- **唯一问题**：host API返回、device执行和memory visibility之间有哪些阶段，如何映射统一Future。
- **In scope**：streams/queues/events、dependencies、async errors、buffer lifetime、cross-device completion。
- **Out of scope**：Engine async scheduling（28）、具体vendor runtime API。

至少区分：host enqueue accepted；命令进入device queue；前置依赖满足；kernel/DMA执行完成；结果对本device后继可见；对host/peer/remote可见；所有rank collective完成。Future若不声明对应层级，就不能作为buffer/KV reuse fence。

同一stream通常保序，不同streams需event/dependency；host线程顺序不自动形成device顺序。DMA和compute overlap要用明确events，错误event可能导致读未完成数据。graph/trace replay也依赖原captured stream/address关系。

异步错误常在后续sync才暴露，必须绑定提交step与artifact。发生错误后相关writes可能部分完成，buffer需quarantine；不能仅忽略output后复用。

跨device collective还需所有participants完成；单rank event完成不代表其他rank可见。P/D RDMA需要remote completion/ack或lease，而非sender local completion。

适配新runtime时应形成completion mapping表：enqueue handle、event记录/查询/等待、error propagation、host visibility、peer visibility、cancel能力。测试用delayed writer、cross-stream读、abort/reuse、rank failure和late completion检测use-after-free。

## 统一 completion lattice

```text
host accepted < device queued < local kernel/DMA complete
 < local consumer-visible < host/peer-visible < all-rank/remote commit
```

allocator free、KV reuse、output commit、remote lease release 所需层级不同，不能共用未注明语义的 `done`。错误后 buffer quarantine，直到明确 completion/reset；cancel 不保证 device 撤销。

`RECOMMENDATION`：backend 提交 API→lattice mapping 与 delayed/abort tests；`UNKNOWN`：无设备时 completion 只能源码推断。
