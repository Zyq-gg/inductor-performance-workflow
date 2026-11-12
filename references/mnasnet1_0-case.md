# mnasnet1_0：DCU 编译模式推理耗时异常

状态：已定位到 HIP Graph 回放路径的最小独立触发条件；底层根因与正式修复仍待确认。流程与指标定义见[通用分析流程](./workflow.md)。本报告中的绝对路径是原分析环境的历史记录；仓库附带的独立复现脚本见[这里](../scripts/micro_hip_graph_repro.py)，原始 trace 和其余临时探针未收录。

## 模型概况

`mnasnet1_0` 是 TorchVision 的 MNASNet 1.0 图像分类网络；TorchBench 通过 `torchbenchmark/models/mnasnet1_0` 包装该模型，默认评估 batch 为 32，使用 ImageNet-1K V1 预训练权重。网络以普通卷积、depthwise separable 卷积开头，接六组 inverted residual 块，最后经 1×1 卷积、全局空间平均和线性分类器输出 1000 类分数。原始异常场景是 BF16 推理，输入约为 `[32, 3, 224, 224]`；关键算子包括 BF16 depthwise 卷积、BatchNorm、ReLU、空间归约和 Linear。本地最小复现采用随机权重及更小的算子片段，只用于定位，不代表 TorchBench 的正式性能结果。

模型实现：`/usr/local/lib/python3.12/site-packages/torchvision/models/mnasnet.py`；TorchBench 包装：`/home/benchmark/torchbenchmark/models/mnasnet1_0/__init__.py`。

## 正式基线与修改后复测状态

原始 CSV 是用户初步验证的数据，来自 BW1101；当前本地设备是 BW1102，安装 PyTorch 2.13.0 / HIP 7.2.26365。因此跨设备数字不能计算修复收益。本地已有完整模型与小图的单变量复现（见下文），但尚未用原 TorchBench 驱动在当前设备完成同口径基线；源码也尚未修改。正式 accuracy、`compilation_latency` 和修改后 benchmark 均待测，不能把下面的局部复现或 trace 时长填作正式结果。

| 证据/阶段 | 设备与模式 | eager ms | compiled `abs_latency` ms | speedup | `compilation_latency` s | accuracy | 状态 |
|---|---|---:|---:|---:|---:|---|---|
| 用户原始 CSV | BW1101，BF16 推理，dynamic + Graph | 未单列 | 2998.422 | 0.003 | 未提供 | 未提供 | 已提供；非本地复测 |
| 用户原始 CSV | BW1101，BF16 推理，no_cudagraphs | 未单列 | 9.615 | 0.951 | 未提供 | 未提供 | 配置还改变了动态形状，不是单变量对照 |
| 修改前正式本地基线 | BW1102，目标配置 | 待测 | 待测 | 待测 | 待测 | 待测 | 待用原驱动复测 |
| 修改后正式本地结果 | 同设备同配置 | 待测 | 待测 | 待测 | 待测 | 待测 | 尚无源码修复 |

建议复测命令（先按测试文档在 `/home/pytorch` 激活环境和准备 TorchBench；`--accuracy` 与 `--performance` 分开执行），原始 CSV 与日志需写入独立结果目录：

```bash
python benchmarks/dynamo/torchbench.py --performance --cold-start-latency \
  --inference --bfloat16 --backend inductor --device cuda \
  --dynamic-shapes --dynamic-batch-only --only mnasnet1_0 \
  --output-directory /workspace/perf_results/mnasnet1_0/before \
  --output before_dynamic_graph.csv
```

## 最新结论（2026-11-11）

异常发生在 Graph replay，已用六层 BF16 eager 小图和普通 `torch.cuda.CUDAGraph` 独立复现；它不依赖 Inductor 或 CUDA Graph Tree。最小独立脚本见 `/home/pytorch/agent_space/micro_hip_graph_repro.py`。静态/动态图的可见 HIP 顶层拓扑相同，但 DTK 返回的 type-200 节点无法用公开 API 展开；两种图的捕获/回放输入地址各自稳定。两次有内容的 VMFault 日志均指向 BF16 MIOpen depthwise kernel，实际是否为 stale pointer 或分配越界仍需 DTK 内部节点及分配器证据。下文保留逐步排查过程及对照数据。

## 数据口径

- 来源：`torchbench数据对比 - dcu-eager-vs-compile.csv` 第 1 个模型，batch=32、bfloat16、inference。
- 表格中的 2998.422 ms 是编译后推理的 `abs_latency`，**不是首次编译时间**。`benchmarks/dynamo/common.py` 的 `speedup_experiment` 把计时中位数乘 1000 写入 `abs_latency`；首次编译另记为 `compilation_latency`，本对比表未提供。
- DCU dynamic：2998.422 ms、speedup 0.003；NV dynamic：3.667 ms、speedup 2.446。DCU no_cudagraphs：9.615 ms、speedup 0.951。后两种 DCU 模式同时改变了动态形状与 Graph 开关，不能直接作为单变量对照。

## trace 证据

trace 位于 `/home/profiler_traces.zip`；profile 在基准计时结束后另跑一次，trace 数值与 CSV 数值不可直接等同。

| DCU trace | eager 段 | compile 段 | 关键事件 |
|---|---:|---:|---|
| `inductor_dynamic_torchbench_bfloat16_inference_rocm.part0_mnasnet1_0_rank_0.json` | 约 12.05 ms | 约 23.43 s | `hipGraphLaunch` 后，`conv2d_nchw_bf16_fwd_implicitgemm_hw128x128_dilate_group_depthwise_common_gfx906` 单次事件约 23.42 s；`hipDeviceSynchronize` 等待约 23.43 s |
| `inductor_no_cudagraphs_torchbench_bfloat16_inference_rocm.part0_mnasnet1_0_rank_0.json` | 约 11.54 ms | 约 13.57 ms | 无超长 GPU kernel；compiled GPU kernel 合计约 9.10 ms |
| `inductor_max_autotune_torchbench_bfloat16_inference_rocm.part0_mnasnet1_0_rank_0.json` | 约 11.75 ms | 约 24.99 s | 同样在 `hipGraphLaunch` 后停顿；超长事件这次是 BF16 copy kernel，约 24.98 s |

dynamic 的 compiled `CompiledFxGraph` CPU 调用约 0.58 ms，长耗时落在 GPU 回放及同步。超长事件在不同 Graph trace 中落在不同 kernel 上，尚不足以认定该卷积实现本身是根因。`common.py` 中 `inductor_config.triton.cudagraphs = not args.disable_cudagraphs`，而 `cudagraph_trees.py` 的 `run_graph()` 调用 `self.graph.replay()`；trace 的 `hipGraphLaunch` 证实进入此路径。

## 当前判断与待验证项

当前证据指向 DCU/ROCm 的 Graph 回放路径导致运行时长时间停顿；编译器首次编译速度尚无数据。应使用相同动态形状设置，只切换 `--disable-cudagraphs` 重测，隔离 Graph 因素。随后检查回放是否因输入、内存地址、图节点或运行时调度而停顿，再决定在 Inductor 或更底层修复。

## 本地单变量复现（2026-11-10）

设备 BW1102，安装的 PyTorch 2.13.0 / HIP 7.2.26365；与表中 BW1101 不是同一设备。临时脚本 `/home/pytorch/agent_space/mnasnet_repro.py` 使用 torchvision MNASNet1_0、batch 32、BF16、仅标记 batch 维动态、`torch.compile(..., backend="inductor")`。权重随机初始化，不依赖下载；两组唯一配置差异为 `torch._inductor.config.triton.cudagraphs`。

- Graph 关闭：首次编译和运行约 47.46 s，后续单次推理 8.47、6.36、6.10、6.10 ms。
- Graph 开启：首次编译和运行约 14.13 s，随后两次 replay 分别约 **23.85 s**、**26.29 s**；第三次在同步等待时中断，避免重复等待。

这说明异常能在当前设备复现，而且不依赖预训练权重或原始 Benchmark 驱动。首次调用包含编译与捕获，不能用来比较推理性能。下一步用纯 eager CUDA Graph 捕获同一模型：若也很慢，优先查 HIP/MIOpen Graph；若正常，重点查 Inductor 图中额外的拷贝、布局变换与图捕获方式。

纯 eager CUDA Graph 对照脚本 `/home/pytorch/agent_space/mnasnet_eager_graph.py` 在捕获时打印 HIPBLASLt `operation not permitted when stream is capturing`，并在 120 秒超时内没有得到一次 replay 结果。此对照未完成，不能据此断定它与 Inductor 的慢 replay 有同一根因。

单个 BF16 depthwise Conv2d（32 通道、3x3、groups=32、32x112x112 输入）的编译 Graph 对照：首次调用约 11.62 s，第二次约 20 ms，第三次约 0.84 ms。单个 depthwise conv 本身没有复现 24 秒回放，说明问题需要更完整的模型图或特定算子组合。

探索性结构测试：前四层 Graph 约 1.03 ms；完整 `model.layers` Graph 约 10.1 ms；再接 `AdaptiveAvgPool2d(1)` 和 Flatten 约 10.1 ms；再接 classifier 约 10.1 ms；独立 BF16 Linear Graph 约 0.33 ms。这些测试**没有标记动态 batch**，而异常复现标记了动态 batch，所以不能据此把根因定位到某一层或排除某一层。

完整模型保持动态 batch 与 Graph 开启，只把原模型的 `x.mean([2,3])` 改为 `AdaptiveAvgPool2d(1).flatten(1)`，回放约 2.3 s，虽比原模型的 24–26 s 快，仍严重异常。池化替换不是完整修复。下一步固定完整模型与 Graph，仅切换动态 batch，检查交互。

该对照完成：完整原模型 **静态 batch + Graph 开启**，首次调用约 12.74 s，后续 33.56、10.11、10.62、9.72 ms，回放正常。结合“动态 batch + Graph 关闭”约 6 ms 与“动态 batch + Graph 开启”约 24–26 s，可以将触发条件定位为 **ROCm 上 Inductor 动态 batch 图的 CUDA Graph 捕获/回放**。静态图和动态非 Graph 路径均正常，当前不应直接修改 depthwise conv 或 BF16 Linear 的 codegen。

进一步控制动态 batch 后，单个 depthwise Conv2d 的 Graph 回放约 0.50 ms；完整 `model.layers` 的 Graph 回放约 6.16 ms，两者均正常。因此“动态形状 + Graph”并非对所有图都慢，异常还依赖后续算子或完整图的组合。

完整 `model.layers` 后接 `AdaptiveAvgPool2d(1).flatten(1)`，保持动态 batch 与 Graph 开启，后两次回放约 **4.35 s、4.33 s**。它未包含 classifier，故 BF16 Linear 不是异常所必需；引入空间归约或其与前面图的融合后即可触发。下一步单独测试 7×7 空间归约，区分归约 kernel 本身与模型上下文的影响。

独立的 BF16 `[32,1280,7,7] -> mean([2,3])`，动态 batch + Graph 开启，回放约 12.5 ms、0.32 ms，也正常。空间归约单独正常，卷积主体单独正常，两者放在同一个动态 Graph 后异常。现有 trace 将异常长耗时记在卷积或 copy kernel，但不同运行落点不一致，故更像图级回放/调度问题，而不是特定归约 kernel 的普通执行速度。

把特征提取和 `mean + classifier` 分成两个 `torch.compile` 调用，克隆中间张量以满足 CUDA Graph 输出生命周期检查后，整体第二次运行仍约 24.31 s。单纯在这里切分图不能修复。需要继续测试独立的动态 `mean + Linear` 组合；原始拆分不克隆时触发的是单独的 CUDA Graph 输出覆盖错误，不是性能异常。

独立的动态 `mean + Linear` Graph 回放约 0.36 ms；把输入改为 channels-last 后仍约 0.36 ms。故归约与 Linear 组合、及 channels-last 布局本身，均不能解释 24 秒异常。需要区分拆分后特征图 Graph 与头部 Graph 哪一个在跨调用时停顿。

拆分后分别同步计时：第二、三次 `features` 分别约 **13.30 s、12.08 s**，`head` 仅 3.54 ms、0.73 ms。`features` 单独循环时约 6.16 ms；只有在它与后续 Graph 交替运行后，下一轮特征图回放变慢。这提示 CUDA Graph Tree 的跨图状态/回放可能参与异常，但完整原模型单 Graph 也慢，尚不能定为唯一根因。

每轮显式调用 `torch.compiler.cudagraph_mark_step_begin()` 后，完整动态模型的回放仍约 8.81、8.79、7.95 s。此调用缩短了停顿，但没有解决问题。

## 结论与后续修改方向

1. 表格异常是编译后运行耗时，不是编译耗时。复现把必要条件收敛到 **动态 batch 的多算子 Graph 回放**：关闭 Graph 或使用静态 batch 即恢复毫秒级；单个动态 depthwise conv、独立空间归约、独立归约+Linear 都正常。HIP/ROCm trace 在 `hipGraphLaunch` 后同步等待，并把长耗时记在 Graph 内的某个 kernel；不同 trace 的 kernel 不相同，尚不能将根因归给具体卷积或 copy 指令。
2. 现阶段经过验证的局部规避是对这个模型的动态推理关闭 Graph：`--dynamic-shapes --dynamic-batch-only --disable-cudagraphs`。应重新用原 TorchBench 驱动测量精度、`abs_latency`、`compilation_latency`，避免把本地随机权重、BW1102 的数字当作 BW1101 正式结果。
3. 不建议未经其他动态模型回归测试就全局关闭 ROCm 动态 Graph；那可能牺牲本来受益的模型。进一步修复应先对同一模型的 Graph 捕获/回放记录图节点和输入内存信息，比较静态/动态图的 HIP Graph 节点序列，并在 HIP runtime 或 Inductor CUDA Graph Tree 中找到可复现的最小错误条件。代码生成/融合优化应在回放问题解决后再进行。

本地源码 `/home/pytorch` 已有用户改动：`benchmarks/dynamo/common.py` 为修改状态，另有未跟踪的 `flash_4_configs.p`；分析和后续实验需保留这些现有改动。

## Graph 节点与输入地址探针（2026-11-11）

已将 `/home/profiler_traces.zip` 解压到 `/home/profiler_traces/`（498 个归档条目，解压大小约 9.17 GB）。原 trace 保留在 `/home/profiler_traces/`，未覆盖。

临时探针 `/home/pytorch/agent_space/graph_probe.py` 通过 `CUDAGraph(keep_graph=True)` 保留 Graph 句柄，调用 HIP 的 `hipGraphGetNodes`、`hipGraphGetEdges` 和 `hipGraphDebugDotPrint`，同时记录 `CUDAGraphNode` 捕获与 replay 的输入 metadata。静态/动态图结果分别在 `agent_space/graph_static.json`、`agent_space/graph_dynamic_verbose.json`，对应 DOT 文件同目录。**这个探针改变了 Graph 的实例化时机和运行耗时；其耗时不可用于性能对比。**

- 两者的顶层 Graph 都是 6 个节点、5 条串行依赖边：5 个 HIP API 报告类型值 `200` 的节点，以及 1 个 48 字节 HtoD memcpy。类型 `200` 不在当前公开头文件的 `hipGraphNodeType` 枚举中；`hipGraphKernelNodeGetParams` 对这些节点返回 `hipErrorInvalidValue`。DOT 即使使用 verbose 选项也没有展开它们的内部 kernel。因此只能断定**顶层拓扑相同**，不能断定内部 kernel 节点序列相同。
- 静态图编译入口有 263 个输入，其中 262 个按静态地址处理，唯一非静态入口是 `[32,3,224,224]` BF16、NCHW contiguous 张量。动态图有 264 个输入，多出的非静态入口是值为 `32` 的符号 batch 整数；其余张量入口的形状、stride 和静态输入数量一致。
- 动态图的外部输入 `data_ptr` 在捕获与回放时相同；Graph Tree 分配的内部重建输入 `data_ptr` 也保持相同，但与外部地址不同，这是非静态输入复制路径的预期行为。未观察到输入地址漂移。
- 补跑第三次调用后，静态图也确认外部与内部输入 `data_ptr` 在捕获和真正 replay 之间分别保持不变。两种模式的地址结构无明显差异。

下一步使用未改变 `CUDAGraph` 实例的 profiler 对照静态/动态 Graph 回放，确认 kernel 序列、设备时间与同步等待。若仍无法展开类型 `200` 的内部节点，需要在 DTK HIP runtime 一侧获取 Graph 内部节点/命令队列调试能力，不能从顶层 DOT 推断内部依赖。

### 未改动 Graph 的 profiler 对照

临时脚本 `/home/pytorch/agent_space/profile_mnasnet_graph.py` 在静态模型捕获完成后启动 profiler。该次运行触发设备 VMFault，未生成可用 trace；诊断文件为 `/var/log/hymgr/1stVMFaultAnalysis_843881_980886485733781.txt`。匹配的命令为 `conv2d_nchw_bf16_fwd_implicitgemm_hw128x128_dilate_group_depthwise_common_gfx906`，grid `[32768,32,1]`、block `[256,1,1]`。原始无 Graph trace 中相同 kernel 用相同 grid/block 仅约 0.357 ms；动态 Graph trace 曾把约 23.42 s 记在这个 kernel 上。设备日志中 11 月 8 日另有两次同名 kernel 的 VMFault（grid y=128，未能证实是同一模型）。profiler 本身改变时序，静态模型普通运行约 10 ms，因此此次 VMFault 说明 Graph/该 kernel 的内存安全风险，但不能据此断言普通静态回放也错误。停止重复该 profiler 实验。

### 结构最小化（普通运行、动态 batch + Graph）

`agent_space/mnasnet_prefix_repro.py` 用前 N 个顶层模块接 `mean([2,3])`：N=8 稳态约 0.69 ms，N=12 约 4.23 ms，N=15 约 6.17 ms，N=16 约 6.17 ms，N=17 约 8.20–12.63 s。第 17 个模块是最终 `ReLU(inplace=True)`；**在固定此前 16 层的上下文中**，加入它会触发秒级回放。仅测试最终 `Conv1x1 + BatchNorm + ReLU + mean`，输入为真实前缀的 `[32,320,7,7]` 输出，稳态约 0.36 ms。不能因此认定 ReLU 本身是根因。

反向测试 `agent_space/mnasnet_tail_repro.py`：移除前 8 个顶层模块后，剩余模块加 mean 约 5.77 ms；移除前 4 个模块后约 5.93 ms；仅移除第一个 Conv2d 后仍约 6.22 s、4.18 s。最前面的 Conv2d 不是必需条件，触发图必须保留前 4 层中的至少一部分（首个 BN/ReLU/depthwise Conv）。正在测试从第 3 层开始的后缀。

从第 3 层 depthwise Conv 开始直到第 17 层再接 mean，回放约 7.18、7.23 s；从第 4 层开始正常。再把最后 ReLU 去掉（第 3 至 16 层加 mean），仍约 8.40、6.57 s。**裁剪图并非单调地消除问题**：完整前缀去最后 ReLU 恢复正常，裁剪掉最初三层后去最后 ReLU 仍慢。图裁剪同时改变布局、融合和 Graph 内存规划，现阶段的最小可复现片段是“第 3 至 16 层加 mean”，但尚未证明它是绝对最小片段。

针对该第 3 至 16 层加 mean 片段补充两个控制：动态 batch **关闭 Graph** 后稳态约 17.61、16.47 ms；**静态 batch + Graph** 仍约 8.09、8.18 s。故该裁剪片段展示的是更一般的 ROCm Graph 慢速条件，并不要求动态 batch；它与完整模型的“仅动态图异常”存在差别，不能未经进一步验证就当作完全同一根因。这个片段已足够小于完整模型，适合作为 HIP runtime 的下一步最小化起点。

继续缩短右端：第 3 至 14 层加 mean 约 5.94 ms；包含第 15 层最终 1×1 Conv（第 3 至 15 层加 mean）后，连续两次约 **22.99 s、27.19 s**。因此末尾 BN 和 ReLU 均不是该裁剪片段的必要条件。当前已知的更短复现片段是 `full.layers[3:15] + mean([2,3])`，输入为前 3 层真实输出 `[32,32,112,112]` BF16。待补该片段的关闭 Graph 和静态 batch 对照。

该片段的动态 batch **关闭 Graph** 稳态约 6.69 ms；**静态 batch + Graph** 第二次调用约 10.09 s。即该更短片段的慢速同样不要求动态形状，而 Graph 是必要条件。与完整模型静态 Graph 正常的差异再次说明：裁剪引发的代码生成、布局或内存规划变化会改变触发条件，不能把某一层本身等同于根因。

### 排除 Inductor Graph Tree

临时脚本 `agent_space/manual_graph_tail.py` 先关闭 Inductor 自带 Graph，用 Inductor 编译 `layers[3:15] + mean`，然后用普通 `torch.cuda.CUDAGraph()` 手动捕获该已编译调用。两次手动 replay 仍约 **6.24 s、4.53 s**。因此 Inductor CUDA Graph Tree 的状态管理不是该片段停顿的必要条件。

再把 `compiled` 换成原始 eager 模块，仍用普通 `torch.cuda.CUDAGraph()` 手动捕获同片段；两次 replay 约 **11.36 s、9.46 s**。该复现**完全不经过 Inductor**，说明该片段的 Graph 停顿可在 HIP/DTK runtime 与 eager 算子（主要为 MIOpen 卷积/BatchNorm 等）的组合中直接出现。它给 HIP runtime 提供了比完整 TorchBench 模型更直接的复现入口。尚不能据此认定原完整模型的所有异常都由同一内部错误导致，但两者均在 Graph replay 中停顿，且曾涉及同名 depthwise kernel。

### 纯 eager HIP Graph 的进一步最小化

`agent_space/manual_graph_tail.py --eager` 的对照：

| 捕获的片段 | Graph replay | 说明 |
|---|---:|---|
| `layers[3:8] + mean` | 0.41 ms | 正常 |
| `layers[3:9] + mean` | 11.87 s | 包含第一个完整倒残差块 |
| `layers[3:8] + 第一个倒残差块`（去掉 mean） | 11.58 s | mean 不是必要条件 |
| `layers[3:8] + 倒残差块前 4 层`（截至第二个 depthwise Conv） | VMFault | 日志 `/var/log/hymgr/1stVMFaultAnalysis_1143692_1023737521340083.txt`，匹配最早 depthwise kernel |
| `layers[3:8] + 倒残差块前 3 层` | 36.98 s | 第二个 depthwise Conv 也不是必要条件 |
| `layers[3:8] + 倒残差块第一层 1×1 Conv`（嵌套 Sequential） | 10.96 s；复测 6.20 s | 当前最短可稳定重现的模块片段 |
| 同样 3 条 Conv + BN32 + ReLU + BN16，扁平 Sequential | 0.47 ms | 运算层次相同，仅模块嵌套不同 |

扁平化的三条 Conv 单独约 0.28 ms；只加 ReLU、只加两个 BN、加 BN32+ReLU、或加 ReLU+BN16，均低于 0.5 ms。**不能把这些结果解释为“模块嵌套本身使 GPU 变慢”**：嵌套可能改变捕获时的张量生命周期、Graph 内存分配或 MIOpen 内部状态。下一步比较嵌套/扁平版本捕获的 HIP 顶层节点与中间张量地址复用关系。

两次独立 VMFault（静态完整模型的 profiler 运行，以及纯 eager 小图回放）均匹配 `conv2d_nchw_bf16_fwd_implicitgemm_hw128x128_dilate_group_depthwise_common_gfx906`；日志中报告的无效地址分别比该 kernel 参数中的第三个指针高 `0x1c00000`。这是重复出现的地址偏移，尚未确认第三个指针对应的缓冲区与分配大小，因此只能作为 HIP/MIOpen 调试线索，不能直接断定越界根因。

### 无 TorchVision / Inductor 的独立复现

独立脚本 `/home/pytorch/agent_space/micro_hip_graph_repro.py` 只使用 PyTorch eager：BF16 输入 `[32,32,112,112]`；依次为 depthwise 3×3 Conv（groups=32）、BN32、ReLU、1×1 Conv 32→16、BN16、以及嵌套 `Sequential(1×1 Conv 16→48)`。预热三次后用普通 `torch.cuda.CUDAGraph` 手动捕获并回放。

| 单变量条件 | eager 稳态 | Graph 回放 | 结论 |
|---|---:|---:|---|
| BF16、嵌套最后一层 | 约 0.76 ms | 118.15 ms | 独立复现明显退化 |
| BF16、最后一层扁平放置 | 约 0.76 ms | 0.49 ms | 对模块结构/捕获状态敏感 |
| FP32、嵌套最后一层 | 约 0.60 ms | 0.58 ms | 该独立复现依赖 BF16 路径 |
| BF16、嵌套、ReLU 改为 out-of-place | 约 0.76 ms | VMFault | inplace ReLU 不是必要条件；诊断文件为空，无法从该次日志确认具体 kernel |

在原 TorchVision 模块实例的更小六算子嵌套片段中，重复测得约 10.96 s 与 6.20 s；把完全相同的六个模块实例按相同顺序扁平化后约 0.47 ms。`flat_micro_graph.json` 与 `nested_micro_graph.json` 记录的捕获顺序都是 Conv→BN→ReLU→Conv→BN→Conv；归一化后的输入/输出地址复用关系也相同。两者 HIP 顶层 Graph 均为 `[memcpy, type-200, type-200]` 串行三节点。公开 HIP API 无法展开 type-200 节点内部，因此不能据此认定底层执行序列完全一致。模块嵌套不是 GPU 算子语义，差异更可能来自捕获期间的内存分配或 HIP/MIOpen 内部状态；这是推断，尚未拿到内部节点或分配器证据。

原 TorchVision 六算子嵌套片段的 batch 对照：batch 1 约 0.13 ms、2 约 0.17 ms、4 约 2.71 s、8 约 82.6 ms、16 约 112.7 ms、32 多次为数秒；非单调，不能概括为单一 batch 阈值。记录均为不同进程的一次 replay，需进一步重复确认分布。

## 当前定位

1. **Graph replay 是必要条件**：同片段正常 eager/关闭 Graph 为毫秒级，手动 HIP Graph 可复现秒级停顿或 VMFault。
2. **Inductor 和 CUDA Graph Tree 不是该小图错误的必要条件**：纯 eager、手动捕获即可出现。完整 `mnasnet1_0` 的原始动态模式异常与此机制高度相关，但还不能证明两者在 HIP runtime 内部是同一个缺陷。
3. **疑点在 DTK/HIP Graph 与 BF16 MIOpen 卷积及捕获内存状态的组合**。两份有内容的 VMFault 日志都指向同名 depthwise kernel，且无效地址相对第三参数指针有一致的 `0x1c00000` 偏移。需要 DTK runtime 提供 type-200 节点的内部命令和分配生命周期，才能判定是 stale pointer、错误分配大小还是执行顺序问题。
4. PyTorch 源码未改动；`/home/pytorch` 原有 `benchmarks/dynamo/common.py` 修改和未跟踪文件保留。现阶段对原模型的局部性能规避仍是动态推理关闭 Graph。不要把该规避当成底层问题已修复。
