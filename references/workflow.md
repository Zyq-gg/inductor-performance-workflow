# Inductor 模型性能分析与优化流程

版本：2026-11-14（随每个模型的分析更新）

本流程用于用户已初步确认存在性能问题的 TorchBench、HuggingFace、timm 等模型：编译后比 eager 慢，或编译相对 eager 的提升明显小于 NV 官方结果。每个模型保留一份独立报告；通用方法和已证实的共性经验回写本文件。当前实例：[mnasnet1_0 报告](./mnasnet1_0-case.md)、[adv_inception_v3 报告](./adv_inception_v3-case.md)、[alexnet 报告](./alexnet-case.md)。

## 1. 建立模型报告与测量口径

复制[模型报告模板](../assets/model-report-template.md)，命名为 `<模型名>_性能异常分析.md`。同一模型的不同 batch、精度或运行模式可在同一报告中分场景记录；问题机制不同且证据复杂时再拆分。先写明：

- 模型来源、用途、主体结构、输入输出、关键算子和 TorchBench/HuggingFace 包装方式；区分预训练权重和随机权重复现。
- 用户提供的异常数据、文件位置、指标定义、硬件及软件版本、推理/训练、batch、dtype、静态/动态形状、Graph 开关。
- `abs_latency` 是编译路径稳态耗时，单位 **ms**；`compilation_latency` 是首次编译耗时，单位 **s**；`speedup = eager / compiled`。没有提供的列写“未提供”，不能从 `abs_latency` 推断编译耗时。
- 结果来源标为“用户原始数据”“本地正式 benchmark”“局部复现”“profiler trace”，不同来源不混作同口径前后对比。

先检查工作树状态，保留已有改动。记录源码提交、安装版 PyTorch、HIP/ROCm 或 CUDA、GPU 型号及驱动、benchmark 仓库版本；若原始数据与本地设备不同，明确说明可复现性和数值可比性的边界。

## 2. 修改前复测基线

优先按[模型编译测试使用文档](./benchmark-guide.md)用原 benchmark 入口重测。保留完整命令、环境变量、原始 CSV、日志、输出目录、测量日期和进程退出码。先做 `--accuracy`，再做 `--performance --cold-start-latency`；两者需要分开运行。动态 batch 场景使用 `--dynamic-shapes --dynamic-batch-only`，Graph 关闭场景使用 `--disable-cudagraphs`。示例在激活环境后的 PyTorch 仓库根目录执行，并按当前机器调整输出路径：

```bash
python benchmarks/dynamo/torchbench.py --performance --cold-start-latency \
  --inference --bfloat16 --backend inductor --device cuda \
  --dynamic-shapes --dynamic-batch-only --only mnasnet1_0 \
  --output-directory /workspace/perf_results/mnasnet1_0/before \
  --output before.csv
```

具体模型换用 `huggingface.py` 或 `timm_models.py`。基线必须与目标异常的 batch、dtype、模式、输入形状、动态维和 Graph 设置一致。清楚记录缓存状态；若设置 `TORCHINDUCTOR_CACHE_DIR`，需确认 `--cold-start-latency` 的清缓存行为。性能至少报告原始多次结果和中位数（条件允许时不少于三次独立运行），同时保留波动范围。若时间、模型依赖或设备条件使正式复测不可行，写明原因及可用替代证据，状态标“待复测”，后续补齐；不得把局部复现数值当作正式基线。

## 3. 定位慢在哪一段

1. **先分阶段。** 区分模型加载、首次编译/捕获、后续稳态推理、同步等待。检查 `compilation_latency`、`abs_latency`、日志和阶段指标；不要因表头或文件名含 compile 就把稳态慢解释为编译慢。
2. **对齐 trace。** 找到与异常场景相同的 eager/compiled trace，核对设备、输入、运行次数和时间单位。profiler 通常在 benchmark 计时后单独运行，会引入开销，trace 时长不可直接代替 CSV 的稳态耗时。
3. **单变量对照。** 按需比较 eager/compiled、Graph 开/关、静态/动态 batch、max-autotune、dtype、输入布局、batch 大小、完整模型/切片；每次只改变一个条件。若原始两行同时改变两个开关，至少补成同一设备的四格矩阵，并在每格测 accuracy 与性能。记录数值及是否触发错误，避免把多个变化共同作用解释为单一根因。
   对 max-autotune 场景还要在日志中查 `Runtime error during autotuning`、被忽略的候选、`best_kernel` 及各候选耗时；完整模型仍能运行不代表调优选择空间正常。Graph trace 若把多个 kernel 包成一次 GPU annotation，保持调优设置不变、仅关 Graph 抓逐 kernel trace，再对比 annotation 时长和候选日志。
4. **找最小错误条件。** 先定位 CPU 调用、GPU kernel、Graph capture/replay 或同步；随后缩减模块/算子，保留同样的形状和输入来源。裁剪模型可能改变融合、内存规划和算法选择，因此每个缩减片段都要重做必要的单变量对照。
5. **选正确修复层。** 分别检验 benchmark 包装、Dynamo 图断开/重编译、Inductor 调度与生成代码、Triton kernel、CUDA Graph Tree、HIP/CUDA runtime 和底层库。能在纯 eager 手动 Graph 重现的问题，不应仅凭表象改 Inductor codegen；需要把是否同一内部缺陷作为假设记录。

对 Graph 问题，记录捕获与回放的输入 shape/stride/dtype、指针与别名关系、节点拓扑和依赖、kernel 序列、同步点、输出生命周期、设备错误日志。公开 HIP API 如只给出不透明节点类型，报告必须说明观察边界，不能据顶层拓扑相同断定内部节点相同。探针和 profiler 可能改变捕获时机，性能结果需用未插桩运行复核。

## 4. 假设、修改与验证

在报告里为每个假设写“证据—单变量实验—结果—保留/排除”。定位到可执行的修复点后，实施尽可能局部的代码修改，记下文件/提交、修改原因、预期机制、正确性影响、适用范围和可能的回归。临时关闭 Graph 等规避方式要与根因修复分开标记。

修改后用与第 2 步**完全相同**的入口、设备和输入设置重新跑 accuracy 与 performance，保存原始结果。记录 eager、compiled、speedup、`compilation_latency`、Graph 开关和波动；必要时测相关邻近模型/形状和 NV 对照。若换了设备或软件版本，不把前后数值放进同一性能提升计算；应重新建立同设备修改前基线。

| 阶段 | 环境/代码版本 | eager ms | compiled ms | speedup | compilation_latency s | accuracy | 原始结果 |
|---|---|---:|---:|---:|---:|---|---|
| 修改前 | 待填 | 待测 | 待测 | 待测 | 待测 | 待测 | 待填 |
| 修改后 | 待填 | 待测 | 待测 | 待测 | 待测 | 待测 | 待填 |

只有同口径数据才能计算改善倍数。没有修复时如实保留“已定位/待底层证据/待修改/待复测”的状态，不填虚构的“修改后”。

## 5. 每个案例结束时维护本流程

在模型报告末尾写“可复用发现”：新的定位顺序、可靠的对照、易误读指标、失败探针、可复用脚本和适用边界。若可迁移到别的模型，把简短规则及模型报告链接补到本文件；特定设备日志、全部试验数字和未证实猜测留在模型报告。修订本流程时更新下方记录。

### 已积累的共性经验

- **指标口径：** `abs_latency` 与 `compilation_latency` 分属稳态运行和首次编译，单位也不同；原始对比 CSV 可能缺后一列。见 [mnasnet1_0](./mnasnet1_0-case.md)。
- **Graph 定位：** 同一输入分别跑静态/动态及 Graph 开/关；把捕获、回放、内存和节点证据分开记录。trace 中“耗时最长的 kernel”不一定是缺陷所在，因为 Graph 停顿可落在不同事件上。
- **缩减验证：** 纯 eager 手动 Graph 能复现时，可排除 Inductor/CUDA Graph Tree 作为该小图触发的必要条件；是否解释完整模型仍需独立证据。裁剪后要重新验证静态/动态条件。
- **插桩影响：** Graph 探针、profiler 可能改变实例化时机、耗时，甚至触发 VMFault；必须把插桩观察与正式 benchmark 分开。
- **布局转换定位：** 把 GPU copy kernel 与 CPU `aten::copy_` 按 profiler `External id` 对齐，再按 `copy_(dest, src)` 顺序解读 shape/stride；用同一模型、同一 Graph 设置下的布局优化开/关验证性能影响。卷积算子自身时间相近时，重复的激活转换可能是主要差距。见 [adv_inception_v3](./adv_inception_v3-case.md)。
- **跨配置比较：** 表中 Graph 开关若与 max-autotune 或动态形状一起变化，不能把两行速度差直接归给 Graph；需补单变量测试。见 [adv_inception_v3](./adv_inception_v3-case.md)。
- **调优与 Graph 拆分：** 同一模型同时打开 max-autotune 和 Graph 时，分别补测两者开关的四格矩阵。adv_inception_v3 推理的 85→44 ms 改善主要来自 max-autotune 改变卷积实现与融合路径；Graph 两组同条件差异很小，不能从原表推断 Graph 加速。见 [adv_inception_v3](./adv_inception_v3-case.md)。
- **动态 batch 计时核对：** 阅读 benchmark 的实际输入路径，确认 eager 与 compiled 是否使用相同 shape。当前 PyTorch `benchmarks/dynamo/common.py` 的 `speedup_experiment` 在 `--dynamic-batch-only` 时只向 eager 的 `timed()` 传 `batch_size`，它会随机改变 batch；compiled 的 `timed()` 没有此参数。这样得到的 speedup 不能当同输入的精确比值，应先比较同配置的 compiled 绝对时延，并另做输入对齐复测。见 [adv_inception_v3](./adv_inception_v3-case.md)。
- **动态形状会改变布局选择：** 分析静态/动态差异时检查 `GraphLowering.decide_layout_opt`，当前源码遇到卷积输入的自由符号会跳过布局优化。动态配置中 copy kernel 消失，不能直接归功于 Graph；应补动态 noGraph 和静态 Graph 两格。见 [adv_inception_v3](./adv_inception_v3-case.md)。
- **调优候选失败：** max-autotune 可忽略调用失败的 ATen 候选，继续用慢的 Triton 候选完成编译；同时检查算子 schema、候选错误和选中 kernel 的耗时。alexnet 训练中 `convolution_backward.out` 参数与 schema 不符，ATen 候选被忽略，11×11 权重梯度退化到约 39 ms 的 Triton kernel；修复后候选约 0.67 ms。见 [alexnet](./alexnet-case.md)。
- **Graph 中的慢 kernel：** Graph 回放后的长同步不一定是 Graph runtime 停顿。保持 max-autotune 等条件不变，只关 Graph；若完整模型同样慢，展开 noGraph kernel trace。alexnet 两路均约 73 ms，noGraph 直接显示约 41 ms 和 10 ms 的慢反向卷积。见 [alexnet](./alexnet-case.md)。
- **实际导入的 torch：** 修改源码树后，先用 `torch.__file__` 核对 benchmark 是否运行安装版。若源码尚未构建，必须让测试环境使用待验证的补丁，并保留原安装文件与缓存隔离；不把只改源码树后的 benchmark 误称为修改后复测。TorchBench 的目录探测也可能优先选到名称相似但不是模型仓库的目录，应验证 `torchbenchmark.__file__`。见 [alexnet](./alexnet-case.md)。

### 修订记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-11-11 | 初版：统一模型报告、基线/复测口径、Graph 单变量定位与回写规则 | mnasnet1_0 分析与 benchmark 使用文档 |
| 2026-11-13 | 增加按 External id 追踪激活布局转换及跨配置比较边界 | adv_inception_v3 分析 |
| 2026-11-13 | 增加多开关四格矩阵及 max-autotune 与 Graph 的归因示例 | adv_inception_v3 推理复测 |
| 2026-11-13 | 增加动态 batch 的 eager/compiled 输入对齐检查 | adv_inception_v3 训练复测及 benchmark 源码 |
| 2026-11-13 | 增加动态形状对布局启发式的影响及训练四格对照 | adv_inception_v3 训练 trace 与 Inductor 源码 |
| 2026-11-14 | 增加调优候选失败、Graph 内慢 kernel、实际导入版本的核查步骤 | alexnet 源码修复与复测 |
