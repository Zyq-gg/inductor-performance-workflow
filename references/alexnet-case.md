# alexnet：反向卷积调优候选失败与布局转换分析

状态：DCU 侧训练退化已修复并复测，推理/训练布局规避已验证；NV 剩余差距缺少同口径 NV trace，尚不能精确归因。更新日期：2026-11-14。按[通用流程](./workflow.md)记录。

## 模型概况

`alexnet` 是 TorchBench 的 TorchVision 图像分类模型，使用 `torchvision.models.alexnet` 和 `AlexNet_Weights.IMAGENET1K_V1` 权重。网络包含五层卷积、ReLU 与多层最大池化，后接自适应池化和三层全连接分类器，输出 ImageNet-1K 的 1000 类分数。TorchBench 包装在 `/home/benchmark/torchbenchmark/models/alexnet/__init__.py`，输入由 `TorchVisionModel` 生成，为 `[128,3,224,224]` 的随机图像；batch 128 是该模型的训练/评估默认值。训练测试包括前向、反向和优化器更新。

## 原始异常和测量口径

来源：`/workspace/torchbench数据对比 - dcu-eager-and-nv-speedup.csv`。表头的“compile 耗时（毫秒）”是编译后稳态 `abs_latency`，不是首次编译时长；表中没有 `compilation_latency`。原始 DCU 为 BW1101，NV 为 A800；本机为 BW1102，跨型号/硬件绝对时延只作现象对照。eager 值按 `compiled × speedup` 反算。

| 模式 | 编译配置 | DCU compiled ms / speedup | DCU eager≈ ms | NV compiled ms / speedup | NV eager≈ ms | 正确性 |
|---|---|---:|---:|---:|---:|---|
| BF16 推理 | max-autotune，Graph 开 | 4.545 / 1.119 | 5.086 | 2.613 / 1.290 | 3.371 | 两卡 pass |
| BF16 推理 | noGraph | 4.740 / 1.070 | 5.072 | 2.404 / 1.395 | 3.354 | 两卡 pass |
| AMP 训练 | max-autotune，Graph 开 | 72.837 / 0.210 | 15.296 | 7.133 / 1.306 | 9.316 | 两卡 pass |
| AMP 训练 | noGraph | 14.470 / 1.037 | 15.006 | 7.146 / 1.305 | 9.325 | 两卡 pass |

两条推理行同时改变 max-autotune 与 Graph；训练行也是如此，因此需在同机补齐两开关四格矩阵。推理加速比偏低，训练 max-autotune 则严重退化，是优先分析对象。

## 环境与修改前基线

本机已安装 PyTorch 2.13.0 / HIP 7.2.26365、TorchVision 0.28.0，GPU 为 BW1102；PyTorch 源码提交 `ba64d4d7c428a007f69b9f228eeffa05a2f061cd`，TorchBench 仓库提交 `0fff8bf35400ec7b733d5080628fc79551205e36`。源码树原有 `benchmarks/dynamo/common.py` 修改和未跟踪 `flash_4_configs.p`，本次不覆盖。`/workspace/env.sh` 最后一行引用本机不存在的 MKL 文件，测试使用 `source /opt/dtk/env.sh`。从 `/tmp` 运行绝对路径入口并设置 `PYTHONPATH=/home/benchmark`，避免 `/home/torchbench` 目录被 harness 错选为 TorchBench 仓库。

基线输出目录：`/workspace/perf_results/alexnet/`。首次从 `/home/pytorch/benchmarks/dynamo` 运行 accuracy 时，路径探测优先选中 `/home/torchbench`，得到 `eager_fail_to_run`；该结果仅是环境错误，已改为上述启动方式重测。

本机训练 max-autotune+Graph 的 accuracy pass（2 图、6 次 graph break），compiled **72.909 ms**、speedup **0.201**，几乎复现原表 BW1101 的 72.837 ms。保持 max-autotune、只关闭 Graph，accuracy 仍 pass，compiled **73.112 ms**、speedup **0.201**。默认编译 noGraph 的 accuracy 也 pass，compiled **13.878 ms**、speedup **1.049**。三组性能均为 `--performance --cold-start-latency --repeat 5`；原始 CSV、日志和 trace 在 `train_max_auto/`、`train_max_auto_no_graph/`、`train_no_graph/`。`train_max_auto/accuracy.csv` 的首行是上述错误工作目录产生的失败记录，第二行才是有效复测。

默认编译 + Graph 的 accuracy 也 pass；compiled **14.083 ms**、speedup **1.043**，结果在 `train_default_graph/`。训练四格同机数据如下，均为 AMP/batch 128；四格 accuracy 均 pass。

| max-autotune | Graph | compiled ms | speedup | 原始 CSV |
|---|---|---:|---:|---|
| 关 | 关 | 13.878 | 1.049 | `train_no_graph/perf.csv` |
| 关 | 开 | 14.083 | 1.043 | `train_default_graph/perf.csv` |
| 开 | 关 | 73.112 | 0.201 | `train_max_auto_no_graph/perf.csv` |
| 开 | 开 | 72.909 | 0.201 | `train_max_auto/perf.csv` |

同一 max-autotune 设置内 Graph 开关相差约 0.2 ms，不能解释 59 ms 的退化；打开 max-autotune 才引入约 **5.25 倍**的 compiled 时延。

```bash
# 工作目录 /tmp；先 source /opt/dtk/env.sh。以下为训练异常格，其它格仅改变环境变量与 Graph 开关。
PYTHONPATH=/home/benchmark TORCHINDUCTOR_MAX_AUTOTUNE=1 \
python /home/pytorch/benchmarks/dynamo/torchbench.py \
  --performance --cold-start-latency --repeat 5 --training --amp \
  --backend inductor --device cuda --only alexnet --batch-size 128 \
  --export-profiler-trace \
  --profiler-trace-name /workspace/perf_results/alexnet/train_max_auto/trace \
  --output-directory /workspace/perf_results/alexnet/train_max_auto --output perf.csv
```

## 原始 trace 初步定位

六份原始 BW1101 trace 在 `/home/profiler_traces/`，包括四条原表配置和额外的 dynamic 推理/训练配置。profile 在正式计时后单独采集，下面的时长只用于定位，不替代 CSV。

训练 max-autotune 原始 trace 中，compiled `actual` 的 CPU 区段为 **73.31 ms**，其中 `hipDeviceSynchronize` 为 **69.48 ms**。它有两次 `hipGraphLaunch`：第一个图的 GPU annotation 约 **5.11 ms**，第二个约 **65.45 ms**；第二个图开始后到下一可见 GPU kernel 之间有约 **65 ms** 的空档。训练默认 noGraph 的 compiled CPU 区段为 **15.59 ms**，GPU kernel 总时间约 **14.01 ms**；额外的 dynamic Graph 训练 trace 的 compiled CPU 区段约 **14.01 ms**。但本机 max-autotune 保持开启、只关 Graph 后，稳态仍为 73.112 ms，因此这个长跨度**并非由 Graph 引起**，Graph 只是遮住了内部逐 kernel 时长。

本机 max-autotune noGraph trace 的 compiled `actual` 有 95 个 GPU kernel，合计约 **72.60 ms**。最慢的两个是 `triton_tem_fused_convolution_backward_40` **41.15 ms** 和 `_33` **10.20 ms**；其余卷积反向 kernel 还有数毫秒的开销。对应 autotune 日志中，11×11 和 5×5 权重梯度的最佳 Triton 候选分别约 **39.56 ms**、**10.19 ms**。日志同时出现 9 次 `Runtime error during autotuning`：ATen `convolution_backward.out` 候选因缺少 `out0` 参数被忽略。`torch/_inductor/kernel/conv.py` 的 `call_aten_dw`/`call_aten_dx` 仍用 `out1/out2/out3` 参数，而本机 `torch.ops.aten.convolution_backward.out._schema` 要求 `out0/out1/out2`。这解释了调优器为何无法比较 ATen 后端而选到慢的 Triton 反向卷积。源码修复及正式结果见下文。

## 局部源码修改

在 `/home/pytorch/torch/_inductor/kernel/conv.py` 的 `call_aten_dw` 和 `call_aten_dx` 中，改用 `torch.ops.aten.convolution_backward` 的函数式重载取请求的梯度，再复制到 `ExternKernelChoice` 传入的预分配 `out`。仅改变卷积反向 autotune 的 ATen 候选，不改 Graph/runtime。直接改参数名仍需提供三个不可选的 out 张量，因此采用函数式重载。小型 CPU 张量上的权重梯度、输入梯度均与直接 ATen 结果一致；`py_compile` 通过。另在 ROCm 上用普通卷积及分组、stride=2、channels-last 两组 FP16 输入比较函数结果与直接 ATen，输入/权重梯度均逐元素一致；脚本在 `/workspace/perf_results/alexnet/check_conv_choice.py`。

测试入口导入的是 `/usr/local/lib/python3.12/site-packages/torch`，不是未构建的源码树；为验证修复，已将同一补丁应用到安装版 `kernel/conv.py`，修改前的安装文件备份在 `/workspace/perf_results/alexnet/installed_conv.py.before`。源码与安装文件补丁后一致。修改后 accuracy 和 performance 分别使用新的空 `TORCHINDUCTOR_CACHE_DIR`，避免复用旧生成图。

修改后训练 `max-autotune + Graph` accuracy **pass**，仍为 2 图、6 次 graph break；完整模型 compiled 从同机修改前 **72.909 ms 降到 13.596 ms（约 5.36 倍）**，speedup 从 **0.201 升到 1.086**。性能两侧均为 AMP、batch 128、Graph 开、`TORCHINDUCTOR_MAX_AUTOTUNE=1`、`--repeat 5`，原始结果分别在 `train_max_auto/perf.csv` 与 `fixed_train_max_auto/perf.csv`。修改后 trace 的 compiled CPU 区段约 **14.02 ms**，最长 `hipDeviceSynchronize` 约 **10.83 ms**，原约 70 ms 的等待消失。修改后 autotune 日志没有 `Runtime error during autotuning`，11×11 权重梯度选择 `call_aten_dw` 约 **0.67 ms**，与旧 Triton 候选约 **39.44 ms** 构成直接机制验证；5×5 权重梯度从约 10.18 ms 的 Triton 候选改选约 0.57 ms 的 ATen 候选，输入梯度也有数处选择 ATen。

| 阶段 | AMP 训练配置 | eager≈ ms | compiled ms | speedup | compilation_latency s | accuracy | 原始结果 |
|---|---|---:|---:|---:|---:|---|---|
| 修改前 | BW1102，max-autotune + Graph | 14.646 | 72.909 | 0.201 | 97.257 | pass | `train_max_auto/` |
| 修改后 | 同设备、同模型和模式，仅修复 ATen 候选 | 14.761 | 13.596 | 1.086 | 188.505 | pass | `fixed_train_max_auto/` |

两次 `compilation_latency` 受 autotune 缓存和运行顺序影响：修改后性能进程使用空缓存，修改前性能进程在 accuracy 后运行，因此不据此计算冷编译回归或收益。修复增加了有效 ATen 候选，编译期可能需要更多候选测量；如需严谨比较，应另做两侧独立冷缓存复测。安装版是本机验证用补丁，后续安装/重建 torch 可能覆盖它；源码树中的修改是持久开发改动。

## BF16 推理的剩余低加速比

原始 BW1101 noGraph trace 的 compiled `actual` 有 42 个 GPU kernel，合计约 **4.59 ms**；eager `expected` 为 32 个、约 **4.95 ms**。五个主要卷积 kernel 在两路中均约 **2.59 ms**，说明编译没有显著改变该部分的执行时间。eager 其它 kernel 约 2.36 ms；compiled 的 Triton 约 0.87 ms、direct-copy 约 0.75 ms、其它约 0.39 ms，合计约 2.00 ms，因此整体只减少约 0.36 ms。`direct-copy` 值得做布局单变量试验，但关闭布局优化也可能改变卷积算法，需以正式 benchmark 判断净收益。

原始 trace 的 10 次 compiled `aten::copy_` 都把 channels-last 张量转为 NCHW contiguous。包括输入 `[128,3,224,224]`、中间激活 `[128,64,27,27]` 等和卷积权重；例如输入拷贝的目标 stride `[150528,50176,224,1]`、来源 stride `[150528,1,672,3]`。这与 adv_inception_v3 的布局转换方向一致，但 AlexNet 的总拷贝只约 0.75 ms，需单变量测试才能判断是否值得关闭布局优化。

本机 max-autotune+Graph 的 BF16 推理 accuracy pass（1 图、0 graph break），compiled **4.720 ms**、speedup **1.038**，与原表 BW1101 的 4.545 ms 同量级；结果在 `infer_max_auto/`。本机推理四格性能均使用 `--performance --cold-start-latency --repeat 10`。

同一 BW1102、BF16/batch 128 的四格矩阵现已完成，各格 accuracy 均 pass、1 图、0 graph break。`max-autotune` 由 `TORCHINDUCTOR_MAX_AUTOTUNE=1` 控制；Graph 关闭时加 `--disable-cudagraphs`。

| max-autotune | Graph | compiled ms | speedup | 原始 CSV |
|---|---|---:|---:|---|
| 关 | 关 | 4.786 | 1.018 | `infer_no_graph/perf.csv` |
| 关 | 开 | 4.831 | 1.015 | `infer_default_graph/perf.csv` |
| 开 | 关 | 4.426 | 1.096 | `infer_max_auto_no_graph/perf.csv` |
| 开 | 开 | 4.720 | 1.038 | `infer_max_auto/perf.csv` |

同条件下 Graph 在本轮没有正收益；max-autotune 在 noGraph 下改善约 **0.36 ms（7.5%）**，在 Graph 下只约 0.11 ms。0.05–0.3 ms 的 Graph 差异需独立进程复测波动，不将单次结果认作稳定回归。原表的 max-autotune+Graph 与默认 noGraph 两行同时改变两个开关，不能单独得出 Graph 效应。

同机 noGraph trace 显示：默认编译的 `actual` GPU 时间约 **4.48 ms**（5 个卷积约 2.50 ms，10 个 copy 约 0.72 ms，16 个 Triton 约 0.86 ms）；max-autotune 后约 **4.22 ms**（3 个卷积约 1.91 ms，6 个 copy 约 0.43 ms，17 个 Triton 约 1.61 ms）。调优减少部分卷积/布局转换，但增加 Triton 时间，净收益有限。推理低加速比主要来自卷积本身未产生大幅收益及残留布局拷贝；与 NV 的具体差别还需要同配置 NV trace。

保持默认 noGraph 的其余条件不变，只加 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0` 后，accuracy pass，compiled 从 **4.786 ms 降到 3.784 ms（约 1.27 倍）**，speedup 从 **1.018 升到 1.280**；独立进程复测为 **3.932 ms**。新 trace 的 compiled `actual` GPU 总时间约 **3.58 ms**：五个卷积仍约 **2.50 ms**，Triton 约 0.67 ms，**无 direct-copy kernel**。这把布局转换及相关融合路径证实为默认 noGraph 推理低加速比的重要原因。输出在 `infer_no_graph_layout_off/`。该开关影响整个进程，尚不能直接作为其它模型的全局修复。

同时打开 max-autotune、关闭 Graph 和布局优化时，accuracy pass，compiled **3.859 ms**、speedup **1.262**，结果在 `infer_max_auto_no_graph_layout_off/`。trace 中也没有 direct-copy，但 GPU 总时间约 3.55 ms，与不打开 max-autotune 的布局关闭配置相近；两者约 0.08 ms 差异不足以断定小幅优劣。max-autotune 的收益不能简单叠加到布局关闭配置。以本轮结果看，默认编译 + noGraph + 布局关闭是推理已验证的较快运行配置，仍应在目标部署环境独立复测。

## 训练的布局补测

保持默认编译、AMP 训练、batch 128、noGraph，其余条件不变，只关闭布局优化：accuracy pass（仍为 2 图、6 次 graph break），compiled 从 **13.878 ms 降到 12.618 ms（约 9.1%）**，speedup 从 **1.049 升到 1.219**。修改后 trace 的 compiled `actual` GPU kernel 合计约 **12.21 ms**、共 115 个，**无 direct-copy kernel**；原始 BW1101 默认 noGraph trace 有布局拷贝，说明同一机制也影响训练。结果在 `train_no_graph_layout_off/`。这仍是进程级配置实验；与源码修复、max-autotune 的组合见下一段。

源码修复后再同时设置 `TORCHINDUCTOR_MAX_AUTOTUNE=1` 和 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0`，保持 AMP 训练、batch 128、Graph 开：accuracy pass，compiled **11.716 ms**、speedup **1.226**，结果在 `fixed_train_max_auto_layout_off/`。相对于同一修复代码、相同 max-autotune/Graph、默认布局的 13.596 ms，布局关闭再减少约 **13.8%**。独立进程复测该组合为 **11.834 ms**；默认编译 noGraph + 布局关闭的两次结果为 **12.618/12.333 ms**。本次组合在已测配置中时延最低，但两配置差距小，不能据此保证在其它设备上仍有相同排序。

## 复现补丁后与布局实验

以下命令从 `/tmp` 执行，先 `source /opt/dtk/env.sh`。准确性与性能分开运行；`/home/pytorch` 的源码修改须已安装到实际导入的 torch，或使用包含相同补丁的构建。性能输出目录中保留 CSV、日志和编译指标。设置自定义 `TORCHINDUCTOR_CACHE_DIR` 的两次修复验证均从空目录开始；后续独立进程复测复用缓存，仅比较稳态，不比较冷编译时长。

```bash
PYTHONPATH=/home/benchmark TORCHINDUCTOR_MAX_AUTOTUNE=1 \
TORCHINDUCTOR_CACHE_DIR=/workspace/perf_results/alexnet/fixed_cache_acc \
python /home/pytorch/benchmarks/dynamo/torchbench.py \
  --accuracy --training --amp --backend inductor --device cuda \
  --only alexnet --batch-size 128 \
  --output-directory /workspace/perf_results/alexnet/fixed_train_max_auto \
  --output accuracy.csv
PYTHONPATH=/home/benchmark TORCHINDUCTOR_MAX_AUTOTUNE=1 \
TORCHINDUCTOR_CACHE_DIR=/workspace/perf_results/alexnet/fixed_cache_perf \
python /home/pytorch/benchmarks/dynamo/torchbench.py \
  --performance --cold-start-latency --repeat 5 \
  --training --amp --backend inductor --device cuda \
  --only alexnet --batch-size 128 --export-profiler-trace \
  --profiler-trace-name /workspace/perf_results/alexnet/fixed_train_max_auto/trace \
  --output-directory /workspace/perf_results/alexnet/fixed_train_max_auto \
  --output perf.csv
# 推理布局实验：改为 --inference --bfloat16 --disable-cudagraphs，
# 设置 TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0，并使用 infer_no_graph_layout_off 目录。
# 训练组合实验：另设 TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0，
# accuracy/performance 使用独立空缓存目录 fixed_layout_cache_acc/perf。
```

## 结论边界与剩余工作

在本机 BW1102 上，训练严重退化的直接原因是 Inductor 卷积反向 ATen 调优候选因调用 schema 不匹配而失败。源码补丁后，原始异常配置从 72.909 ms 降到 13.596 ms；布局关闭后该配置为 11.716/11.834 ms。推理弱加速比的已证实原因是重复 channels-last→NCHW 拷贝和卷积时间下限，默认 noGraph 关闭布局优化后从 4.786 ms 降到 3.784/3.932 ms。两种模式 accuracy 均通过。

原表 NV A800 训练 max-autotune speedup 为 1.306、本机修复与布局关闭组合为 1.226；原表 NV noGraph 推理为 1.395、本机推理布局关闭为 1.280。两卡硬件、软件和采集批次不同，这些数值只说明差距缩小，不能用来量化同硬件修复收益或推断 NV 的具体 kernel 优势。这里没有 NV profiler trace，也没有在同一软件栈上完成 NV 重测；若要解释剩余差距，需要同输入、同口径的 NV eager/compiled trace 与卷积算法选择记录。

`TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0` 是影响整个进程的运行配置；已验证 AlexNet，但其它模型可能因失去 channels-last 而退化。源码补丁通过 AlexNet 全模型 accuracy 和两组 ROCm 卷积梯度对照，尚未做多模型、大范围回归。补丁在 `/home/pytorch` 保持未提交，安装版为本机验证而同步修改；重新安装 torch 后需重新应用或构建源码。编译期候选集合变化可能改变首次编译耗时，本次缓存状态不足以作公平冷编译比较。

## 可复用发现

- `max-autotune` 的候选异常可能被记录为 `Ignoring this choice`，benchmark 仍返回 pass；必须查候选错误、选中 kernel 和耗时，避免把慢 kernel 误判为 Graph 回放停顿。
- Graph trace 把内部 kernel 隐藏时，保持其它设置不变只关 Graph，结合 noGraph 的逐 kernel trace 与 autotune 日志定位；本案例两路都约 73 ms。
- 对弱加速比模型，按 `copy_(dest, src)` 的 stride 判断布局方向，再用同机同输入关闭布局优化验证净收益；该开关与 max-autotune 的收益不保证叠加。
- 源码树与 benchmark 实际导入的安装版 torch 要分别核对；TorchBench 自动寻找目录可能选中名称相似但不含 `torchbenchmark` 包的路径。以上方法已回写[通用流程](./workflow.md)。
