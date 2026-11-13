# adv_inception_v3：DCU 编译后推理和训练性能分析

状态：推理和训练四格单变量矩阵均已完成，DCU 侧主要机制已定位；跨硬件剩余差距缺少 NV trace，尚不能归因。更新日期：2026-11-13。使用的[通用流程](./workflow.md)。

## 当前结论

本机推理四格单变量矩阵说明原表 85→47 ms 的改善主要来自 max-autotune 改变卷积实现和融合路径，而非 Graph。训练四格矩阵显示：静态输入下 Graph 仅从 221.18 降到 218.91 ms；动态 batch 下则从 200.52 降到 177.61 ms。动态形状本身还会避开 Inductor 布局优化，消除大量激活拷贝；所以原表训练 226→179 ms 是动态形状与 Graph 的共同作用。详细数据见下文。

这不是 `mnasnet1_0` 那样的 HIP Graph 长时间卡顿。原始 Graph trace 的回放正常；原表中 Graph 开启的行数值较快，但还伴随其它配置变化。两个退化的 noGraph 场景都出现大量 channels-last 激活被卷积入口转回 NCHW 的拷贝。保持模型、batch、精度和 Graph 设置不变，只用 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0` 关闭 Inductor 布局优化，本机推理从 84.36 ms 降至 45.09 ms，训练从 221.18 ms 降至 158.15 ms，两者 accuracy 均 pass。相同的布局开关在 `max_autotune + Graph` 推理中未见稳态收益（44.11 对 44.72 ms），因此不能把它作为四条配置的统一修复。该环境变量作用于整个进程，且其它模型尚无回归数据，所以它是当前 noGraph 场景的运行规避，不是通用源码修复。

## 模型概况

`adv_inception_v3` 是 timm 中 `inception_v3.tf_adv_in1k` 的旧别名，使用对抗训练的 ImageNet-1K 权重，任务是图像分类。网络是 Inception-v3：起始卷积和池化后，依次经过多分支 Inception A/B/C/D/E 块；各分支包含不同尺寸的卷积和池化，末端拼接、全局池化并输出 1000 类分数。多分支结构产生大量卷积、BatchNorm、ReLU、池化与 `cat`，布局传递较复杂。

`/home/pytorch/benchmarks/dynamo/timm_models.py` 用 `timm.create_model(..., pretrained=True)` 加载模型；模型实现为本机 `timm/models/inception_v3.py`。评估 batch=128、输入 `[128,3,299,299]`。推理用 BF16；训练用 AMP（trace 中卷积为 FP16），包括前向、反向和 SGD 更新。仓库模型列表给定 batch 128。当前本机 timm 1.0.22，原始采集时的 timm 版本未在对比表中给出。

## 原始异常和指标口径

来源：`/workspace/torchbench数据对比 - dcu-eager-and-nv-speedup.csv`。表头“compile 耗时（毫秒）”实际为编译后运行 `abs_latency`，**不是首次编译时间**；表中未提供 `compilation_latency`。下表 eager 为 `compiled × speedup` 反算的近似值。原始 DCU 是 BW1101，NV 是 A800；硬件不同，不能用两卡的绝对时延直接衡量编译器质量。

| 模式 | 配置 | DCU eager≈ ms | DCU compiled ms / speedup | NV eager≈ ms | NV compiled ms / speedup | DCU/NV 峰值 GB | 正确性 |
|---|---|---:|---:|---:|---:|---:|---|
| BF16 推理 | max_autotune，Graph 开 | 68.503 | 46.728 / 1.466 | 36.866 | 20.630 / 1.787 | 1.847 / 1.455 | 两卡 pass |
| BF16 推理 | no_cudagraphs | 68.480 | 85.386 / 0.802 | 36.593 | 19.780 / 1.850 | 1.130 / 0.711 | 两卡 pass |
| AMP 训练 | dynamic，Graph 开 | 209.978 | 179.468 / 1.170 | 163.156 | 112.058 / 1.456 | 5.475 / 5.420 | 两卡 pass |
| AMP 训练 | no_cudagraphs | 200.391 | 226.175 / 0.886 | 155.941 | 75.334 / 2.070 | 5.569 / 5.452 | 两卡 pass |

推理两行同时改变 max_autotune 与 Graph 开关；训练两行同时改变动态形状与 Graph 开关，故原表只能说明各自配置的结果，不能单独计算 Graph 的因果效应。

## 修改前本地基线

本机是 BW1102，PyTorch 2.13.0 / HIP 7.2.26365、timm 1.0.22，PyTorch 源码提交 `ba64d4d7c428a007f69b9f228eeffa05a2f061cd`。源码树原有 `benchmarks/dynamo/common.py` 修改和未跟踪的 `flash_4_configs.p`，本次未触碰。`/workspace/env.sh` 最后一行引用不存在的 MKL 文件，因此测量使用 `source /opt/dtk/env.sh`；从 `/tmp` 启动绝对路径脚本以加载已安装的 torch，不从未构建的源码树导入 torch。原始输出、日志和编译阶段指标在 `/workspace/perf_results/adv_inception_v3/`。

已完成的匹配配置：BF16 推理、batch 128、no_cudagraphs。accuracy 为 pass（1 图、0 graph break）；`--performance --cold-start-latency --repeat 5` 得到 compiled `abs_latency=84.361265 ms`、speedup `0.771276`、反算 eager 约 `65.06 ms`、`compilation_latency=34.854106 s`。它与原表 BW1101 的 85.386 ms、0.802 接近，但不是同设备的前后对比。

```bash
# 工作目录：/tmp；已先执行 source /opt/dtk/env.sh
python /home/pytorch/benchmarks/dynamo/timm_models.py \
  --accuracy --inference --bfloat16 --backend inductor --device cuda \
  --disable-cudagraphs --only adv_inception_v3 --batch-size 128 \
  --output-directory /workspace/perf_results/adv_inception_v3/before \
  --output accuracy_inference_no_graph.csv
python /home/pytorch/benchmarks/dynamo/timm_models.py \
  --performance --cold-start-latency --repeat 5 \
  --inference --bfloat16 --backend inductor --device cuda \
  --disable-cudagraphs --only adv_inception_v3 --batch-size 128 \
  --output-directory /workspace/perf_results/adv_inception_v3/before \
  --output perf_inference_no_graph.csv
```

| 阶段 | 环境/配置 | eager≈ ms | compiled ms | speedup | compilation_latency s | accuracy | 原始结果 |
|---|---|---:|---:|---:|---:|---|---|
| 修改前本地基线 | BW1102，BF16 推理，Graph 关，默认布局优化 | 65.06 | 84.361 | 0.771 | 34.854 | pass | `/workspace/perf_results/adv_inception_v3/before/` |
| 布局优化关闭实验 | 同设备、同输入，`TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0` | 65.28 | 45.086 | 1.448 | 39.133 | pass | `/workspace/perf_results/adv_inception_v3/layout_off/` |
| 修改前本地基线 | BW1102，AMP 训练，Graph 关，默认布局优化 | 191.68 | 221.179 | 0.867 | 218.182 | pass | `/workspace/perf_results/adv_inception_v3/train_before/` |
| 布局优化关闭实验 | BW1102，AMP 训练，Graph 关，其余同上 | 191.50 | 158.147 | 1.211 | 198.506 | pass | `/workspace/perf_results/adv_inception_v3/train_layout_off/` |

`--repeat 5` 是单次 harness 运行中的五组计时，不是五次独立进程；后续若形成修复结论，仍需独立进程复测波动。Graph 开启的推理和训练配置均在下文补测。不以 profiler 时长替代正式 benchmark。关闭布局优化后的本地编译耗时从 34.854 s 变为 39.133 s，属于不同编译图的冷编译结果；这次稳态收益不应与冷编译收益混淆。

AMP 训练 noGraph 的本地 accuracy 也已 pass；该场景捕获 2 个不同图，报告 6 次 graph break（5 个不同位置），accuracy 运行中的编译耗时约 169.93 s。`--performance --cold-start-latency --repeat 3` 得到 compiled `221.179469 ms`、speedup `0.866640`（反算 eager 约 191.68 ms）、编译耗时 `218.181941 s`、7 次 graph break；与原表 BW1101 的 226.175 ms、0.886 接近。结果在 `/workspace/perf_results/adv_inception_v3/train_before/`。推理的 0 graph break 结果不能直接用于解释训练开销。

训练中设置 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0` 后的 accuracy 也 pass，仍为 2 个图、6 次 graph break。`--performance --cold-start-latency --repeat 3` 得到 compiled `158.146727 ms`、speedup `1.210901`（反算 eager 约 191.50 ms）、编译耗时 `198.505556 s`，较同设备默认布局的 221.179 ms 缩短约 **28.5%（1.40 倍）**。性能运行中的 graph break 都为 7 次，因此该改善不是 graph break 数量减少造成的。结果在 `/workspace/perf_results/adv_inception_v3/train_layout_off/`；计时所用的唯一编译配置变化是 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0`。此性能命令还增加了 `--export-profiler-trace`，但 `common.py` 在计时循环结束后另行采集 trace，未把 profiler 开销计入 CSV。两次编译耗时随生成图不同而变化，不据此声称冷编译优化。

```bash
# 工作目录：/tmp；已先执行 source /opt/dtk/env.sh
python /home/pytorch/benchmarks/dynamo/timm_models.py \
  --performance --cold-start-latency --repeat 3 \
  --training --amp --backend inductor --device cuda --disable-cudagraphs \
  --only adv_inception_v3 --batch-size 128 \
  --output-directory /workspace/perf_results/adv_inception_v3/train_before \
  --output perf_train_no_graph.csv
```

布局关闭的 accuracy 使用与各自基线相同命令并在进程前设置 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0`。性能复测的完整命令如下；trace 采集位于 benchmark 计时循环之后：

```bash
# 工作目录：/tmp；已先执行 source /opt/dtk/env.sh
TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0 python /home/pytorch/benchmarks/dynamo/timm_models.py \
  --performance --cold-start-latency --repeat 5 \
  --inference --bfloat16 --backend inductor --device cuda --disable-cudagraphs \
  --only adv_inception_v3 --batch-size 128 --export-profiler-trace \
  --profiler-trace-name /workspace/perf_results/adv_inception_v3/layout_off/trace \
  --output-directory /workspace/perf_results/adv_inception_v3/layout_off \
  --output perf_inference_no_graph.csv
TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0 python /home/pytorch/benchmarks/dynamo/timm_models.py \
  --performance --cold-start-latency --repeat 3 \
  --training --amp --backend inductor --device cuda --disable-cudagraphs \
  --only adv_inception_v3 --batch-size 128 --export-profiler-trace \
  --profiler-trace-name /workspace/perf_results/adv_inception_v3/train_layout_off/trace \
  --output-directory /workspace/perf_results/adv_inception_v3/train_layout_off \
  --output perf_train_no_graph.csv
```

## max-autotune + Graph 推理补测

按原表相同的 BF16 推理、batch 128、`TORCHINDUCTOR_MAX_AUTOTUNE=1`，不加 `--disable-cudagraphs`，在 BW1102 上做同机对照。两次 accuracy 均 pass，均为 1 张图、0 graph break；性能使用 `--cold-start-latency --repeat 5`。完整输出在 `/workspace/perf_results/adv_inception_v3/graph_max_auto_default/` 和 `graph_max_auto_layout_off/`。

| 布局优化 | eager≈ ms | compiled ms | speedup | 编译耗时 s | compiled 峰值 GB |
|---|---:|---:|---:|---:|---:|
| 默认开启 | 65.88 | 44.110 | 1.493 | 255.413 | 1.847 |
| `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0` | 65.96 | 44.725 | 1.475 | 166.919 | 1.670 |

默认布局的稳态与原表 BW1101 的 46.728 ms、speedup 1.466 接近；关闭布局优化后没有观察到 noGraph 那样的加速，约 0.6 ms 差异不足以判断小幅好坏。冷编译时间因不同布局与 autotune 缓存/候选不同而变化，不以此声称编译加速。

```bash
# 工作目录：/tmp；已先执行 source /opt/dtk/env.sh
TORCHINDUCTOR_MAX_AUTOTUNE=1 python /home/pytorch/benchmarks/dynamo/timm_models.py \
  --performance --cold-start-latency --repeat 5 --inference --bfloat16 \
  --backend inductor --device cuda --only adv_inception_v3 --batch-size 128 \
  --output-directory /workspace/perf_results/adv_inception_v3/graph_max_auto_default \
  --output perf.csv
# 第二次保持上述命令，只加 TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0，
# 并把输出目录改为 graph_max_auto_layout_off。
```

### 推理：拆开 Graph 与 max-autotune

在同一台 BW1102、同一 BF16/batch 128 输入下补全四格单变量矩阵，accuracy 均 pass、0 graph break。Graph 不加 `--disable-cudagraphs` 即开启；max-autotune 用 `TORCHINDUCTOR_MAX_AUTOTUNE=1` 控制。四次性能均用 `--repeat 5`。

| Graph | max-autotune | compiled ms | speedup | 对应结果 |
|---|---|---:|---:|---|
| 关 | 关 | 84.361 | 0.771 | `before/perf_inference_no_graph.csv` |
| 开 | 关 | 85.882 | 0.781 | `graph_default/perf.csv` |
| 关 | 开 | 43.410 | 1.506 | `no_graph_max_auto/perf.csv` |
| 开 | 开 | 44.110 | 1.493 | `graph_max_auto_default/perf.csv` |

这组数据表明 **max-autotune 是主要收益来源**；Graph 在两组内部的约 0.7–1.5 ms 差异很小，单次运行不足以证明小幅好坏。原表把 max-autotune 与 Graph 同时打开，因此不能把 85→47 ms 归因于 Graph。额外的 noGraph max-autotune profiler trace 显示：与原始默认 noGraph trace 相比，compiled `aten::_convolution` 从 94 次降到 17 次，`aten::copy_` 从 148 次降到 34 次，GPU kernel 从 484 个降到 271 个；direct-copy GPU 时间从约 43.54 ms 降到 6.08 ms。出现 `triton_tem_fused_*convolution*` 模板及与 BN/ReLU 的融合，说明调优改变了卷积实现与布局转换路径。跨 BW1101/BW1102 的 trace 时长只作机制对照，正式速度比较使用本机 CSV。

本机 max-autotune noGraph accuracy 的首次尝试在 360 秒时超时，日志显示模型准备约 260 秒；改用已缓存的权重并设置 `HF_HUB_OFFLINE=1` 后通过。性能命令也使用该缓存设置，profile 仍在计时后单独运行。两份日志及结果在 `/workspace/perf_results/adv_inception_v3/no_graph_max_auto/`。

### 训练：拆开动态 batch 与 Graph

在同一 BW1102、AMP 训练、batch 128 下补齐四格矩阵。静态输入是不加 `--dynamic-shapes --dynamic-batch-only`；动态输入同时加两项。Graph 关闭时加 `--disable-cudagraphs`。四格 accuracy 均 pass，均为 2 张图、6 次 graph break；性能运行都用 `--cold-start-latency --repeat 3`，其 graph break 均为 7 次。

| 动态 batch | Graph | compiled ms | 输出 speedup | compilation_latency s | 结果目录 |
|---|---|---:|---:|---:|---|
| 关 | 关 | 221.179 | 0.867 | 218.182 | `train_before/` |
| 关 | 开 | 218.911 | 0.873 | 84.481 | `static_train_graph_default/` |
| 开 | 关 | 200.516 | 5.133* | 80.767 | `dynamic_train_no_graph/` |
| 开 | 开 | 177.606 | 4.396* | 181.389 | `dynamic_train_graph_default/` |

\*动态行的 speedup 由于本机 harness 对 eager 和 compiled 使用不同 batch，不能按同输入加速比解读；见下文。比较相同动态设置的 compiled 绝对时延，Graph 开启缩短约 **22.91 ms（11.4%）**。静态设置仅缩短约 **2.27 ms（1.0%）**，一次运行不足以证明这点小差异。保持 Graph 关闭、仅打开动态 batch 则从 221.179 降至 200.516 ms（约 **9.3%**）。原表静态 noGraph 与动态 Graph 的性能差由两项变化共同造成，并有明显交互；不能简单把动态、Graph 的收益推广到对方模式。四次冷编译耗时受缓存与运行顺序影响，未进行独立冷缓存复测，不据此比较编译性能。

动态 noGraph 的本机 trace 在 `actual` 段有 **1445 个 GPU kernel、约 171.02 ms**；整个 trace 中只有 **2 个 direct-copy kernel、约 0.09 ms**，Graph launch 为 0。它与静态默认 noGraph 原始 trace 的约 148 个 direct-copy 形成机制对照。本机源码 `torch/_inductor/graph.py:GraphLowering.decide_layout_opt` 遇到卷积输入含自由符号时返回 False，跳过布局优化；动态 batch 会触发这一条件。故动态配置的拷贝消失不能归功于 Graph。相较本机静态布局关闭的训练 trace（1454 个 GPU kernel、约 154.77 ms），动态 noGraph 仍有额外开销，表明布局转换并非训练的唯一问题。

```bash
# 工作目录：/tmp；先执行 source /opt/dtk/env.sh；已缓存的权重设置 HF_HUB_OFFLINE=1
HF_HUB_OFFLINE=1 python /home/pytorch/benchmarks/dynamo/timm_models.py \
  --performance --cold-start-latency --repeat 3 --training --amp \
  --backend inductor --device cuda --dynamic-shapes --dynamic-batch-only \
  --disable-cudagraphs --only adv_inception_v3 --batch-size 128 \
  --export-profiler-trace \
  --profiler-trace-name /workspace/perf_results/adv_inception_v3/dynamic_train_no_graph/trace \
  --output-directory /workspace/perf_results/adv_inception_v3/dynamic_train_no_graph \
  --output perf.csv
# 对照格分别去掉 --disable-cudagraphs、去掉两项 dynamic 参数，
# 并把输出目录改为相应的目录；accuracy 用相同配置另行运行。
```

## Trace 定位：与 mnasnet 的 Graph 停顿不同

**动态 batch 的 speedup 口径注意：**本机 benchmark 源码 `/home/pytorch/benchmarks/dynamo/common.py` 的 `speedup_experiment` 在 `--dynamic-batch-only` 时仅给 eager 的 `timed()` 传入 `batch_size`；`timed()` 会把输入 batch 随机变为原值的 80%–120%，compiled 则保持原始 batch 128。两路计时没有完全对齐输入，动态训练的 `speedup` 不应当作严格的同输入加速比。本机动态 Graph 复测的 compiled 为 177.606 ms，与原表 179.468 ms 接近，但其输出 `speedup=4.396` 反算 eager 约 780.7 ms，显著偏离原表约 210.0 ms，具体原因还需独立复测。分析 Graph 因果效应时优先比较同机同编译输入下的 compiled 绝对时延。

五份原始 BW1101 trace 位于 `/home/profiler_traces/`，其中四份对应表中四配置，另有 BF16 推理 dynamic trace。profile 是计时后单独运行，以下时长仅用于定位，不能直接代替 CSV。

| trace 配置 | eager `expected` CPU 段 | compiled `actual` CPU 段 | Graph 调用 | 主要观察 |
|---|---:|---:|---|---|
| 推理 max_autotune | 71.54 ms | 48.43 ms | 1 次 `hipGraphLaunch` | compiled GPU annotation 约 43.98 ms；正常回放，无秒级停顿 |
| 推理 dynamic（附加 trace，非本表四行） | 71.09 ms | 48.63 ms | 1 次 `hipGraphLaunch` | compiled GPU annotation 约 46.13 ms；同样无异常停顿 |
| 推理 no_cudagraphs | 71.81 ms | 87.12 ms | 0 | compiled GPU kernel 合计约 83.94 ms，eager 约 68.30 ms |
| 训练 dynamic | 200.89 ms | 181.31 ms | 2 次 `hipGraphLaunch` | 两个 compiled GPU annotation 约 116.64 + 58.92 ms；正常回放 |
| 训练 no_cudagraphs | 203.31 ms | 331.33 ms | 0 | compiled GPU kernel 合计约 222.38 ms，eager 约 198.89 ms；profiler 对 CPU 段影响明显 |

因此证据不支持把此模型归为 mnasnet 那样的 Graph replay 卡顿。原始数据中 Graph 打开的配置反而较快，但这些行还伴随其它参数变化；上文两套四格矩阵已分离推理 max-autotune、训练动态 batch 与 Graph 的影响。

## 定位到重复布局转换

在推理 no_cudagraphs 的 `actual` 区段，148 个 `at::native::direct_copy_kernel_cuda` GPU kernel 合计约 **43.54 ms**。通过 kernel 的 `External id` 对回 CPU `aten::copy_`，其中 98 个 batch=128 的激活拷贝合计约 **42.97 ms**，50 个权重/其它拷贝仅约 0.57 ms。训练 no_cudagraphs 也有 148 个同类拷贝，约 **43.61 ms**，其中 98 个激活拷贝约 43.04 ms。它们不是 eager 区段的相同拷贝模式。

`aten::copy_(dest, src)` 的 stride 显示方向为 **channels-last → NCHW contiguous**。例如 `[128,768,17,17]`：目标 stride `[221952,289,17,1]`，来源 stride `[221952,1,13056,768]`。此拷贝被 `aten::contiguous`/`aten::clone` 包在 `aten::_convolution` 内，说明卷积入口收到 channels-last 激活后，在该路径上转成 NCHW。

按父级 `_convolution` 的权重尺寸继续归因，98 次激活拷贝中有 40 次发生在 1×1 卷积（GPU 合计约 29.22 ms）、17 次发生在 3×3 卷积（约 9.90 ms），其余在 1×7、7×1 等分支卷积。瓶颈分散在多条 Inception 分支，并非单个异常 kernel。

本地单变量验证：只设置 `TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0`，其余配置、batch 和输入保持不变；accuracy 仍 pass，编译后稳态从 **84.361 ms 降到 45.086 ms（约 1.87 倍）**，speedup 从 **0.771 升到 1.448**，compiled 峰值显存从 1.129848 GB 降到 0.961482 GB。新 profiler trace 中 compiled 区段的 direct-copy kernel 从原始 trace 的 **148 个、43.54 ms** 降到 **3 个、0.37 ms**，GPU kernel 总时间从原始 BW1101 trace 的 83.94 ms 降为本机 43.22 ms。跨 BW1101/BW1102 的 trace 时长只能作机制对照；正式性能收益只用本机两组 benchmark 计算。该结果支持重复布局转换是默认 noGraph 推理退化的主要原因；还需独立进程复测波动。Graph 开启时的 max-autotune 配置已在上文回归，关闭布局优化没有进一步收益。

训练布局关闭后的本地 trace 也显示 compiled 区段的 direct-copy kernel 仅 **8 个、1.09 ms**，Triton kernel 为 694 个、约 60.67 ms，全部 GPU kernel 为 1454 个、约 154.77 ms。原始 BW1101 训练 noGraph trace 分别为 148 个 direct-copy、1296 个 Triton、2196 个全部 kernel；跨设备的时间不能直接计算收益，但 kernel 数量和拷贝模式的变化与本机 benchmark 改善一致。训练残余开销仍包含大量 Triton kernel，不能认为布局关闭已解决所有性能差距。

具体选择点在 `/home/pytorch/torch/_inductor/graph.py:GraphLowering.decide_layout_opt`，推理按卷积 FLOP 与一组经验倍率决定是否使用 channels-last；源码注释说明倍率需要按硬件更新。实际卷积路径见 `torch/_inductor/kernel/conv.py` 的 `require_channels_last`。本案例中启用布局优化后，某些卷积仍在 ATen `_convolution` 内把激活转回 NCHW。`TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0` 是已验证的**当前模型运行配置规避**，但环境变量作用于整个进程，尚不能安全推广到其它模型，也不是源码修复。

推理 eager 与 compiled 的 MIOpen 相关 kernel 时间都约 26.5 ms，训练都约 88.4 ms，差距主要不在这些已观察到的卷积 kernel 自身。推理 noGraph compiled GPU 总时间比 eager 多约 15.6 ms，尽管转换本身有 43 ms，表明其它融合/算子优化抵消了部分开销。训练 noGraph compiled 有 2196 个 GPU kernel（其中 1296 个 Triton），eager 为 1510 个；compiled 还产生约 5471 次 `hipPointerGetAttribute`，CPU 调度和 GPU 算子数也值得后续定位。profiler 下的 CPU 段差值大于正式 benchmark 差值，不能直接用于量化稳态收益。

## 假设、实验与下一步

| 假设 | 对照与证据 | 状态 |
|---|---|---|
| Graph 回放像 mnasnet 一样卡住 | Graph trace 回放为几十至百余毫秒；推理四格矩阵中 Graph 几乎无差异，动态训练中 Graph 反而有约 11.4% 稳态收益 | 不支持作为当前异常主因 |
| 大量布局转换压低 noGraph 加速比 | 推理 98 个激活 NHWC→NCHW 拷贝占约 43 ms；关闭布局优化后降到 3 个 copy、稳态加快约 1.87 倍；训练也从 221.18 降到 158.15 ms | 已确认是两个 noGraph 配置的主要瓶颈 |
| 推理原表 Graph 开启导致 85→47 ms 提速 | 本机 Graph/max-autotune 四格矩阵：max-autotune 关闭约 84–86 ms，开启约 43–44 ms | 否；主要来自 max-autotune |
| 训练原表 226→179 ms 全由 Graph 导致 | 本机静态 noGraph 221.18、静态 Graph 218.91、动态 noGraph 200.52、动态 Graph 177.61 ms；动态 noGraph 的 copy 基本消失 | 否；动态形状与 Graph 共同作用，且 Graph 收益依赖动态模式 |
| NV 加速比差异完全来自布局 | 目前只有 DCU trace，没有 NV kernel/布局 trace | 未证实；需要同口径 NV trace |

后续应补同配置的多次独立进程 benchmark、对齐动态 eager/compiled 输入的 speedup 测量，以及同口径 NV trace，才能解释剩余跨硬件差距并决定是否修改 Inductor 布局启发式。关闭布局优化后推理 speedup 1.448 仍低于原表 NV 1.850，训练静态 noGraph 1.211 仍低于原表 NV 2.070；这些跨硬件、跨环境数值仅作差距提示，不是同设备修复收益。当前没有源码修改；上面的布局关闭结果是配置实验，不是源码修复后的正式结果。若要做通用源码改动，需对其它卷积模型回归，避免全局关闭 channels-last 使本来受益的模型退化。

## 可复用发现

- 识别 `aten::copy_` 布局方向时，按目标、来源顺序解释 `Input Strides`，不要把 NCHW 与 channels-last 转换方向写反。
- Graph 打开和 max_autotune、动态形状等配置同时变化时，原表不能直接给出 Graph 的单变量收益；需用相同 workload 额外重测。
- 在当前 benchmark 的 `--dynamic-batch-only` 模式下，先检查 eager 与 compiled 是否收到相同 shape；本机 harness 只随机改变 eager 的 batch，输出 speedup 不能直接当同输入比值。
- 动态形状可让 `GraphLowering.decide_layout_opt` 跳过布局优化。此时 copy kernel 减少不能归给 Graph；用动态 noGraph 与静态 Graph 两格区分。
- profiler 中将 GPU kernel 与 CPU `aten::copy_` 按 `External id` 关联，再结合 shape/stride，可以把布局问题量化到具体激活张量。
