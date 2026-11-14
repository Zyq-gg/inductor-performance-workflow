# Inductor Model Performance Workflow

这是一个用于分析和优化 PyTorch Inductor 模型性能问题的 Codex skill。适用于用户已经筛出异常的 TorchBench、HuggingFace、timm 等模型，例如编译后比 eager 慢，或相对 NV 的加速幅度偏小。

skill 要求每个模型留一份可追溯报告：先说明模型任务和结构，尽量用原 benchmark 入口测修改前基线，再结合 trace 与单变量实验定位，完成修改后用相同口径测正确性和性能。无法复测或尚无修复时，报告会保留明确的状态与缺口。每个案例得出的共性方法会回写流程文档。

## 安装

将仓库 clone 到 Codex 的 skills 目录：

```bash
git clone https://github.com/Zyq-gg/inductor-performance-workflow.git \
  "${CODEX_HOME:-$HOME/.codex}/skills/inductor-performance-workflow"
```

之后可直接要求 Codex 使用 `$inductor-performance-workflow`，也可让它在相关任务中自动选用。该 skill 不绑定固定的 PyTorch 仓库或 GPU 路径；开始时应读取当前机器的环境和用户给出的数据。

## 内容

| 文件 | 用途 |
|---|---|
| [`SKILL.md`](SKILL.md) | 触发范围和核心工作约束 |
| [`references/workflow.md`](references/workflow.md) | 持续维护的分析、修改、验证流程 |
| [`assets/model-report-template.md`](assets/model-report-template.md) | 单模型报告模板 |
| [`references/benchmark-guide.md`](references/benchmark-guide.md) | TorchBench/HuggingFace/timm 测试入口与结果说明 |
| [`references/inductor-map.md`](references/inductor-map.md) | Inductor 性能分析工具与配置索引；选项以本机版本为准 |
| [`references/mnasnet1_0-case.md`](references/mnasnet1_0-case.md) | 首个案例：HIP Graph 回放异常的逐步定位与待办 |
| [`references/adv_inception_v3-case.md`](references/adv_inception_v3-case.md) | 第二个案例：布局转换、max-autotune 与 Graph 的单变量分析 |
| [`references/alexnet-case.md`](references/alexnet-case.md) | 第三个案例：卷积反向调优候选失败与布局转换的定位、修复和复测 |
| [`references/AlbertForMaskedLM-case.md`](references/AlbertForMaskedLM-case.md) | 第四个案例：无 mask 注意力在追踪中丢失 Flash Attention 的定位与复测 |
| [`scripts/micro_hip_graph_repro.py`](scripts/micro_hip_graph_repro.py) | 该案例的纯 eager 手动 Graph 最小复现 |
| [`scripts/albert_sdpa_mask_overlay/sitecustomize.py`](scripts/albert_sdpa_mask_overlay/sitecustomize.py) | Transformers 5.10.1 无 mask SDPA 路径的进程内诊断修正 |

`assets/benchmark-image-*.png` 是测试说明中的两张示意图。

`mnasnet1_0` 案例中的 `/home/...` 与 `/workspace/...` 是原分析机器上的路径；仓库只收录文档和最小脚本，不包含原始 trace、设备日志或模型权重。案例目前已定位到 Graph 回放路径，但底层根因、源码修复以及正式同口径前后 benchmark 尚未完成。

AlbertForMaskedLM 的覆盖脚本可用 `PYTHONPATH="$PWD/scripts/albert_sdpa_mask_overlay"` 作用于单次 Python 进程。它只处理 `padding_mask is None` 且无局部注意力限制的情况，用于验证编译时冗余 mask 对 Flash Attention 的影响；其余情况调用 Transformers 原函数。具体版本、命令、正确性和性能证据见案例报告。

## 使用示例

```text
使用 $inductor-performance-workflow 分析 TorchBench 模型 <模型名> 的 BF16 推理性能。
原始 CSV 在 <路径>，trace 在 <路径>，PyTorch 源码在 <路径>。
请先复测基线，记录模型报告；定位并修改后复测，将可复用方法更新到流程。
```

## 维护方式

新模型从模板创建报告。把新的可靠测量方法、常见误判和可复用的定位步骤更新到 `references/workflow.md`，并在修订记录中链接对应案例。模型特有的数字、日志与未证实假设留在该模型报告。此仓库是 skill 的维护版本；工作区原文档是首次整理时的副本。提交前在有 `skill-creator` 的环境运行校验：

```bash
python /path/to/skill-creator/scripts/quick_validate.py .
```

维护这个 skill 不等于自动向 PyTorch 源码仓库提交或推送；实际源码修改和仓库操作以当前任务授权为准。
