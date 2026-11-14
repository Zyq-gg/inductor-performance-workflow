# AlbertForMaskedLM：DCU 编译加速比偏低分析

状态：已定位并完成三配置局部修正复测；更新日期：2026-11-14。遵循[通用流程](./workflow.md)。

## 模型概况

`AlbertForMaskedLM` 是 ALBERT 的掩码语言模型：将被遮盖的 token 预测为词表中的类别。主体包含词嵌入、重复的 Transformer 自注意力与前馈层、LayerNorm，以及词表预测头。ALBERT 使用参数共享，因此同一组注意力与前馈权重可在多个层重复调用。此案例使用 `/home/pytorch/benchmarks/dynamo/huggingface.py` 的包装：从默认 `AlbertConfig()` **随机初始化模型**，而非下载预训练权重；本机默认配置为 12 层、64 个注意力头、hidden size 4096、intermediate size 16384、embedding size 128、词表 30000。输入为 batch 4、序列长 512 的随机 `input_ids` 和 `labels`。推理模式仍传 labels，会计算预测 loss。BF16 推理的主要设备算子是 GEMM、注意力、归一化、激活和词表头。

## 原始问题与指标口径

用户提供的 `/workspace/torchbench数据对比 - dcu-eager-and-nv-speedup.csv` 有三行。表头“compile 耗时（毫秒）”实为编译后稳态 `abs_latency`，不是首次编译时长；未提供 `compilation_latency`。下表 eager 是 `compiled × speedup` 反算近似值。原始 DCU 为 BW1101，NV 为 A800；跨硬件绝对时延不能直接作为编译器改进幅度。

| BF16 推理配置 | DCU eager≈ ms | DCU compiled ms / speedup | NV eager≈ ms | NV compiled ms / speedup | accuracy |
|---|---:|---:|---:|---:|---|
| max-autotune，Graph 开 | 61.956 | 59.630 / 1.039 | 59.127 | 51.459 / 1.149 | 两卡 pass |
| dynamic batch，Graph 开 | 61.962 | 60.628 / 1.022 | 58.440 | 52.365 / 1.116 | 两卡 pass |
| noGraph | 61.983 | 60.471 / 1.025 | 58.948 | 53.154 / 1.109 | 两卡 pass |

三行没有编译后性能劣化，但 DCU 收益只有约 2%–4%，NV 约 11%–15%。三行的 Graph、动态形状与 max-autotune 设置不同，不能直接归因于某一开关；需在本机做同输入单变量对照。动态 batch 的 eager/compiled 输入可能不一致，须核查当前 harness 口径。

## 环境与基线

本机 GPU 为 BW1102，PyTorch 2.13.0 / HIP 7.2.26365、Transformers 5.10.1；PyTorch 源码提交 `ba64d4d7c428a007f69b9f228eeffa05a2f061cd`。源码树已有用户修改 `benchmarks/dynamo/common.py`、未跟踪 `flash_4_configs.p`，另有前一模型分析产生的 `torch/_inductor/kernel/conv.py` 补丁；本案例不覆盖它们。`/workspace/env.sh` 最后一行引用不存在的 MKL 文件，benchmark 使用 `source /opt/dtk/env.sh`，从 `/tmp` 启动绝对路径脚本，以导入已安装的 torch。输出在 `/workspace/perf_results/AlbertForMaskedLM/`。

原始数据的 Transformers 版本未在表中给出，本机版本与测试说明文档中的建议版本不同。后续同机比较仅使用本机同版本结果。

先复测原表 noGraph 基线；accuracy 已通过（1 图、0 graph break，`compilation_latency=28.041 s`）。性能用 `--performance --cold-start-latency --repeat 5 --export-profiler-trace --profile-details`，compiled **61.090 ms**、speedup **1.037**、`compilation_latency=35.978 s`，与原表 BW1101 的 60.471 ms、1.025 接近；原始结果在 `no_graph/`。两次 benchmark 分别运行，profiler 在计时后采集，不把 trace 时长当 CSV 结果。

原表 max-autotune + Graph 也已在 BW1102 复测：accuracy pass（1 图、0 graph break），compiled **59.323 ms**、speedup **1.039**，与原表 BW1101 的 59.630 ms 接近；结果在 `max_auto/`。这次 accuracy 加 `TORCH_LOGS=aot_graphs`，在 `accuracy_aot.log` 保存 AOT 图；性能测试不加该日志开关，仍用 `--repeat 5`。

原表 dynamic + Graph 的本机 accuracy pass（1 图、0 graph break），compiled **59.997 ms**，与原表 BW1101 的 60.628 ms 接近；结果在 `dynamic/`。它的输出 speedup=1.031，但本机 `--dynamic-batch-only` harness 在 eager 计时里随机改变 batch，compiled 保持 batch 4，不是严格同输入比值。与其它模式比较时优先使用 compiled 绝对时延。

为分离开关效果，还补齐同机六格对照；各格 accuracy 都 pass（1 图、0 graph break）。本机不同于原表的 BW1101，只能用本机格子做局部归因。

| 配置 | noGraph compiled ms | Graph compiled ms | 观察 |
|---|---:|---:|---|
| 静态默认 | 61.090 | 59.617 | Graph 约省 1.47 ms |
| 静态 max-autotune | 59.167 | 59.323 | 差距很小 |
| dynamic batch | 60.307 | 59.997 | 差距很小 |

Graph 与 max-autotune 对 60 ms 左右的整体瓶颈影响有限；动态配置的 speedup 因 eager 与 compiled 输入 batch 不完全相同，不用来做精确比值。

```bash
# 工作目录 /tmp，先 source /opt/dtk/env.sh
HF_HUB_OFFLINE=1 python /home/pytorch/benchmarks/dynamo/huggingface.py \
  --accuracy --inference --bfloat16 --backend inductor --device cuda \
  --disable-cudagraphs --only AlbertForMaskedLM --batch-size 4 \
  --output-directory /workspace/perf_results/AlbertForMaskedLM/no_graph \
  --output accuracy.csv
HF_HUB_OFFLINE=1 python /home/pytorch/benchmarks/dynamo/huggingface.py \
  --performance --cold-start-latency --repeat 5 \
  --inference --bfloat16 --backend inductor --device cuda \
  --disable-cudagraphs --only AlbertForMaskedLM --batch-size 4 \
  --export-profiler-trace --profile-details \
  --profiler-trace-name /workspace/perf_results/AlbertForMaskedLM/no_graph/trace \
  --output-directory /workspace/perf_results/AlbertForMaskedLM/no_graph \
  --output perf.csv
```

## 原始 trace 的第一轮定位

`/home/profiler_traces/` 有三份原始 BW1101 对应 trace。它们是在计时后单独采集，以下时长只用于定位。noGraph trace 的 eager `expected` CPU 区段约 65.09 ms，GPU kernel 合计约 61.21 ms；compiled `actual` CPU 区段约 62.17 ms，GPU kernel 合计约 59.80 ms。Graph 配置的 compiled 区段分别约 61.35/61.54 ms，没有秒级回放停顿。

noGraph trace 的 eager 中有 **12 次** `aten::_scaled_dot_product_flash_attention`，对应 12 个 `flash_fwd_kernel`，GPU 合计约 **2.72 ms**。compiled 区段没有 flash attention，而有 24 个额外 GEMM kernel，合计约 **3.64 ms**，及 12 个 `triton_per_fused_*softmax*`，合计约 **6.90 ms**。eager GEMM 约 **42.23 ms**，compiled GEMM 约 **45.81 ms**；其它 eager kernel 约 **18.98 ms**，compiled Triton 约 **13.98 ms**。编译减少了归一化/激活等算子开销，但注意力从 flash 路径变为 GEMM+softmax，抵消大部分收益。这是目前最值得验证的假设；不能仅凭 trace 断定是 Dynamo、AOTAutograd、Inductor 还是模型包装导致 flash 路径丢失。

本机 BW1102 的 noGraph trace 再次证实：eager 有 12 个 `flash_fwd`，compiled 为 0；compiled 有 12 个 Triton softmax 和比 eager 多 24 个 GEMM。这个路径差异不是 BW1101 trace 的单机偶发现象。

`max_auto/accuracy_aot.log` 进一步定位到交给 Inductor 的 AOT 图：原模型调用 `torch.nn.functional.scaled_dot_product_attention`，但 AOT 图已变成 `aten.bmm`、全可见 mask 的 `where`、FP32 `amax/exp/sum/div` 等显式 math 路径。它没有把 flash SDPA 算子交给 Inductor lowering，因此单纯调整 GEMM autotune 不会恢复 flash。这里的图只能确定 math 路径在 Inductor 之前已形成；具体是 Dynamo 追踪时构造 mask，后续 SDPA 分解选择 math。

对照脚本 `/workspace/perf_results/AlbertForMaskedLM/sdpa_micro.py` 用与模型相同的 Q/K/V 形状 `[4,64,512,64]`、stride `[2097152,64,4096,1]`、BF16、scale 0.125、dropout 0；单独的 SDPA 经 `torch.compile` 后仍有一个 `flash_fwd` kernel，说明硬件与形状本身可用 flash。完整模型的区别是 Transformers 的 mask 构造：`AlbertModel.forward` 调 `create_bidirectional_mask`；`masking_utils._ignore_bidirectional_mask_sdpa` 在 eager、padding mask 为 None 时跳过 mask，而 Dynamo tracing 时 `is_tracing()` 为真，未走跳过分支，产生全可见 mask。无 padding、无局部注意力限制时，这张 mask 在语义上冗余，却让编译后的 SDPA 使用 math 路径。

更小的复现 `/workspace/perf_results/AlbertForMaskedLM/mask_sdpa_micro.py` 在相同 Q/K/V 前用 `AlbertConfig()` 的 SDPA 设置调用 `create_bidirectional_mask(attention_mask=None)`。eager 仍用 1 个 flash kernel，约 **0.258 ms**；编译后使用 **2 个 GEMM + 1 个 softmax**，约 **1.363 ms**，accuracy 通过。它将形状、dtype、输入 stride、scale 固定，仅引入 mask 构造，复现了完整模型的路径变化。按相同代码对比，编译单层注意力约慢 **5.3 倍**；12 层重复执行会放大损失，但不能直接将微例差值乘 12 当正式模型时延。

## 局部修正与验证

修正点是 Transformers `masking_utils._ignore_bidirectional_mask_sdpa`：在 `padding_mask is None and local_attention_size is None` 时，无论是否 tracing 都直接返回 `True`，使无约束的双向注意力跳过 mask 构造。其余情况仍调用原函数，保留 padding 与局部注意力行为。目前使用 `/workspace/perf_results/AlbertForMaskedLM/patch_overlay/sitecustomize.py` 做进程内覆盖，避免修改系统安装包；可复用副本在 [`../scripts/albert_sdpa_mask_overlay/sitecustomize.py`](../scripts/albert_sdpa_mask_overlay/sitecustomize.py)。覆盖随 `PYTHONPATH` 启用，不是已提交的 Transformers 上游补丁。

同一最小复现使用独立编译缓存修正后通过 `torch.testing.assert_close`，eager **0.254 ms**、compiled **0.286 ms**；修正前 compiled **1.363 ms**，约降 **79%**。修正后 profiler 在 eager 和 compiled 区段各有一个 `flash_fwd_kernel`，无新增 GEMM/softmax。原 trace 保存在 `mask_sdpa_micro_trace_before.json`，修正后 trace 为 `mask_sdpa_micro_trace.json`。微例证明了路径因果，不代表完整模型会有相同比例收益。

完整模型的三配置复测脚本为 `/workspace/perf_results/AlbertForMaskedLM/run_patched.sh`，分别运行 accuracy 与 performance，并用独立 `TORCHINDUCTOR_CACHE_DIR` 避免旧图复用；结果写到 `patched_no_graph/`、`patched_max_auto/` 和 `patched_dynamic/`。复测保持同一台 BW1102、本机 Transformers 5.10.1、batch 4、BF16 推理和同一 benchmark 入口。改进百分比按本机修改前后的 compiled `abs_latency` 计算，单次正式运行的 `--repeat 5` 结果；由于没有多次独立进程重复测量，不给出置信区间。

| 配置 | 修改前 compiled ms | 修正后 compiled ms | 修正后 accuracy | 修正后算子路径 |
|---|---:|---:|---|---|
| noGraph | 61.090 | **51.843**（约快 15.1%，speedup 1.188） | pass，1 图 0 graph break | compiled 12 个 flash |
| max-autotune + Graph | 59.323 | **51.219**（约快 13.7%，speedup 1.213） | pass，1 图 0 graph break | 按同一无 mask 修正编译 |
| dynamic + Graph | 59.997 | **51.421**（约快 14.3%；报告 speedup 1.199，仅供参考） | pass，1 图 0 graph break | 按同一无 mask 修正编译 |

修正后 noGraph trace `patched_no_graph/trace_AlbertForMaskedLM_rank_0.json` 的 eager 与 compiled 区段各有 **12 个 `flash_fwd_kernel`**；compiled 不再有 12 个注意力 softmax，仍有 2 个词表 loss 所需 softmax。与修改前同机 noGraph trace 中 eager 12、compiled 0 个 flash 对应。compiled `calls_captured` 从 395 降至 386；`compilation_latency` 在 noGraph 中从 35.978 s 到 22.578 s，在 max-autotune + Graph 中从 65.774 s 到 51.066 s，在 dynamic + Graph 中从 34.841 s 到 26.066 s。首次编译时间受缓存与运行条件影响，当前只作为附带观测。

**机制结论：** Transformers 的 `create_bidirectional_mask` 在 eager 的无 padding mask、无局部限制路径返回 `None`，而在 Dynamo tracing 时 `_ignore_bidirectional_mask_sdpa` 的 tracing 守卫阻止了这个静态判断，生成全可见 mask。该 mask 让 SDPA 在编译图里分解为 GEMM + 显式 softmax，丢掉原本可用的 Flash Attention；额外注意力成本抵消了 Inductor 对其他算子的融合收益。只修改这一静态无 mask 分支，最小复现和完整模型均恢复 flash 路径与性能。原 CSV 上 DCU 仅 2%–4% 加速与这个机制吻合；没有原始 NV trace，不能确定 NV 为什么有 11%–15% 加速。

## 证据边界与后续工作

- 本次补丁是进程内覆盖，用于验证根因；尚未提交 Transformers 上游。可把相同条件判断移入其 `masking_utils.py` 并补相关注意力 mask 行为测试。当前 PyTorch 源码中改 Inductor 不能修正这个特定前端行为。
- 实际模型未提供 padding mask；覆盖脚本在 mask 非空或有局部注意力限制时仍调用原函数。该修正的完整语义覆盖范围仅为无 mask、无局部限制的双向 SDPA。对其它模型、训练、不同 Transformers 版本应重新测正确性和性能。
- 没有 NV trace，不能确定 NV 编译路径是否保留 flash，也不能把 DCU/NV 差距全部归于注意力。

## 可复用发现

- 同 shape/dtype 的裸 SDPA 可用 flash 时，应向外逐层加回模型包装，尤其检查 `is_tracing()` 等追踪专用分支是否创建了 eager 不需要的 mask。
