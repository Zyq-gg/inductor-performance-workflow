# Inductor 优化map



> 来源：《PyTorch torch compile 和AOTInductor 调试，性能分析等调研》，仅保留性能相关。变量取值以实际 PyTorch 版本为准。
>
>

## 1、性能分析（看哪里慢）

|变量 / 工具|作用|
|---|---|
|`torch.profiler`|预热后采集 CPU/GPU 时间线，导出 trace 到 Perfetto 查看|
|`TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`|kernel 用唯一名字，profiler 里好区分|
|`TORCHINDUCTOR_BENCHMARK_KERNEL=1`|给每个 Triton kernel 生成独立 benchmark 代码|
|`TORCH_COMPILE_DEBUG=1`|保存 FX 图、融合前后 IR、`output_code.py` 到 `torch_compile_debug` 目录|
|`TORCH_LOGS="output_code"`|打印生成的 Triton/C\+\+ 代码|
|`TORCH_TRACE=目录` \+ `tlparse`|记录编译过程，可视化 graph break、重编译、生成代码|
|`INDUCTOR_PROVENANCE=1`|记录 FX 节点到最终 kernel 的映射（配合 `TORCH_TRACE`）|
|`compile_times()`|各编译阶段总耗时|
|`TORCHINDUCTOR_CPP_ENABLE_KERNEL_PROFILE=1`|CPU：记录 C\+\+ kernel 耗时|
|`TORCHINDUCTOR_TRITON_PROTON_PROFILING=1`|用 Triton Proton 做 kernel 内部分析|

## 2、编译选项（怎么优化）

|变量 / 参数|作用|
|---|---|
|`mode="reduce-overhead"`|降低 Python 启动开销，可能用 CUDA Graph|
|`mode="max-autotune"`|最大化性能，编译最久|
|`TORCHINDUCTOR_MAX_AUTOTUNE=1`|尝试更多 kernel 配置（编译变慢）|
|`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1`|只对 GEMM 做 autotune|
|`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`|GEMM 候选后端，如 `ATEN,TRITON,CPP`|
|`TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1`|坐标下降法微调 kernel 参数|
|`TORCHINDUCTOR_BENCHMARK_FUSION=1`|融合前先 benchmark，确认融合有收益|
|`TORCHINDUCTOR_CUDAGRAPHS=1`|开启 CUDA Graph，减少 launch 开销|
|`TORCHINDUCTOR_CUDAGRAPH_OR_ERROR=1`|CUDA Graph 失败时直接报错，不静默回退|
|`TORCHINDUCTOR_CPP_WRAPPER=1`|用 C\+\+ wrapper 代替 Python wrapper|
|`TORCHINDUCTOR_LAYOUT_OPTIMIZATION=1/0`|开/关布局优化，|
|`TORCHINDUCTOR_SHAPE_PADDING=1`|对 shape 做 padding 以提升对齐|
|`TORCHINDUCTOR_FREEZING=1`|冻结参数，CPU 推理推荐|
|`TORCHINDUCTOR_USE_FAST_MATH=1`|快速数学（可能损失精度）|
|`TORCHINDUCTOR_ENABLE_PDL=1`|开启 Programmatic Dependent Launch|

## 3、编译耗时与缓存

|变量|作用|
|---|---|
|`TORCHINDUCTOR_FX_GRAPH_CACHE=1`|本地 FX graph 缓存，加速二次编译|
|`TORCHINDUCTOR_AUTOGRAD_CACHE=1`|AOTAutograd 缓存（依赖上一项）|
|`TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR`|指定缓存目录，方便隔离实验|
|`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`|禁用缓存，测冷编译|
|`TORCHINDUCTOR_COMPILE_THREADS`|编译并行线程数|
|`TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=1`|远程缓存（需 Redis 等）|
|`TORCH_COMPILE_JOB_ID`|标识任务，用于 PGO（动态形状经验）持久化|

### `3.1 TRITON_CACHE_DIR`

看triton 生成实际运行指令，包括ttir,ttgir,amdgcn等信息

## 4、graph break / 重编译 / 动态形状

|变量 / 参数|作用|
|---|---|
|`TORCH_LOGS="graph_breaks,recompiles,guards"`|打印断图位置、重编译原因、guard 失败|
|`torch._dynamo.explain(fn)(*inputs)`|汇总所有 graph break|
|`TORCH_LOGS="dynamic"`|查看动态形状细节|
|`dynamic=True/False/None`|强制动态 / 强制静态 / 自动|
|`@torch.compiler.disable`|跳过无法编译的函数|

常见断图原因：依赖数据的 `if`（改用 `torch.where`）、`.item()` / `.numpy()`。

## 5、AOTInductor 运行时打印（需重新编译）

|变量|作用|
|---|---|
|`AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3`|只打印 kernel 名，定位出错位置|
|`...=2`|打印 kernel 实参张量信息|
|`...=1`|把中间张量保存为 `.pt`|
|`AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT`|只打印指定 kernel（配合 `=2`）|
|`AOTI_RUNTIME_CHECK_INPUTS=1`|运行时检查入口输入的 size/dtype/stride（无需重编译）|

## 6、测量注意

- 先预热再测；分开记录首次编译、缓存命中、稳态耗时。

- 对比时固定输入、精度、warmup、CUDA Graph、缓存状态。

- 以下调试开关会改变性能，**不要用于基准测试**：`CUDA_LAUNCH_BLOCKING`、`AOTI_RUNTIME_CHECK_INPUTS`、`TORCHINDUCTOR_NAN_ASSERTS`、AOTI 中间值打印、各类 `*_ASSERTS`。

- 改代码生成类变量后需新开进程重新编译。

## 7、优化约束

1. 优先确认算子使用指令，确认是否走到慢指令上，分析是inductor还是triton层级导致

2. 分析生成算子生成逻辑是否最优，比如inductor某些配置或代码导致生成问题

3. 快指令用上的基础上还是慢或当前指令无法更新；

3\.1、对比nv确认参数是否不是针对dcu最优，参数分析比如block，num\_warp等进行tuning

3\.2、分析算子语义，可以从图层面考虑语义替换或优化

4. **Inductor优化涉及上下算子融合，对改图或改生成逻辑需要保证性能没有劣化。**
