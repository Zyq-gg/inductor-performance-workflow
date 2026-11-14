---
name: inductor-performance-workflow
description: Analyze and optimize user-supplied PyTorch Inductor model performance regressions or weak speedups versus eager or another GPU; maintain per-model evidence, reproducible benchmarks, and a living investigation workflow. Use for TorchBench, HuggingFace, timm, and similar model cases.
---

# Inductor model performance workflow

Use this skill when the user provides a model or results showing a compiled performance problem. Treat the supplied model as a screened case. Work in the user's actual checkout and environment; the paths and measurements in the bundled mnasnet case are historical evidence, not defaults for another machine.

## Working contract

1. Create or update one report for each model using [the report template](assets/model-report-template.md). Explain the model's task, structure, inputs, key operators, and benchmark wrapper before presenting measurements.
2. Read [the workflow](references/workflow.md) for measurement, investigation, change, and verification rules. For TorchBench/HuggingFace/timm harness details, consult [the benchmark guide](references/benchmark-guide.md). Use [the optimization map](references/inductor-map.md) only when choosing a relevant diagnostic or tuning knob; verify options against the installed PyTorch version.
3. Before editing source, rerun the matching benchmark baseline when feasible. Record the exact command, environment, raw results, and failure or reason if it cannot run. Keep user-supplied CSV results, local full-model benchmarks, isolated reproductions, and profiler measurements separate. `abs_latency` is compiled runtime in milliseconds; `compilation_latency` is compile time in seconds.
4. Localize with evidence and one-variable controls. Distinguish compile/capture time from steady replay, CPU delay from device work, and Graph behavior from ordinary eager or compiled execution. Record unsuccessful hypotheses and the smallest reproducible condition without overstating what a reduced case proves.
5. Once a fix point is supported, make a scoped change and verify accuracy plus the same benchmark configuration before and after. Report nearby regressions and material limits. If a low-level defect remains unresolved, record that status and the missing evidence; do not label a workaround as a root-cause fix.
6. After each model, add genuinely reusable lessons to the workflow reference and leave model-specific logs, raw numbers, and conjectures in its report. Edit this skill's core instructions only when a general decision rule changes. Do not commit or publish changes unless the current user request authorizes it.

## Case reference

[mnasnet1_0](references/mnasnet1_0-case.md) illustrates a Graph replay regression, evidence boundaries, and an unfinished baseline/fix status. The included [minimal HIP Graph reproduction](scripts/micro_hip_graph_repro.py) is an optional case artifact; it requires a compatible GPU/PyTorch environment and may trigger a device fault on affected systems.

[AlbertForMaskedLM](references/AlbertForMaskedLM-case.md) illustrates a weak compile speedup caused by Transformers tracing an unnecessary attention mask and selecting math SDPA. The [opt-in process overlay](scripts/albert_sdpa_mask_overlay/sitecustomize.py) demonstrates the scoped correction with an isolated Python process.
