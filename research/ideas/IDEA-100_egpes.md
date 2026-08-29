# IDEA-100：Expanded-Hard Gated Pair Evidence Selector

status: rejected
problem: GPES 25分位硬pair仅169个而过拟合；GWPS 4041个全pair可靠有效；类别平衡补救均过强。
hypothesis: 将训练硬pair门槛扩大到50分位、保持均匀CE，同时推理仍用25分位，可用更多困难pair降低过拟合并超过GWPS。
evidence_refs: IDEA-096小pair过拟合；IDEA-097两seedsupported；IDEA-098/099关闭标签平衡。
base_commit: be08d5833acad19c211c85d54dd8d21c4a5d87ef
core_change: selector与推理不变；训练pair margin门槛从25分位扩大到50分位，不使用soft/class权重。
success_condition: H大于78.414246，pair数量位于169和4041之间且selector有限。
failure_condition: H不超过GWPS、pair仍过少或扩大后引入噪声。
experiment: V2-INNOVATION-066
result: 386 pair得到H=78.367537，高于SDCR但低于GWPS 78.414246；pair范围轴关闭。
