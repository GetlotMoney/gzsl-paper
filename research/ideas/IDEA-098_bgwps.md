# IDEA-098：Balanced Gate-Weighted Pair Selector

status: testing
problem: GWPS两seed可靠，但pair中top1真类占约93%，selector仍偏向保持原预测，可能限制top2纠错上限。
hypothesis: 在soft gate权重上再乘pair标签逆频率，使top1/top2两类总训练质量相等，可提升真正翻转错误pair的能力并超过GWPS。
evidence_refs: IDEA-097两seedsupported；AGCT_SOURCE_ORACLE_001显示大量gated错误可由top2纠正。
base_commit: 4f4522335ad7a442002b2e75877239a820d927ee
core_change: GWPS模型和推理完全不变；训练pair CE新增top1/top2标签逆频率权重并归一化mean=1。
success_condition: H大于78.414246，pair两类总权重平衡且selector有限。
failure_condition: H不超过GWPS、过度偏向top2导致破坏正确pair或U/S失衡。
experiment: V2-INNOVATION-064
