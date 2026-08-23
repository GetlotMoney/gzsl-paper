# IDEA-097：Gate-Weighted Pair Selector

status: testing
problem: GPES仅169个硬gate pair，5参数selector对seen过拟合；大量高margin同族top2样本仍可提供证据方向监督。
hypothesis: 纳入全部同族真类top2 seen pair，并用目标soft gate作为pair CE权重，可扩大训练集又保持低margin重点，从而提高跨域选择泛化并超过AGCT。
evidence_refs: AGCT_SOURCE_ORACLE_001；IDEA-096暴露169 pair过拟合。
base_commit: 161c281c2775197ae89638ee9d3a987c62ec5510
core_change: GPES模型和推理不变；pair训练范围扩为全部同族真类top2，并按soft gate归一化加权CE。
success_condition: H大于78.357224，pair数量显著大于169、权重非零且selector有限。
failure_condition: H不超过AGCT、扩大pair后仍过拟合或高margin样本淹没低margin规则。
experiment: V2-INNOVATION-063
interim_result: seed5 H=78.375328、ZS=84.009010，超过AGCT；pair 4041但top1标签占93.17%，追加seed7。
