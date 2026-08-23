# IDEA-078：Importance-Aware Sentence Dropout Routing

status: testing
problem: SDCR均匀随机mask让低权重句和高权重句被屏蔽的概率相同，训练预算没有集中处理模型最依赖的语义证据。
hypothesis: 按当前完整句权重采样被mask句，使高依赖句更常缺失，可迫使其余句学习互补证据，并超过SDCR；推理仍使用完整8句。
evidence_refs: IDEA-075 SDCR两seed可靠；IDEA-077表明二选一最坏CE过强且无增益，因此改为概率型温和干预。
base_commit: a70dad04517b1f96621f7a71aa092447aa472e25
core_change: CASR父权重、mask1、CE、KL和推理结构不变；训练期mask从均匀采样改为按当前完整句权重采样。
success_condition: H大于SDCR最高78.320510，U和S任一项下降不超过2个百分点，完整句权重不塌缩。
failure_condition: H不超过SDCR、权重塌缩或mask覆盖严重失衡到有句从未训练。
experiment: V2-INNOVATION-044
