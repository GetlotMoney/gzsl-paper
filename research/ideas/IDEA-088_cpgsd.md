# IDEA-088：Centered Patch-Guided Sentence Dropout

status: rejected
problem: PGSD权重有有效方差，但均值1.164同时放大CE并削弱相对KL，无法分辨相对patch可靠性是否有用。
hypothesis: 把patch置信度中心化并缩放到均值1、范围[0.75,1.25]，可只改变样本相对重要性而保持总loss尺度，从而超过SDCR。
evidence_refs: IDEA-087暴露未中心化权重的loss尺度混杂；IDEA-049证明top2 patch本身含局部信号。
base_commit: da50d8b84aadfa48810fde4d3b663b10933b7ed0
core_change: PGSD其他部分不变，仅把样本权重改为全训练集中心化、最大绝对值归一化的均值1公式。
success_condition: H大于78.320510，U和S任一项下降不超过2个百分点，样本权重mean=1且std大于0，句权重不塌缩。
failure_condition: H不超过SDCR或中心化后仍产生seen偏置。
experiment: V2-INNOVATION-054
result: 样本权重mean=1且有真实方差，但所有条件仍低于父模型；patch可靠性不能改善SDCR句权重，方向关闭。
