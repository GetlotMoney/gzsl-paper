# IDEA-054：Patch Consensus-Margin Evidence

status: rejected
problem: CCPE top2均值有效，但一个孤立patch异常高、第二个明显较低时，均值仍可能把背景噪声当成类别证据。
hypothesis: 固定CCPE绝对top2权重，仅学习top1-top2差距的有界残差；若孤立匹配不可靠，训练应学到负权重并进一步提高H。
evidence_refs: IDEA-049证明top2均值有效；IDEA-050说明空间距离不是可靠共识；IDEA-051说明最大匹配会累积孤立噪声，因此使用分数共识而非空间共识。
base_commit: 897b445ce3a6d69e3f49d4da368fe66bdcaac90d
core_change: 在固定CCPE top2均值分支后增加top1-top2差距残差，只训练gap beta；不使用人工属性。
success_condition: H大于CCPE最高77.666533，U和S任一项下降不超过2个百分点，gap beta不饱和。
failure_condition: H不超过CCPE，或gap beta达到98%上限。
experiment: V2-INNOVATION-020
result: gap权重方向多为负但所有非零条件均低于CCPE，best退回gap=0；分数共识不提供可用增益。
