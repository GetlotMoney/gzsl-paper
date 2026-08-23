# IDEA-116：Cross-Source Disagreement Selector

status: testing
problem: 当前selector分别输入Claude和merge差值，但线性层不能直接表达两来源的绝对矛盾程度。
hypothesis: 增加`abs(Claude pair diff - merge pair diff)`，可提供跨文本源可靠性并提高稳定SNPS H。
evidence_refs: IDEA-103双文本pair特征有效；MAGT显示两来源全局余弦很高但局部差异仍可能影响选择；角色投票轴失败。
base_commit: b5cc054aadb8ce00b01f7ab32a4e178f6dc919e3
core_change: 稳定SNPS 12维输入新增Claude/merge pair差值的绝对差；其余不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、来源分歧退化或U/S任一下降超过2个百分点。
experiment: V2-INNOVATION-082
paper_core_innovation: false
