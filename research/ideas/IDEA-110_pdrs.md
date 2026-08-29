# IDEA-110：Pair-Discriminative Role Selector

status: rejected
problem: C-RGWPS/SNPS对所有类别对共用相同角色证据尺度，但不同鸟类pair的判别属性并不相同。
hypothesis: 用top1/top2两类在八角色文本上的余弦距离生成pair-specific角色权重，可将图像证据集中到真正区分类别对的角色并提高H。
evidence_refs: IDEA-105中心化角色证据两seed有效；IDEA-106稳定top-3关系图；TCPS说明继续添加全局上下文无稳定收益。
base_commit: ceb348d504d3f1676729f1a93b9ea746f82b6d85
core_change: 八个中心化图像角色差值分别乘以当前类别对的八角色文本距离权重；权重每pair均值归一为1，其余不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、角色权重退化或U/S任一下降超过2个百分点。
experiment: V2-INNOVATION-076
paper_core_innovation: false
result: seed5 H=78.409596，比稳定SNPS top-3低0.057115；pair角色加权扩大特征方差但没有改善H。
decision: 拒绝，不追加seed7或权重幅度补救。
