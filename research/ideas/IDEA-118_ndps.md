# IDEA-118：Neighborhood Degree Pair Selector

status: rejected
problem: SNPS只把语义图用于二值关系门控，没有告诉selector两个候选类别各自处于多拥挤的语义区域。
hypothesis: 加入top1/top2的log语义邻居度数差，可补充局部竞争密度并提高稳定SNPS H。
evidence_refs: IDEA-106 top-3语义图两seed稳定；SDCR_ERROR_AUDIT_001显示错误集中在细粒度密集类别。
base_commit: f56e141b51add526ab0819608a965c5f039e03f0
core_change: 稳定SNPS 12维输入新增`log1p(degree_top1)-log1p(degree_top2)`；其余不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、度数差退化或U/S任一下降超过2个百分点。
experiment: V2-INNOVATION-084
paper_core_innovation: false
result: seed5/seed7 H=78.478738/78.446100，相对top-3增量+0.012027/+0.000000。
decision: 类别度数差未跨seed产生正增益，拒绝。
