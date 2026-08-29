# IDEA-122：Focal Boundary Pair Selector

status: rejected
problem: 全量CE被93%容易top1正确pair主导；逆频率权重梯度过强，1:1匹配又把样本缩至608而过拟合。
hypothesis: 在保留全部pair的同时使用gamma=2焦点CE，可平滑降低容易正确pair影响并提高H。
evidence_refs: B-GWPS/M-BGWPS高权重失败；IDEA-121匹配小样本H=78.374042；SNPS全量pair仍最稳。
base_commit: bede006f8ce3c362296c816db95ffc55dba8c096
core_change: 稳定SNPS全量pair CE替换为gamma=2 focal pair CE；特征、采样和推理不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、loss非有限或U/S任一下降超过2个百分点。
experiment: V2-INNOVATION-088
paper_core_innovation: false
result: seed5/seed7 H=78.477298/78.363739，相对top-3增量+0.010588/-0.082361。
decision: 焦点CE未跨seed成立，拒绝并保留普通pair CE。
