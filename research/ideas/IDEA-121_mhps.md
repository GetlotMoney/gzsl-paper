# IDEA-121：Matched Hard-Pair Selector

status: rejected
problem: SNPS训练pair约93%真类已是top1；逆频率与平方根权重会放大梯度方差，而全量CE持续强化正确pair。
hypothesis: 保留全部top2错误pair并匹配等量最低margin正确pair，可在不引入大权重的情况下聚焦决策边界并提高H。
evidence_refs: GWPS/SNPS pair top1 rate约0.93；B-GWPS/M-BGWPS因高权重失败；GPES小样本过拟合提示需中等样本量。
base_commit: 9bb13d82b348f4f3219ef2949af92b64435b4ed5
core_change: 稳定SNPS全pair训练集改为“全部错误+等量最低margin正确”确定性匹配；特征、loss和推理不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、匹配后标签非1:1或样本少于50。
experiment: V2-INNOVATION-087
paper_core_innovation: false
result: 有效RERUN使用304错误+304困难正确pair，H=78.374042，低于稳定top-3 0.092668。
decision: 等量匹配避免大权重但样本量下降后仍过拟合seen，拒绝。
