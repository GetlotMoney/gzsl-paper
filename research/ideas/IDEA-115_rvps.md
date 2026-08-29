# IDEA-115：Role Vote Pair Selector

status: rejected
problem: 中心化角色向量保留数值方向，但对单个角色异常幅度敏感，且不直接表达八角色多数支持哪一类。
hypothesis: 加入有符号角色多数投票，可提供鲁棒类别方向共识并提高稳定SNPS H。
evidence_refs: IDEA-105中心化角色有效；IDEA-112角色尺度最高seed有效但不稳定；IDEA-114训练边界补救未稳定。
base_commit: 9d760291934e94486d9f66234a19b1a48606358d
core_change: 稳定SNPS 12维输入新增第13维mean(sign(role_top1-role_top2))；其余不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、投票退化或U/S任一下降超过2个百分点。
experiment: V2-INNOVATION-081
paper_core_innovation: false
result: RUN-001 best退回父模型H=78.320510、selected iteration=-1；投票特征有效但无增益。
decision: 多数投票重复角色方向并强化seen偏好，拒绝且不追加seed7。
