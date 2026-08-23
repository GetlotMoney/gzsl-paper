# IDEA-099：Mildly Balanced Gate-Weighted Pair Selector

status: testing
problem: B-GWPS完整逆频率把top2类权重提高到7.32并导致H崩到76.75；GWPS不平衡但可靠有效。
hypothesis: 平方根逆频率提供约3.7倍相对补偿，可增强top2纠错而不破坏大多数正确pair，从而超过GWPS。
evidence_refs: IDEA-097两seedsupported；IDEA-098证明完整逆频率过强。
base_commit: 02232cd41bf9ac8ea07bff0d7209ea2d44a27cd4
core_change: GWPS模型/推理不变；pair类别补偿改为平方根逆频率并归一化mean=1。
success_condition: H大于78.414246，top2权重显著低于7.32且selector有限。
failure_condition: H不超过GWPS或仍过度翻转正确pair。
experiment: V2-INNOVATION-065
