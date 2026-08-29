# IDEA-129：CEPS证据一致性选择器

status: rejected
problem: S-EDPS通过屏蔽证据提高两seed H，但masked训练没有直接约束恢复完整证据后的修正一致性。
hypothesis: 对同一seen pair约束完整证据与缺失一个证据的修正一致，可减少证据组合敏感性并进一步提高H。
evidence_refs: IDEA-128的S-EDPS低学习率两seed均提升；EDPS2跨seed不稳定说明仅从零dropout不足。
base_commit: 028ba13caa3316c5d2c1361ef6417defb4b1926f
core_change: 在S-EDPS masked pair CE上新增固定0.1权重的full/masked correction一致性MSE。
success_condition: seed5 H超过同seed S-EDPS 78.572828；通过后追加seed7。
failure_condition: H不超过S-EDPS或best退回SNPS父权重；最多三次方法级补救。
experiment: V2-INNOVATION-096
paper_core_innovation: false
result: consistency_weight=0.1与S-EDPS指标逐项相同，H=78.572828；loss量级不足，RESCUE-1按实测尺度提高到100。
final_result: consistency_weight=100时H=78.528077，低于S-EDPS 0.044751。
decision: 弱约束无效、强约束有害，证明证据缺失视图不应被强制与完整视图相同；提前关闭一致性权重轴。
