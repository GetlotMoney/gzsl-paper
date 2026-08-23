# IDEA-089：Taxonomic Intra-Group Residual

status: testing
problem: SDCR误差审计显示主要错误集中在Warbler、Sparrow、Cormorant等同族内部，且最差unseen类在ZSL空间仍分不开。
hypothesis: 使用类别名最后词建立真实语义族群，对每类增加“SDCR原型减同族中心”的身份方向，可提高同族类内判别并超过SDCR。
evidence_refs: SDCR_ERROR_AUDIT_001；IDEA-043 HGCS已证明直接调整组公共logit无效，本次改为类别身份差分而非组公共量。
base_commit: 60e3f837ae7c7a4cf9d96b11a6c98db94c5d208c
core_change: SDCR全部冻结；按类别名族群构造类内中心化身份原型，只训练范围±5的一个beta。
success_condition: H大于78.320510，U和S任一项下降不超过2个百分点，beta不在±5边界，至少覆盖两个以上族群。
failure_condition: H不超过SDCR、beta退回0/饱和或同族身份方向仍伤害unseen。
experiment: V2-INNOVATION-055
