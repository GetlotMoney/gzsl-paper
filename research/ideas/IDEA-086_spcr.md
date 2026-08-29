# IDEA-086：Sentence-Patch Complementary Residual

status: rejected
problem: SDCR只使用全局CLIP图像特征；CCPE top2在SEBC父链上提高H，但从未与当前SDCR直接组合。
hypothesis: 固定SDCR后增加小幅每类top-2局部patch证据，可利用局部视觉定位补充全局句子语义并超过SDCR。
evidence_refs: IDEA-049 CCPE top2在SEBC上提高H 0.148151；IDEA-075 SDCR是当前最高可靠父模型；仓库检索未发现二者直接组合。
base_commit: 0467360142fc2994aa8bc47cb7f5f79b96ad437e
core_change: SDCR全部冻结；复用CCPE top2局部证据，只训练范围±5的一个patch beta。
success_condition: H大于78.320510，U和S任一项下降不超过2个百分点，patch beta不在±5边界。
failure_condition: H不超过SDCR、beta退回0/饱和或局部证据破坏SDCR平衡。
experiment: V2-INNOVATION-052
result: patch beta从小幅到边界均降低H，best退回0；局部证据不适合作为SDCR推理残差。
