# IDEA-077：Worst-of-Two Sentence Dropout Routing

status: testing
problem: SDCR随机mask一句的两seed增益可靠但较小；随机采样未必每次训练到当前最脆弱的缺句条件。
hypothesis: 每批随机提出两个不同mask候选，只对CE更大的候选反传，可强化最坏缺句稳健性并超过SDCR，推理仍完整8句。
evidence_refs: IDEA-075 SDCR两seed可靠且mask1优于mask2；IDEA-076一致性蒸馏无效，因此本次直接优化候选中的较难CE。
base_commit: d4fc2c2f8a2a5f5421fb52ae9c000207367c85bf
core_change: SDCR模型、CASR父权重、mask1和KL不变；每批采样两个不同mask并对最大CE反传。
success_condition: H大于SDCR最高78.320510，U和S任一项下降不超过2个百分点，推理权重std/min通过非塌缩门槛。
failure_condition: H不超过SDCR或权重塌缩。
experiment: V2-INNOVATION-043
