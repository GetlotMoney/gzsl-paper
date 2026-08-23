# IDEA-083：Full-Semantic Symmetric Diagonal Metric

status: testing
originality_claim: false
problem: RSDM只变换SDCR残差导致三条语义分支尺度失衡；共享度量必须对所有基于余弦的语义分支一致生效。
hypothesis: 对TG主原型、SDRS类名和SDCR残差同时应用同一个有界正对角度量，并保持SEBC偏置不变，可保留组合尺度一致性并超过SDCR。
evidence_refs: IDEA-028支持完整原型链对称度量；IDEA-082证明只变换SDCR残差位置错误。
base_commit: 538d1dae225b17e362d640a07385d60956666e7f
core_change: 相对RSDM，把同一共享度量扩展到TG、SDRS和SDCR三个原型分支；其他权重与偏置全部冻结。
success_condition: H大于78.320510，U和S任一项下降不超过2个百分点，weight std大于0且不全部触边界。
failure_condition: H不超过SDCR、权重塌缩或完整变换仍破坏unseen。
experiment: V2-INNOVATION-049
