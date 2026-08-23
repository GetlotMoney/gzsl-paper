# IDEA-082：Residual Symmetric Diagonal Metric

status: testing
originality_claim: false
problem: SDCR残差余弦仍等权使用768个CLIP维度；语义路由和近邻差分方向均已收口，但维度可靠性尚未在当前无专家父链上训练。
hypothesis: 仅对SDCR残差分支的图像与文本同步施加有界正对角度量，可重标定细粒度判别维度，同时保持跨模态对齐并超过SDCR。
evidence_refs: IDEA-028已证明对称对角度量在旧CCGR父链有稳定微增益；IDEA-075提供当前无专家SDCR父模型。
base_commit: 47f473a57db4ae34752e23e631f5a0d7b9d99b84
core_change: SDCR父结构与beta冻结；只在其残差余弦前增加共享768维正对角变换，训练seen CE。
success_condition: H大于78.320510，U和S任一项下降不超过2个百分点，weight std大于0且权重不全部触边界。
failure_condition: H不超过SDCR、维度权重塌缩或只改善seen而伤害unseen。
experiment: V2-INNOVATION-048
