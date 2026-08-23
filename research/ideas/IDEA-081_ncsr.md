# IDEA-081：Neighborhood-Contrastive Semantic Residual

status: testing
problem: SDCR强化了类内语义证据，但细粒度鸟类最容易在语义近邻之间混淆；继续改变句权重已经止损。
hypothesis: 对每类SDCR原型减去top-5语义近邻均值，并只保留与原型正交的判别方向；seen图像训练一个有界gamma后，可减少近邻混淆并超过SDCR。
evidence_refs: IDEA-075 SDCR是当前可靠父模型；IDEA-079/080表明继续类别句路由会在饱和和关闭态之间摆动，因此改为显式近邻差分方向。
base_commit: a18340587bb8a9a661f3518fe194a6294147768a
core_change: SDCR父原型和beta冻结；新增文本top-5近邻对比正交方向，只训练一个范围±5的gamma。
success_condition: H大于78.320510，U和S任一项下降不超过2个百分点，gamma不在±5边界，正交余弦绝对值小于1e-5。
failure_condition: H不超过SDCR、gamma饱和或近邻差分只改变U/S权衡。
experiment: V2-INNOVATION-047
