# IDEA-087：Patch-Guided Sentence Dropout

status: testing
problem: CCPE局部patch在SEBC父链有正信号，但作为SDCR推理残差会破坏平衡；局部证据更适合判断训练样本是否具有可靠部位信息。
hypothesis: 只用train patch真类相对置信度在[0.75,1.25]内重加权SDCR dropout CE，可让句权重更关注局部证据可靠样本，并在patch-free推理时超过SDCR。
evidence_refs: IDEA-049证明top2 patch包含有效局部信号；IDEA-086证明该信号不能直接叠加到SDCR推理。
base_commit: 1562f4ca29d95bdbde344f94a7b2dbbf0f2ad3f3
core_change: SDCR结构和推理完全不变；训练CE按train-only patch可靠性固定加权，只更新八维句权重。
success_condition: H大于78.320510，U和S任一项下降不超过2个百分点，样本权重有非零方差且句权重不塌缩。
failure_condition: H不超过SDCR、样本权重近常数或patch加权仍导致seen偏置。
experiment: V2-INNOVATION-053
