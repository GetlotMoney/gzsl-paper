# IDEA-091：Taxonomic Pairwise Logit Deconvolution

status: rejected
problem: TWLS统一锐化不能改变族内排序，只会放大错误第一名；需要每个类别使用不同的语义近邻组合。
hypothesis: 用SDCR原型相似度构造同族非均匀成对矩阵，对最终logit做保持组均值的高通，可改变错误族内排序并超过SDCR。
evidence_refs: SDCR_ERROR_AUDIT_001；IDEA-090证明统一族内缩放失败。
base_commit: 4b404bcb4c30749745f6d0e46539729f2bf9c26a
core_change: 最终SDCR logits按同族语义相似度矩阵做成对高通，只训练范围[-0.5,0.5]的一个alpha。
success_condition: H大于78.320510，U和S任一项下降不超过2个百分点，alpha不在边界，组均值保持且affinity非均匀。
failure_condition: H不超过SDCR、alpha退回0/饱和或成对高通仍放大错误。
experiment: V2-INNOVATION-057
result: affinity非均匀但正alpha显著降低H，best退回0；固定成对图高通仍放大错误邻接。
