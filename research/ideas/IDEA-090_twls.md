# IDEA-090：Taxonomic Within-Group Logit Sharpening

status: rejected
problem: SDCR错误样本margin远低于正确样本且集中在同族类别；HGCS/TIGR表明改语义方向无效，应直接处理最终族内logit差值。
hypothesis: 保持每个族群平均logit不变，只学习放大或收缩族内差值，可提升细粒度决策margin而不改变组级与seen/unseen竞争。
evidence_refs: SDCR_ERROR_AUDIT_001；IDEA-043与IDEA-089关闭语义空间族群方向。
base_commit: c27dc9c1f9495ac3b8a90bfe5323a6e53124264d
core_change: SDCR与所有原型冻结；在最终logits中按类名族群中心化，只训练范围[-1,1]的一个alpha。
success_condition: H大于78.320510，U和S任一项下降不超过2个百分点，alpha不在边界且组均值数值保持不变。
failure_condition: H不超过SDCR、alpha退回0/饱和或族内缩放放大错误预测。
experiment: V2-INNOVATION-056
result: 正alpha显著降低H，best退回0；统一族内锐化不能改变错误排名，方向关闭。
