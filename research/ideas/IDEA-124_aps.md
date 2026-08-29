# IDEA-124：Antisymmetric Pair Selector

status: rejected
problem: 训练pair始终以父模型top1在前，93%标签为top1；仅去掉bias仍不能消除输入方向不平衡。
hypothesis: 为每个pair加入交换顺序、特征取反、标签翻转的严格镜像，并使用绝对margin门控，可强制方向反对称并提高H。
evidence_refs: IDEA-123零bias略低于top-3；SNPS pair top1 rate约0.93；MHPS小样本与FBPS loss调整均未稳定。
base_commit: 372b8ed729081703a556742433e8f807a176ce66
core_change: 全部pair增加swap-and-negate镜像；bias固定0，训练门控使用abs margin；推理在正margin下与原门控一致。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、增强后标签不平衡或交换等变测试失败。
experiment: V2-INNOVATION-090
paper_core_innovation: false
result: 8916个镜像增强样本、标签严格1:1，但H=78.358011，低于稳定top-3 0.108700。
decision: 严格反对称删除父排序有用信息，拒绝。
