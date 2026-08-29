# IDEA-102：Text-Only Gate-Weighted Pair Selector

status: rejected
problem: GWPS最高可靠但依赖来源不完整的patch，限制论文主结果资格；NPS证明增加容量无收益。
hypothesis: 去除patch差值，只用parent margin、Claude差和merge差训练三维共享selector，可保留主要提升并获得patch-free推理。
evidence_refs: IDEA-097两seedsupported但patch provenance不完整；IDEA-101关闭容量轴。
base_commit: 08d74dc2499e436afdc4c11fbe5b4e2d05270548
core_change: GWPS训练pair、soft gate与线性结构不变；删除patch特征及所有patch文件读取，selector从4维变3维。
success_condition: H高于AGCT 78.357224且两seed可复现；若达到GWPS 78.414246更佳。
failure_condition: H不超过AGCT、三文本特征无法迁移或selector退回零。
experiment: V2-INNOVATION-068
result: 正确4041-pair RERUN H=78.352250，高于SDCR但低于AGCT；patch-free次级对照保留，不晋级。
