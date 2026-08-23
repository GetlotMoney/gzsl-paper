# IDEA-103：Semantic Gate-Weighted Pair Selector

status: supported
problem: T-GWPS patch-free但略低于AGCT；删除patch后缺少第四个独立差值。短类名是稳定、可追溯且已在父链有效的语义信号。
hypothesis: 在T-GWPS三特征上加入短类名top1-top2差值，可恢复四维选择能力并以patch-free方式超过AGCT。
evidence_refs: IDEA-102 patch-free H=78.352250；IDEA-045类名语义辅助有效；PATCH_CACHE_PROVENANCE_AUDIT_001。
base_commit: a4a1f3d0852c8866fdc35f89e02773ec809bdca7
core_change: T-GWPS训练/推理与线性selector不变，只新增短类名差值作为第四特征。
success_condition: H大于patch-free AGCT 78.357224且不读取patch；两seed可复现后supported。
failure_condition: H不超过AGCT、类名特征重复父margin或selector退回零。
experiment: V2-INNOVATION-069
result: seed5/seed7 H=78.368367/78.350691，两seed均超过各自SDCR父条件及同seed AGCT；patch-free辅助候选成立。
paper_core_innovation: false
decision: 保留为两seed支持的patch-free辅助模块；增益较小，不作为当前论文核心创新。
