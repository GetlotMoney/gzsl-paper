# IDEA-104：Role-aware Gate-Weighted Pair Selector

status: testing
problem: S-GWPS只使用聚合文本和短类名，虽可复现但提升约0.01个百分点；聚合会隐藏颜色、形状、部位等不同语义角色对细粒度鸟类的相反判断。
hypothesis: 保持S-GWPS训练边界与线性选择器不变，额外输入八个角色句各自的top1-top2分数差，可用patch-free文本证据更准确地纠正同族低margin混淆。
evidence_refs: IDEA-103两seed弱正提升；SDCR_ERROR_AUDIT_001同族低margin错误；现有GPT-5.6八句cache。
base_commit: 845ec3dfb76bea0a4a72a1dbb5b57ef81bc8af3e
core_change: 将S-GWPS的4维输入扩展为12维，新增8个角色句差值；其余训练、gate和评估语义不变。
success_condition: seed5 H大于S-GWPS 78.368367且不读取patch；若提升为正再追加seed7。
failure_condition: H不超过S-GWPS、角色特征退化或出现非有限参数。
experiment: V2-INNOVATION-070
paper_core_innovation: false
