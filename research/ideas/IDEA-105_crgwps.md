# IDEA-105：Centered Role Gate-Weighted Pair Selector

status: supported
problem: R-GWPS的八角色原始差值都携带公共类别身份，直接并列会重复放大seen偏好。
hypothesis: 对每张图的八角色差值做样本内中心化和标准化，删除公共身份与整体幅度，仅保留哪个语义角色相对更支持top1/top2，可形成比S-GWPS更可靠的patch-free纠错证据。
evidence_refs: IDEA-104全部非零条件失败；其八角色特征均值接近且同号，符合公共身份重复；SDCR_ERROR_AUDIT_001。
base_commit: 30682e927626a612ac4bf679725b27b5f4caed0f
core_change: R-GWPS八角色差值在进入selector前按样本减均值并除以标准差；其余公式、训练和评估不变。
success_condition: seed5 H大于S-GWPS 78.368367且不读取patch；正提升后再追加seed7。
failure_condition: H不超过S-GWPS、best退回关闭态或标准化产生非有限值。
experiment: V2-INNOVATION-071
paper_core_innovation: false
result: seed5/seed7 patch-free H=78.393178/78.412709，两seed分别超过同seed S-GWPS 0.024811/0.062018。
decision: 保留为两seed支持的patch-free辅助候选；相关工作检索前不作论文核心原创claim。
