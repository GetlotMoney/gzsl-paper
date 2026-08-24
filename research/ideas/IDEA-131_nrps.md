# IDEA-131：NRPS非线性残差Pair选择器

status: testing
problem: S-EDPS剩余可处理top2错误的多数delta方向错误，放大全局幅度反而降低H，线性证据组合无法表达条件交互。
hypothesis: 冻结已稳定的线性selector，仅学习受限零初始化非线性残差，可修正条件性方向错误而不破坏父模型。
evidence_refs: SEDPS_ERROR_AUDIT_001显示top2-related oracle H=87.565486；SEDPS_MARGIN_AUDIT_001显示seen/unseen方向正确仅61/155与110/288，且scale>1均降低H。
base_commit: d9d526b880c25dec5c4112208ad22d6c9fdcc0bf
core_change: 在冻结S-EDPS raw score上增加12→8→1受限非线性残差，只训练新增MLP。
success_condition: seed5 H超过S-EDPS 78.572828；通过后追加seed7。
failure_condition: best退回父模型、H下降或残差饱和；最多三次方法级补救。
experiment: V2-INNOVATION-098
paper_core_innovation: false
