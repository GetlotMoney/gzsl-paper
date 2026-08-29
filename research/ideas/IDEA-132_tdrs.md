# IDEA-132：TDRS最具判别性角色选择器

status: rejected
problem: 全角色线性混合对多数unseen可达错误给出错误方向，但类别对通常只由少数角色真正区分。
hypothesis: 用两类文本距离选择最具判别性的单个角色，再训练一个S-EDPS残差系数，可提供目标类别自身的互补方向而不引入seen视觉映射偏置。
evidence_refs: ROLE_DIRECTION_AUDIT_001中max-text-distance role的unseen错误方向正确率43.4%；互补审计显示71个error-role-only和476个correct-role-only样本。
base_commit: 22192efd76adf9d4069245af07087991ccfaf091
core_change: 冻结S-EDPS，只新增top文本距离角色的图像差值和一个标量系数。
success_condition: seed5 H超过S-EDPS 78.572828；通过后追加seed7。
failure_condition: best退回父模型、H下降或role_weight方向与审计矛盾；最多三次方法级补救。
experiment: V2-INNOVATION-099
paper_core_innovation: false
result: best-H与best-ZS均退回S-EDPS父模型；训练最后role_weight=-0.202035，与official unseen审计所需正方向相反。
decision: trainval pair与unseen发生符号翻转，目标类别文本信号不能用普通seen CE学习单一系数；停止非负投影和幅度轴。
