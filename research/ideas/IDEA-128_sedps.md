# IDEA-128：S-EDPS分阶段证据稳健化

status: testing
problem: EDPS2从零训练时seed5提升、seed7下降，随机起点会覆盖稳定SNPS已经学到的证据方向。
hypothesis: 先固定稳定SNPS为阶段一结果，再用循环证据屏蔽继续训练，可在不破坏父决策的前提下学习证据冗余并提高H。
evidence_refs: IDEA-106的SNPS top-3两seed稳定；IDEA-127的EDPS2 seed5正增益但跨seed不一致；S-RDSS证明分阶段执行边界可行但单一尺度无增益。
base_commit: 08df694bfb1ba520f5bcff24174ffbb6bacf0f1e
core_change: selector从SNPS top-3 checkpoint初始化，再按EDPS循环屏蔽11个非margin证据继续训练；推理公式不变。
success_condition: seed5 H高于78.466710；通过后追加seed7验证。
failure_condition: best退回父模型或H低于父模型，不重复相同分阶段方案。
experiment: V2-INNOVATION-095
paper_core_innovation: false
result: seed5 H=78.547901，相对稳定SNPS提高0.081191；最佳位于第282次更新，追加seed7验证。
