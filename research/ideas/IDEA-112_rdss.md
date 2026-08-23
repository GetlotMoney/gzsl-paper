# IDEA-112：Role Disagreement Scale Selector

status: testing
problem: C-RGWPS把每个样本的八角色差值标准化为单位方差，保留相对方向但丢失原始角色分歧幅度。
hypothesis: 增加归一化前八角色差值的标准差，可提供样本级语义证据可靠性并提高稳定SNPS H。
evidence_refs: IDEA-105中心化角色方向稳定有效；IDEA-109第三类全局上下文无效；IDEA-110 pair文本幅度加权过强。
base_commit: 0dce83c9b8530402b89efb594f69761f66d2d3fa
core_change: 稳定SNPS 12维输入新增第13维raw_role_difference_std；其余不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、尺度特征退化或U/S任一下降超过2个百分点。
experiment: V2-INNOVATION-078
paper_core_innovation: false
interim_result: seed5 U/S/H/ZS四项全升，patch-free H=78.555039；raw role std权重为负，追加seed7。
