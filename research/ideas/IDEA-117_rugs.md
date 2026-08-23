# IDEA-117：Role Uncertainty-Gated Selector

status: rejected
problem: RDSS把角色尺度作为线性特征可能改变pair修正方向，导致跨seed不稳定；完全冻结后加性尺度又无增益。
hypothesis: 冻结SNPS方向，仅让非负gamma按角色分歧乘法衰减delta，可把尺度解释为纯不确定性并稳定提高H。
evidence_refs: IDEA-112尺度权重两seed均负；IDEA-113加性冻结失败；IDEA-114联合信赖域未跨seed成立。
base_commit: f757219ac0949214d51b1e90fb4d28aadfcda7be
core_change: 冻结SNPS pair delta，新增`exp(-gamma × raw_std/mean_std)`乘法门控；gamma投影到[0,1]且为唯一训练参数。
success_condition: seed5 H大于SNPS父模型78.466710且gamma>0；通过后追加seed7，两seed均正则supported。
failure_condition: 初始态不复现父模型、H不超过父模型或gamma退回0。
experiment: V2-INNOVATION-083
paper_core_innovation: false
result: RUN-001全程best为SNPS父模型H=78.466710，gamma=0、selected iteration=-1。
decision: seen CE要求放大而非衰减角色分歧，纯不确定性门控假设相反，拒绝。
