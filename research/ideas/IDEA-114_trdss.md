# IDEA-114：Trust-Region Role Disagreement Scale Selector

status: rejected
problem: RDSS从零联合训练可产生最高seed，但跨seed扰动父权重；S-RDSS完全冻结父权重又无法获得增益。
hypothesis: 从SNPS初始化13维selector，并用固定0.1信赖域约束旧12维和偏置，可允许必要协调同时保留父模型稳定性。
evidence_refs: IDEA-112联合训练最高H=78.555039但不稳定；IDEA-113冻结训练best退回父模型。
base_commit: bbd476ab3b88387cce96339d79b477189f2bad80
core_change: SNPS初始化后联合训练13维，但增加0.1×旧权重/偏置L2信赖域；输入与推理公式不变。
success_condition: seed5 H大于SNPS父模型78.466710且旧权重drift有限；通过后追加seed7，两seed均正则supported。
failure_condition: 初始态不复现父模型、H不超过父模型或drift非有限。
experiment: V2-INNOVATION-080
paper_core_innovation: false
result: seed5/seed7 H=78.533653/78.446100，相对各自SNPS父模型+0.066943/+0.000000。
decision: 信赖域仅改善seed5，未跨seed成立，拒绝且不替代稳定SNPS。
