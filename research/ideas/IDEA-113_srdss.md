# IDEA-113：Staged Role Disagreement Scale Selector

status: testing
problem: RDSS两seed尺度权重方向一致为负，但联合重训旧12维selector使seed7略低于稳定SNPS。
hypothesis: 冻结每个seed已训练的SNPS top-3 12维选择器，只训练新增尺度系数，可保留稳定父模型并获得跨seed一致增益。
evidence_refs: IDEA-112 seed5/7尺度权重均负；IDEA-106稳定SNPS top-3 checkpoint。
base_commit: f5338971c61d60b05eabd0a7a495999354299a9a
core_change: 加载并冻结SNPS top-3 selector及其特征统计，仅训练一个raw role std系数；输入与推理语义不变。
success_condition: seed5 H大于其SNPS父模型78.466710且scale_weight非零；通过后追加seed7，两个seed均正则supported。
failure_condition: 初始态不能精确复现父模型、H不超过父模型或scale_weight非有限。
experiment: V2-INNOVATION-079
paper_core_innovation: false
