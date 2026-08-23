# IDEA-111：Error-Targeted Pair Correction

status: testing
problem: 当前pair CE在正确top1占约93%的训练集上持续强化已正确pair，pair loss下降但official H在早期峰值后持续恶化。
hypothesis: 正确pair目标修正为0，错误top2只学习刚好消除当前margin的最小负delta，可减少seen过强化并提高稳定SNPS H。
evidence_refs: GWPS/SNPS/TCPS训练日志均在早期达到最佳后随pair loss继续下降而H下降；pair top1 target rate约0.932。
base_commit: bb0c17f63e2a15a9ea72b532b7dd04de93fe62f7
core_change: 稳定SNPS的pair CE替换为minimal_flip_regression；特征、关系图和推理公式不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、best退回关闭态或错误target非有限。
experiment: V2-INNOVATION-077
paper_core_innovation: false
