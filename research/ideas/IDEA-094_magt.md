# IDEA-094：Multi-Source Ambiguity-Gated Tie-Breaker

status: testing
problem: AGCT窄gate两seed可靠，但只使用Claude且增益很小；OMLR merge文本在旧父链提供更高ZS，可能在同一歧义top2中提供独立纠错。
hypothesis: 固定AGCT 25分位gate，联合学习Claude与merge两个top2系数，可在不扩大受影响样本的前提下超过AGCT。
evidence_refs: IDEA-092固定25分位两seedsupported；IDEA-065 OMLR提供高ZS次级文本信号；全局跨LLM混合失败不代表窄gate内不互补。
base_commit: 99a36a2edec690f322e45f2bfd61e71b5ed49c31
core_change: AGCT gate和SDCR冻结；在Claude top2校正旁新增merge正交top2校正，联合训练两个范围±5的beta。
success_condition: H大于78.357224，U和S任一项下降不超过2个百分点，两个beta至少一个非零且均不饱和。
failure_condition: H不超过AGCT、第二文本源退化为零/复制Claude或双源共同伤害。
experiment: V2-INNOVATION-060
