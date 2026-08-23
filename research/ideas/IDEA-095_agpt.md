# IDEA-095：Ambiguity-Gated Patch Tie-Breaker

status: rejected
problem: AGCT窄gate有效但文本增益有限；MAGT证明第二文本源与Claude高度重复。局部patch是异质视觉证据，但全局叠加patch会破坏SDCR。
hypothesis: 只在AGCT同族低margin top2内使用局部patch做二选一，可利用异质部位证据改变错误排名，同时避免伤害其他样本，从而超过AGCT。
evidence_refs: IDEA-092固定25分位两seedsupported；IDEA-086证明patch全局推理叠加失败；IDEA-094证明第二文本源重复。
base_commit: b850a3dd2f45d01e510f7c652912f56d94c9b477
core_change: 固定SDCR和25分位gate；将Claude top2分数替换为CCPE top2局部patch分数，只训练范围±5的一个beta。
success_condition: H大于78.357224，U和S任一项下降不超过2个百分点，gate非零且beta不饱和。
failure_condition: H不超过AGCT、patch在歧义子集仍给出错误排序或beta退回0/饱和。
experiment: V2-INNOVATION-061
result: 窄gate内patch beta非零条件均降H，best退回0；局部patch二选一无效。
