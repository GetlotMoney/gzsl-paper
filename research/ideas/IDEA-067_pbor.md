# IDEA-067：Partial Bi-Orthogonal Residual

status: testing
problem: OCLR有效，BOCR完整删除TG父方向则过强；需要检验父方向的小幅正/负调整而非二元保留/删除。
hypothesis: 固定OCLR beta，从OCLR精确起步，只学习TG父方向投影系数lambda∈[-1,1]，可找到比OCLR更优的部分去重程度。
evidence_refs: IDEA-063 OCLR两seed可靠；IDEA-066 BOCR完整去除失败，明确给出过度正交化故障。
base_commit: cfa84702531a3fefd4298caec115c3162efee9b6
core_change: 固定OCLR beta，只训练父原型方向的有界有符号去投影系数；lambda=0严格回到OCLR。
success_condition: H大于OCLR最高78.072185，U和S任一项下降不超过2个百分点，lambda不在±1边界。
failure_condition: H不超过OCLR，或lambda达到98%边界。
experiment: V2-INNOVATION-033
