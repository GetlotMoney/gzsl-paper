# IDEA-068：Orthogonal Residual Episodic Recalibration

status: testing
problem: OCLR显著改变了200类logits分布，但当前seen竞争gamma仍是OCLR加入前由SEBC训练的旧值。
hypothesis: 固定OCLR语义与beta，在三个100/50类episode中只学习旧SEBC gamma附近的±0.1残差，可重新平衡OCLR后的U/S并提高H。
evidence_refs: IDEA-063 OCLR两seed可靠；IDEA-047 SEBC episode校准有效；IDEA-067证明继续改变OCLR几何无效，应转训练目标。
base_commit: 73ad43377d6c586970c7d0cac8b7d3f7abf3f5c8
core_change: 固定OCLR全部语义权重，只在旧SEBC gamma附近训练一个episode gamma残差。
success_condition: H大于OCLR最高78.072185，U和S任一项下降不超过2个百分点，gamma残差不饱和。
failure_condition: H不超过OCLR，或残差达到98%边界。
experiment: V2-INNOVATION-034
