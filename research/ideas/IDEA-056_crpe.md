# IDEA-056：Class-Reliability Patch Evidence

status: rejected
problem: CCPE让200类共享同一个patch beta，但不同类别的六句局部描述中，真正独立于类名身份方向的语义量不同。
hypothesis: 固定CCPE已验证beta，按每类局部文本正交残差强度学习一个小幅类别权重斜率，可降低局部语义弱类别的噪声并提高H。
evidence_refs: IDEA-049证明局部patch证据有效；IDEA-046证明类别语义分歧可用于小幅权重修正；IDEA-055说明fold episode不适合训练局部证据强度。
base_commit: f32b9fe2e083ade82a884dfce9fd09ec085ca33f
core_change: 固定CCPE beta，依据局部文本相对类名的正交残差强度为每类增加一个共享斜率delta；只训练delta。
success_condition: H大于CCPE最高77.666533，U和S任一项下降不超过2个百分点，delta不饱和。
failure_condition: H不超过CCPE，或delta达到98%上限。
experiment: V2-INNOVATION-022
result: 所有非零delta均未超过CCPE，best退回delta=0；正交文本强度不是稳定类别可靠性。
