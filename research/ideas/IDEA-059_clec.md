# IDEA-059：Cross-LLM Local Evidence Composition

status: testing
problem: CLRE全局Claude语义达到H=77.808093，CCPE局部GPT语义达到H=77.666533；两者分别成立但尚未证明组合互补。
hypothesis: 固定两套已训练权重，以CLRE全局证据为主并只学习CCPE局部分支±25%的协调比例，可同时利用跨LLM全局语义与类别条件局部patch证据并超过CLRE。
evidence_refs: IDEA-058证明Claude全局证据四项同时提高；IDEA-049证明GPT局部patch证据提高H；两分支输入、文本来源和视觉粒度均不同。
base_commit: 6f45cd92d502220a1a667d5ce0982e449da86e15
core_change: 固定CLRE beta和CCPE beta，联合推理时只训练局部patch分支比例；不重新训练两套父权重。
success_condition: H大于CLRE最高77.808093，U和S任一项下降不超过2个百分点，patch scale不在边界。
failure_condition: H不超过CLRE，或patch scale达到0.75/1.25边界。
experiment: V2-INNOVATION-025
