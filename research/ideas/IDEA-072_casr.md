# IDEA-072：Conservative Adaptive Sentence Routing

status: testing
problem: AOSR seed7 H更高但三个句子权重近零，违反非塌缩门槛；需要在保留路由差异的同时阻止直接删除句子。
hypothesis: 在AOSR seen CE上增加0.1×KL(w||uniform)，可得到min>0.01的非塌缩权重并保持或超过有效AOSR最高H。
evidence_refs: IDEA-071 seed5有效、seed7塌缩；seed7高H证明路由方向有潜力，失败模式明确为权重删除而非无增益。
base_commit: bb5a6a8b5318a29c9fbef445859f6be391d6dc8f
core_change: AOSR模型/父权重/seed7链不变，只在loss中增加固定0.1×KL(w||uniform)。
success_condition: H大于有效AOSR最高78.210580，权重std大于0.01且min大于0.01，U/S任一项下降不超过2个百分点。
failure_condition: H不提高、权重仍塌缩或退化近等权。
experiment: V2-INNOVATION-038
