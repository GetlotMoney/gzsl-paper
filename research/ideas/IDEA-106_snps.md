# IDEA-106：Semantic Neighborhood Pair Selector

status: testing
problem: C-RGWPS只在类名最后词相同的类别间纠错；这一人工族群规则会遗漏名称不同但结构化描述和视觉属性相近的细粒度类别。
hypothesis: 保留C-RGWPS的中心化角色证据，用固定SDCR文本原型top-5语义邻接扩展关系门控，可覆盖额外真实混淆并提高H。
evidence_refs: IDEA-105两seed成立；SDCR_ERROR_AUDIT_001显示错误集中于细粒度邻近类别；类名suffix只是关系近似。
base_commit: e55bf98571dfcef3c90eaad5cc4793d6549c0b77
core_change: pair关系从“同suffix”改为“同suffix或固定语义top-5邻居”；12维selector与训练目标不变。
success_condition: seed5 H大于同seed C-RGWPS 78.393178；正提升后追加seed7。
failure_condition: H不超过C-RGWPS、best退回关闭态或新增关系导致U/S任一下降超过2个百分点。
experiment: V2-INNOVATION-072
paper_core_innovation: false
