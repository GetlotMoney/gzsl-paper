# IDEA-127：Evidence-Dropout Pair Selector

status: rejected
problem: selector可能过度依赖某一个文本证据维度，导致seed变化时权重组合不稳定。
hypothesis: 训练每批屏蔽一个非margin证据、推理恢复完整证据，可学习更鲁棒的多源组合并提高H。
evidence_refs: SDCR训练期句子dropout两seed有效；SNPS seed5/7权重方向相近但局部权重差异存在；静态特征和loss调整已收口。
base_commit: 357860fcc3c17023f6e2bc26076ab2205b1d787e
core_change: 稳定SNPS训练每batch循环屏蔽11个非margin维度之一；推理使用完整12维，loss与数据不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、11维覆盖不均或U/S任一下降超过2个百分点。
experiment: V2-INNOVATION-093
paper_core_innovation: false
result: 两次启动均在训练前因schema特征分发错误失败，没有有效方法结果。
decision: 按连续两次工程失败规则关闭当前实现；不把工程失败写成方法失败。
reimplementation_experiment: V2-INNOVATION-094（集中化schema分发后的首次有效验证）
result: EDPS2 seed5/seed7 H=78.519956/78.399828，相对top-3增量+0.053246/-0.046272。
final_decision: 集中化实现得到有效结果，但证据dropout未跨seed成立，拒绝。
