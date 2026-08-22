# IDEA-026：CCGR调和Episode目标

```yaml
idea_id: IDEA-026
source_type: metric_objective_mismatch
evidence_refs: [V2-TRY-070, V2-TRY-073, V2-TRY-078, V2-TRY-082]
base_commit: cb162d22e9e5d7803e4c7c01fc3f18ab91b420d8
problem: CCGR训练的平均CE不直接约束pseudo-seen与pseudo-unseen两组同时正确，而最终指标H恰好惩罚任一组偏低。
hypothesis: 在均衡episode中最大化两组软正确率的调和平均，可让CCGR更新优先改善较弱组且避免只做全局logit平移。
core_change: 从TRY-078最佳权重继续训练4维CCGR，增加权重1.0的组间软调和损失；结构、原型公式、数据和推理均不变，并把epoch 0纳入official-test选择。
success_condition: seed17最高H超过77.572682%，优先目标H达到78.0%，U/S任一下降不超过2个百分点。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过当前最高结果。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-078 / TG-VPR + TST + NTR + CCGR
current_attempt: V2-TRY-083
last_attempt: none
last_decision: none
```

训练只在150个seen类的三折pseudo-unseen episode中计算调和目标；true-unseen图像不进入梯度。HEO是CCGR的训练目标增强，不新增论文割裂模块；无提升则不晋级。
