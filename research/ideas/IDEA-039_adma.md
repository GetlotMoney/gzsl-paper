# IDEA-039：ADMA属性对角度量

```yaml
idea_id: IDEA-039
source_type: attribute_space_metric_reframing
evidence_refs: [V2-INNOVATION-005, V2-INNOVATION-008]
base_commit: c4006efa6174d18c966d780bc9ec65d80c2e06d0
problem: CRA属性余弦默认312个属性维度同等可靠，无法突出跨seen/unseen都稳定的细粒度属性。
hypothesis: 对预测属性和类别属性同时学习有界正对角度量，可在不破坏属性空间对齐的前提下提高最终JBEC的类内判别与H。
core_change: 冻结CCGR、CRA ridge、VPA和JBEC标量，只训练312维中心化log权重范围+-0.1的对称属性度量。
success_condition: seed17 H超过80.482768%，U/S任一下降不超过2个百分点，权重std>0.005且不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过JBEC父条件。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-131 / TG-VPR + TST + NTR + CCGR + CRA + VPA + JBEC
current_attempt: V2-TRY-136
last_attempt: none
last_decision: none
```

ADMA只使用seen图像训练属性度量；true-unseen图像不进入梯度。该方向与原型迁移和域校准正交，仍属于辅助度量实验。
