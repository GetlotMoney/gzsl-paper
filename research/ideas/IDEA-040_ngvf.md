# IDEA-040：NGVF归一化几何视觉融合

```yaml
idea_id: IDEA-040
source_type: prototype_geometry_reframing
evidence_refs: [V2-INNOVATION-008, V2-TRY-118]
base_commit: aff1212476c014398c8945c0536d7632895271a2
problem: VPA将属性视觉原型作为logit直接相加，没有保证CCGR与视觉原型的合成方向仍位于单位球面。
hypothesis: 在加性logit与单位球面归一化原型融合之间训练单一eta，可保留属性视觉方向并改善类间几何、ZS和H。
core_change: 冻结完整JBEC，只训练eta=tanh(raw)插值加性VPA与`Norm(p_CCGR+(beta/scale)p_visual)`两条logit。
success_condition: seed17 H超过80.482768%，U/S任一下降不超过2个百分点，eta位于(0,0.98)。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过JBEC父条件。
status: rejected
paper_core_innovation: false
parent_condition: V2-TRY-131 / TG-VPR + TST + NTR + CCGR + CRA + VPA + JBEC
current_attempt: none
last_attempt: V2-TRY-137
last_decision: drop
```

NGVF在三折pseudo-unseen episode中只训练eta；所有视觉中心、ridge和父模型冻结，true-unseen图像不进入梯度。

## V2-TRY-137结果与止损

第14轮得到`H=80.495362%`，但learned eta=`-0.112825`，与“向单位球面归一化融合靠近”的正向假设相反。微小增益来自远离归一化分支的负外推，不能支持NGVF机制；IDEA-040标记`rejected`，不围绕负eta追加实验。
