# IDEA-031：CARA置信度属性残差

```yaml
idea_id: IDEA-031
source_type: supported_auxiliary_refinement
evidence_refs: [V2-INNOVATION-004, V2-TRY-096]
base_commit: bb7f75dc79fba2abda1ea68752a07ae9d6eb488c
problem: ARA使用全局beta，无法区分属性预测可靠与不可靠的图像，可能对unseen样本过度融合属性证据。
hypothesis: 根据属性预测范数、最大分数、top-2间隔和熵预测每图beta残差，可保留seen增益并提高U/H。
core_change: 冻结CCGR、ARA ridge与全局beta，只训练4→16→1置信度Gate输出+-4的样本级beta残差；末层零初始化。
success_condition: seed17 H超过79.386082%，U/S任一下降不超过2个百分点，残差std>0.01。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过ARA父条件。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-096 / TG-VPR + TST + NTR + CCGR + ARA
current_attempt: V2-TRY-102
last_attempt: none
last_decision: none
```

CARA只用seen图像训练置信度Gate；true-unseen图像不进入梯度。该方向只用于继续提高已经可靠的辅助分支，不改变三项论文核心创新。
