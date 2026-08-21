# IDEA-006：EPC分折先验校准

```yaml
idea_id: IDEA-006
source_type: local_result_analysis
evidence_refs:
  - V2-INNOVATION-002
  - V2-TRY-015
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: TST的seed7 S=79.957026显著高于U=74.225152，类别竞争先验仍偏向seen，H距离目标差0.038637个百分点。
hypothesis: 用seen类三折pseudo-unseen任务训练有界竞争边际，可以在不使用true-unseen图像梯度的前提下改善U/S平衡并提高H。
core_change: 在TG-VPR+TST logits上加入一个由pseudo-unseen episode训练的有界标量边际，推理时只加到unseen候选类。
success_condition: seed7相对TG-VPR+TST的DeltaH不低于0.05个百分点，U和S各自下降不超过2个百分点，边际绝对值小于上限的98%。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: rejected
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: none
last_attempt: V2-TRY-020
last_decision: drop
```

EPC只训练一个有界参数，数据来自150个seen类的三折episode；true-unseen official图像在训练结束后才加载。关闭EPC或边际为0时严格回到TG-VPR+TST。

## V2-TRY-019结果

CE训练得到边际`+0.062557`，使`U`提高`0.336754`、`S`下降`0.794458`，最终`H=76.793393%`、相对TST `ΔH=-0.191152`。失败原因是CE目标与最终U/S调和均值错位。补救1保持相同参数和数据边界，只把episode目标改为pseudo-seen/pseudo-unseen软准确率的可微调和均值。

## V2-TRY-020结果与止损

软H目标学到更大的正边际`+0.412576`，真实评估中`U`提高`2.253824`但`S`下降`4.851633`，`H=75.785961%`、`ΔH=-1.198584`。这证明折内先验方向不能可靠迁移到true-unseen；继续限制边际只会退化成参数搜索，因此IDEA-006提前止损并标记`rejected`。
