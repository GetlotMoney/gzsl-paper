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
status: testing
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: V2-TRY-019
```

EPC只训练一个有界参数，数据来自150个seen类的三折episode；true-unseen official图像在训练结束后才加载。关闭EPC或边际为0时严格回到TG-VPR+TST。
