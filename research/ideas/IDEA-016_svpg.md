# IDEA-016：SVPG语义到视觉原型生成

```yaml
idea_id: IDEA-016
source_type: first_principles_cross_modal_gap
evidence_refs: [V2-INNOVATION-002, V2-TRY-052]
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: TST/NTR主要提高S，unseen U仍约74，说明文本原型与冻结CLIP视觉空间之间仍有跨模态偏差。
hypothesis: 用150个seen类别学习共享语义到视觉残差映射，并应用到全部200类TST原型，可在不读取unseen图像的前提下提高U和ZS。
core_change: 冻结TG-VPR+TST，训练768-128-768共享残差原型生成器；最终层零初始化，关闭时严格回到TST。
success_condition: seed7相对TG-VPR+TST最高H提高至少0.20个百分点，U和S各自下降不超过2个百分点，残差不爆炸。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: testing
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: V2-TRY-053
```

SVPG只用seen图像训练共享映射；unseen原型只通过相同网络前向生成，true-unseen图像在训练结束后才加载。
