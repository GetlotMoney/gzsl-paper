# IDEA-017：ORT正交残差迁移

```yaml
idea_id: IDEA-017
source_type: first_principles_transfer_subspace
evidence_refs: [V2-INNOVATION-002, V2-TRY-055]
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: 完整seen视觉映射迁移到unseen会产生域偏置，但seen原型残差中可能存在可重复的低秩共享方向。
hypothesis: 将TST切向方向与seen残差PCA子空间投影进行有界混合，可只保留跨类别共享的迁移分量并提高U/H。
core_change: 每个fold从pseudo-seen原型残差学习rank-32正交子空间，训练一个混合系数调整TST完整切向与子空间投影；mix=0严格回到TST。
success_condition: seed7相对TG-VPR+TST最高H提高至少0.20个百分点，U和S各自下降不超过2个百分点，mix不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: testing
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: V2-TRY-056
```

ORT子空间只由seen或pseudo-seen训练原型残差建立；true-unseen图像在训练结束后才加载。
