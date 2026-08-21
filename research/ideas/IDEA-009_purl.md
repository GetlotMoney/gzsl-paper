# IDEA-009：PURL pseudo-unseen风险重加权

```yaml
idea_id: IDEA-009
source_type: local_failure_analysis
evidence_refs:
  - V2-INNOVATION-002
  - V2-TRY-025
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: TST的episode虽然32/32采样平衡，但共享CE仍让pseudo-seen与pseudo-unseen各只贡献一次；真实结果显示继续强化seen会严重伤害U。
hypothesis: 对未参与fold H1训练的pseudo-unseen样本额外计算一次分类风险，可让切空间gate更重视可迁移性并提高U与H。
core_change: TST gate loss增加1.0倍pseudo-unseen子批CE；结构、原型公式和推理路径不变。
success_condition: seed7相对TG-VPR+TST的DeltaH不低于0.05个百分点，U和S各自下降不超过2个百分点，并保持TST步长与角位移安全门槛。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: testing
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: V2-TRY-026
```

PURL只重加权150个seen训练类构造的pseudo-unseen子批；true-unseen图像不加载、不进入梯度。额外权重为0时严格回到TST训练目标。
