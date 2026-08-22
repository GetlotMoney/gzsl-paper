# IDEA-023：MFRA元学习视觉残差适配

```yaml
idea_id: IDEA-023
source_type: feature_adapter_meta_generalization
evidence_refs: [V2-TRY-068, V2-TRY-069, V2-INNOVATION-003]
base_commit: 033b0a6af9465bd67d4d7b3435cb65de6e3a04d2
problem: 普通seen CE训练的视觉adapter即使有强一致性和0.1残差边界仍系统性伤害U，说明训练目标没有要求跨类别泛化。
hypothesis: 内层pseudo-seen临时更新、外层pseudo-unseen元损失更新原始adapter，可只保留跨类别泛化的视觉残差并改善CCGR。
core_change: 固定CCGR 0.20原型，训练有界768-64-768 feature adapter；每步一阶内外层更新，一致性权重1.0。
success_condition: seed7相对CCGR最高H提高至少0.20个百分点，U/S任一下降不超过2个百分点，残差不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-067 / TG-VPR + TST + NTR + CCGR
current_attempt: V2-TRY-075
```

MFRA内外层都只使用150个seen类构造的episode；true-unseen图像在训练结束后才加载。
