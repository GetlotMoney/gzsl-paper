# IDEA-024：CGFG条件高斯视觉特征生成

```yaml
idea_id: IDEA-024
source_type: generative_gzsl_reframing
evidence_refs: [V2-INNOVATION-003, V2-TRY-075]
base_commit: 0c8e162a0f3a5d6839229ee326f886d5e705f7f0
problem: 原型、margin和视觉adapter路线均在77.46附近平台化，根本限制是unseen类没有视觉训练样本参与200类分类器学习。
hypothesis: 用seen类别学习语义到视觉均值生成器，并以seen类内残差为噪声合成unseen视觉特征，可训练real-seen/synthetic-unseen平衡分类器并直接提高U/H。
core_change: CCGR语义原型→有界条件视觉均值；每unseen类生成300个视觉特征；与real-seen平衡训练200类余弦分类器。
success_condition: seed7相对CCGR最高H提高至少0.20个百分点，U/S任一下降不超过2个百分点。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: rejected
paper_core_innovation: false
parent_condition: V2-TRY-067 / TG-VPR + TST + NTR + CCGR
current_attempt: none
last_attempt: V2-TRY-076
last_decision: drop
```

CGFG生成器与分类器只使用seen图像及unseen文本；true-unseen图像在全部训练结束后才加载。

## V2-TRY-076结果与止损

平衡训练分类器仍得到`U=2.046328%`、`S=89.209664%`，synthetic-unseen分布无法逼近真实unseen视觉域；增加合成数量或后验校准没有可靠基础，IDEA-024提前止损。
