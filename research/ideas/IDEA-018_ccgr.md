# IDEA-018：CCGR类别条件几何生成

```yaml
idea_id: IDEA-018
source_type: failure_constrained_generation
evidence_refs: [V2-TRY-028, V2-TRY-055, V2-TRY-057]
base_commit: 42cd4457a65f89023ff342ba13679471d5db0942
problem: 共享seen视觉映射会系统性偏置unseen，而低秩seen残差子空间也没有迁移价值；生成方向必须由目标类别自身文本结构限定。
hypothesis: 每类只在Value/local/unique/overall四个文本切向方向内生成有界残差，并由类别几何预测组合和幅度，可改善NTR unseen原型而不引入seen视觉方向偏置。
core_change: 以TG-VPR+TST+NTR为父框架，训练4维类别几何到四方向权重及最大0.1幅度的Gate；训练用seen视觉中心，推理只改写unseen。
success_condition: seed7相对NTR最高H提高至少0.20个百分点，U和S各自下降不超过2个百分点，幅度不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-028 / TG-VPR + TST + NTR
current_attempt: V2-TRY-058
```

CCGR生成方向全部来自目标类别文本；true-unseen图像在训练结束后才加载。关闭CCGR时严格回到NTR父框架。
