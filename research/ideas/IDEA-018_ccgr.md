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
current_attempt: V2-TRY-061
```

CCGR生成方向全部来自目标类别文本；true-unseen图像在训练结束后才加载。关闭CCGR时严格回到NTR父框架。

## V2-TRY-058-R1结果

工程重跑后`H=77.100834%`，相对NTR `Delta H=+0.014298`，成为当前最高观察；但生成幅度降到mean/max=`0.009905/0.011706`，seen中心对齐信号过弱。补救1改用三折pseudo-unseen图像的32/32联合CE直接训练同一类别条件Gate。

## V2-TRY-059结果

episodic CCGR得到`H=77.237120%`，相对NTR提高`0.150584`并成为当前最高；U下降`0.359970`、S提高`0.751114`，最大幅度顶到`0.1`。补救2保持幅度边界，增加`0.25`倍pseudo-unseen子批CE，温和修正U/S偏向。

## V2-TRY-060结果

unseen平衡CCGR得到`U=74.429679%`、`S=80.583262%`、`H=77.384331%`、`ZS=81.815892%`，相对NTR四项全部提高且`Delta H=+0.297795`，达到核心创新增益门槛并成为当前最高。部分类别幅度仍顶到`0.1`，最后一次补救增加`0.01`幅度平方约束，验证提升是否依赖边界饱和。
