# IDEA-005：TST切空间语义迁移

```yaml
idea_id: IDEA-005
source_type: local_failure_analysis
evidence_refs:
  - V2-TRY-006
  - V2-TRY-007
  - V2-TRY-008
  - V2-TRY-009
base_commit: 3dc078c0d52bf358bf24a26e48346c97de9e99ca
problem: ELPT凸混合出现强正指标但迁移系数越过预注册安全范围，限制alpha后又饱和并损害S。
hypothesis: 只沿Mean8单位球面的切向分量迁移，可以保留原型的径向身份并学习足够的类别方向变化。
core_change: 将凸混合改为p=Norm(p_mean + beta × tangent(p_value,p_mean))，beta由三折pseudo-unseen gate训练。
success_condition: seed7相对V2基线DeltaH不低于0.20个百分点、U提高、S下降不超过2个百分点、beta均值位于(0.02,1.45)、beta标准差大于0.01、最大角位移小于45度。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: testing
paper_core_innovation: false
parent_condition: FRAMEWORK-V2 / V2-CONFIRM-001 / RUN-001
current_attempt: V2-TRY-016_to_018
```

TST复用V2-TRY-006中只由100类图像训练得到的三折H1 checkpoint，但重新训练自己的gate。true-unseen official图像仍只在训练结束后加载。

## V2-TRY-015结果

seed7得到`U=74.225152%`、`S=79.957026%`、`H=76.984545%`、`ZS=81.264132%`，相对V2基线`ΔH=+2.961363`。beta mean/std=`0.780799 / 0.085320`，最大角位移`36.594°`，全部通过预注册门槛。结构现已冻结，下一步为seed 5/6/8各自从头训练对应三折模型和gate，不复用seed7 fold权重。
