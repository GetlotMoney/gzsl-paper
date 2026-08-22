# IDEA-005：TST切空间语义迁移

```yaml
idea_id: IDEA-005
source_type: local_failure_analysis
evidence_refs:
  - V2-TRY-006
  - V2-TRY-007
  - V2-TRY-008
  - V2-TRY-009
  - PAPER-005
  - PAPER-006
base_commit: 3dc078c0d52bf358bf24a26e48346c97de9e99ca
problem: ELPT凸混合出现强正指标但迁移系数越过预注册安全范围，限制alpha后又饱和并损害S。
hypothesis: 只沿Mean8单位球面的切向分量迁移，可以保留原型的径向身份并学习足够的类别方向变化。
core_change: 将凸混合改为p=Norm(p_mean + beta × tangent(p_value,p_mean))，beta由三折pseudo-unseen gate训练。
success_condition: seed7相对V2基线DeltaH不低于0.20个百分点、U提高、S下降不超过2个百分点、beta均值位于(0.02,1.45)、beta标准差大于0.01、最大角位移小于45度。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: supported
paper_core_innovation: true
novelty_claim: inductive_target_text_tangent_transport_not_first_hyperspherical_transport
parent_condition: FRAMEWORK-V2 / V2-CONFIRM-001 / RUN-001
current_attempt: none
last_attempt: V2-TRY-018
last_decision: promote
experiment_ref: V2-INNOVATION-002
```

TST复用V2-TRY-006中只由100类图像训练得到的三折H1 checkpoint，但重新训练自己的gate。true-unseen official图像仍只在训练结束后加载。

## V2-TRY-015结果

seed7得到`U=74.225152%`、`S=79.957026%`、`H=76.984545%`、`ZS=81.264132%`，相对V2基线`ΔH=+2.961363`。beta mean/std=`0.780799 / 0.085320`，最大角位移`36.594°`，全部通过预注册门槛。结构现已冻结，下一步为seed 5/6/8各自从头训练对应三折模型和gate，不复用seed7 fold权重。

## 四seed结论

seed 5/6/7/8全部提升H，增量分别为`+2.947878 / +3.019490 / +2.961363 / +3.123876`个百分点。候选H mean=`76.866245%`，超过四seed目标`76.853093%`；min/max/range=`76.657331 / 76.984545 / 0.327213`。IDEA-005标记`supported / paper_core_innovation`并晋级为`V2-INNOVATION-002`。seed7单点`76.984545%`仍比`77.023182%`目标低`0.038637`个百分点，不能宣称单点目标已完成。

新增seed9从头训练得到`H=76.698446%`，相对seed9 TG-VPR基线提高`3.219761`个百分点，继续支持TST跨seed稳定有效；当前TST最高仍为seed7 `76.984545%`。
