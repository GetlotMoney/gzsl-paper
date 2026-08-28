# IDEA-004：ACGR全类中心化三组路由

```yaml
idea_id: IDEA-004
source_type: local_failure_analysis
evidence_refs:
  - model/frameworks/v2/model.py
  - V2-TRY-010
  - V2-TRY-011
  - V2-TRY-012
base_commit: 3dc078c0d52bf358bf24a26e48346c97de9e99ca
problem: ICGR沿用H1的role_part，而true-unseen类role_part恒为零，动态路由无法直接作用于unseen类别，三次结果的ZS均完全不变。
hypothesis: 将三组原始语义构造成覆盖200类且组均值为零的残差分数，可以保持等权时严格等价于V2，同时让训练式路由直接作用于seen和unseen竞争。
core_change: logits改为V2父logits加0.65倍的全类中心化三组语义残差；图像CLS路由结构仍为768-64-3。
success_condition: seed7相对V2基线DeltaH不低于0.20个百分点，U和S各自下降不超过2个百分点，三组平均权重均不低于0.05。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: rejected
paper_core_innovation: false
parent_condition: FRAMEWORK-V2 / V2-CONFIRM-001 / RUN-001
current_attempt: none
last_attempt: V2-TRY-014
last_decision: drop
```

该候选只使用7057张seen训练图像训练路由gate；CLIP和TG-VPR冻结，true-unseen图像不进入梯度。它是相对ICGR的新forward语义，因此独立编号，不复用失败实验ID。

## V2-TRY-013结果

全类残差使`U`提高`0.818336`、`ZS`提高`0.348657`，证明路由已能直接作用于unseen；但`S`下降`1.149035`，最终`H=73.881786%`、`ΔH=-0.141395`。三组均值均高于0.05，不是权重塌缩。补救1只将固定残差幅度从0.65降到0.25并从头训练；若仍无H提升则停止，不搜索更多幅度。

## V2-TRY-014结果与止损

保守幅度得到`H=73.978963%`、`ΔH=-0.044218`，仍未提升；unique组平均权重降至`0.013238`，出现塌缩。按预先写明的止损条件，不再搜索其他幅度或增加loss，IDEA-004标记`rejected`。
