# IDEA-015：PGO梯度冲突优化

```yaml
idea_id: IDEA-015
source_type: repeated_failure_gradient_analysis
evidence_refs: [V2-TRY-037, V2-TRY-038, V2-TRY-047]
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: 多个候选反复表现为U提高时S下降或相反，说明pseudo-seen与pseudo-unseen目标在Gate参数上存在直接梯度冲突。
hypothesis: 在TST安全残差范围内分别计算seen和pseudo-unseen梯度，冲突时做对称投影再合并，可寻找不破坏任一目标的更新方向。
core_change: 冻结TST Gate，只训练范围+-0.1的残差；Gate更新由seen/unseen双目标对称PCGrad产生，原型公式和推理不变。
success_condition: seed7相对TG-VPR+TST最高H提高至少0.20个百分点，U和S各自下降不超过2个百分点，残差不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: rejected
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: none
last_attempt: V2-TRY-049
last_decision: drop
```

PGO只在150个seen训练类构造的pseudo-seen/pseudo-unseen目标上计算梯度；true-unseen图像在训练结束后才加载。

## V2-TRY-048结果

seen/unseen梯度冲突率全程为`1.0`，但相对TST `Delta H=-0.040093`。补救1在冲突投影前分别把两组梯度归一到相同全局范数，消除较大目标通过梯度幅度继续主导更新的问题。

## V2-TRY-049结果与止损

单位范数PCGrad相对TST `Delta H=+0.025671`，但U下降`0.999606`、S提高`1.250404`，仍只是目标间换位且未过核心门槛。两种PGO均证明冲突率为`1.0`，但简单投影不能产生更优共同方向，IDEA-015提前止损。
