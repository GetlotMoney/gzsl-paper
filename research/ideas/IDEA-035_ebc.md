# IDEA-035：EBC episodic偏置校准

```yaml
idea_id: IDEA-035
source_type: CRA_domain_competition_diagnostic
evidence_refs: [V2-INNOVATION-005, V2-TRY-104]
base_commit: b56ccad712c401aa69cff61f6735d05cca2f507b
problem: CRA的U仍低于S约8.7个百分点，只读seen-bias扫描显示H上界79.803270，但测试gamma不能作为训练创新。
hypothesis: 在三折pseudo-unseen episode中训练单一seen logit扣减gamma，可把CRA的属性判别增益转化为更平衡的U/S并提高H。
core_change: 冻结TG-VPR/TST/NTR/CCGR/CRA，只训练一个范围+-0.2的全局gamma；每折ridge仅用pseudo-seen视觉中心拟合。
success_condition: seed17 H超过79.448210%，U/S任一下降不超过2个百分点，gamma不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过CRA父条件。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-104 / TG-VPR + TST + NTR + CCGR + CRA
current_attempt: V2-TRY-114
last_attempt: V2-TRY-113
last_decision: rescue
```

EBC训练只使用seen类构造pseudo-seen/pseudo-unseen episode；true-unseen图像不进入gamma梯度。只读上界仅作动机，正式结果必须来自训练gamma。

## V2-TRY-113结果

第4轮得到`U=77.081305%`、`S=82.539904%`、`H=79.717270%`、`ZS=86.219549%`，相对CRA H提高`0.269060`且U/S变化均在2个百分点内；但gamma=`0.196359`接近0.2上限，未通过非饱和门槛。补救1将max_gamma收紧到0.15，其他条件不变。
