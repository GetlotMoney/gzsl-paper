# IDEA-020：EDC样本条件联合竞争

```yaml
idea_id: IDEA-020
source_type: joint_competition_limit_analysis
evidence_refs: [V2-INNOVATION-003, V2-TRY-067]
base_commit: 0c8e162a0f3a5d6839229ee326f886d5e705f7f0
problem: CCGR原型侧已平台化，固定全局margin上界也只有77.52；不同图像的seen/unseen竞争状态需要不同校正。
hypothesis: 用pseudo-unseen episode训练样本条件Gate，根据两域最大分数、log-sum-exp、熵和差值预测有界unseen logit校正，可改善联合竞争而不改写图像或原型。
core_change: 固定CCGR 0.20原型，训练7-16-1竞争Gate，校正范围+-0.2；关闭时严格回到CCGR。
success_condition: seed7相对CCGR最高H提高至少0.20个百分点，U和S各自下降不超过2个百分点，校正有样本差异且不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: rejected
paper_core_innovation: false
parent_condition: V2-TRY-067 / TG-VPR + TST + NTR + CCGR
current_attempt: none
last_attempt: V2-TRY-071
last_decision: drop
```

EDC只使用seen类构造的pseudo-seen/pseudo-unseen episode训练；true-unseen图像在训练结束后才加载。

## V2-TRY-070结果

校正std=`0.185955`且min/max饱和到`+-0.2`，`U`提高`0.615638`但`S`下降`1.043212`，相对CCGR `Delta H=-0.155490`。补救1把校正范围缩到`+-0.05`，与只读margin诊断的有效量级一致。

## V2-TRY-071结果与止损

校正范围`+-0.05`后，`U`提高`0.206119`、`S`下降`0.486124`，相对CCGR `Delta H=-0.113622`。两种范围均只改变U/S权衡而不能提高H，IDEA-020提前止损。
