# IDEA-021：DALN原型密度感知归一化

```yaml
idea_id: IDEA-021
source_type: prototype_crowding_analysis
evidence_refs: [V2-INNOVATION-003, V2-TRY-067, V2-TRY-071]
base_commit: 0c8e162a0f3a5d6839229ee326f886d5e705f7f0
problem: CCGR和margin均已平台化，细粒度类别在最终原型空间中的局部拥挤程度不同，但当前所有类别使用同一logit尺度。
hypothesis: 由最终原型top-1/top-5/top-10密度预测零均值类别尺度，可对拥挤类别做相对温度归一化并改善联合竞争。
core_change: 固定CCGR 0.20原型，训练4-16-1密度Gate输出+-0.1 log尺度；seen平均log尺度强制为0，关闭时回到CCGR。
success_condition: seed7相对CCGR最高H提高至少0.20个百分点，U和S各自下降不超过2个百分点，尺度比例不塌缩。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-067 / TG-VPR + TST + NTR + CCGR
current_attempt: V2-TRY-073
```

DALN密度只由类别原型计算，Gate只用seen图像训练；true-unseen图像在训练结束后才加载。

## V2-TRY-072结果

密度Gate使`U`提高`0.539619`、`ZS`提高`0.370640`，但`S`下降`0.709808`，相对CCGR `Delta H=-0.039483`。密度特征有判别信号，失败来自seen CE偏置；补救1改用三折pseudo-unseen episode训练同一Gate。
