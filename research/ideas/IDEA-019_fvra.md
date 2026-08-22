# IDEA-019：FVRA视觉特征残差适配

```yaml
idea_id: IDEA-019
source_type: representation_side_limit_analysis
evidence_refs: [V2-INNOVATION-003, V2-TRY-067]
base_commit: 0c8e162a0f3a5d6839229ee326f886d5e705f7f0
problem: 原型侧CCGR已在0.15至0.20幅度平台化，说明剩余误差部分来自冻结CLIP图像特征与最终语义原型之间的表示偏差。
hypothesis: 在固定CCGR原型下训练零初始化低秩图像特征残差，并约束适配特征保持接近原CLIP方向，可提高seen与unseen共同对齐。
core_change: 冻结TG-VPR/TST/NTR/CCGR，训练768-64-768图像特征残差adapter；一致性权重0.1，关闭时回到CCGR。
success_condition: seed7相对CCGR最高H提高至少0.20个百分点，U和S各自下降不超过2个百分点，视觉残差不爆炸。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-067 / TG-VPR + TST + NTR + CCGR
current_attempt: V2-TRY-069
```

FVRA只使用seen训练图像更新adapter；true-unseen图像在训练结束后才加载，CLIP主干和所有文本原型模块冻结。

## V2-TRY-068结果

seen CE显著下降，但视觉残差mean/max达到`0.496306/0.762622`，`S`提高而`U`下降`19.189996`，发生严重seen过拟合。补救1增加每图残差L2上限`0.1`，并把特征一致性权重提高到`1.0`。
