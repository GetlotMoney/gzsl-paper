# IDEA-042：CNEBC类名后episodic偏置校准

```yaml
idea_id: IDEA-042
source_type: CNRA_domain_balance_diagnostic
evidence_refs: [V2-INNOVATION-009, V2-TRY-138]
base_commit: 1b3d88c220136f6d305b693688bb42b0f0f44201
problem: CNRA提高类名判别后仍存在seen/unseen竞争差距，固定模型额外gamma上界约可提高H 0.10。
hypothesis: 在重建CNRA父路径的pseudo-unseen episode中训练额外seen扣减，可保持类名ZS增益并提高H。
core_change: 冻结CNRA全部路径，只训练范围+-0.10的额外gamma；gamma residual=0严格回到CNRA。
success_condition: seed17 H超过80.712565%，U/S任一下降不超过2个百分点，gamma残差不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过CNRA父条件。
status: rejected
paper_core_innovation: false
parent_condition: V2-TRY-138 / TG-VPR + TST + NTR + CCGR + CRA + VPA + JBEC + CNRA
current_attempt: none
last_attempt: V2-TRY-145
last_decision: drop
```

CNEBC每折正反ridge仅使用pseudo-seen中心，额外gamma梯度只来自seen图像；true-unseen图像不进入训练。该方向只是最终辅助平衡细化。

## V2-TRY-142结果

第3轮得到`U=77.844131%`、`S=84.026349%`、`H=80.817183%`、`ZS=87.423056%`，相对CNRA提高H `0.104618`；gamma残差=`0.049823`未饱和。继续运行父CNRA seed7/27/37可靠性。

## 多seed否定结论

seed7/17/27/37相对CNRA的Delta H为`0.000000/+0.104618/0.000000/+0.005971`。仅seed17有实质提升，2/4 seed直接选回epoch 0，seed37增益可忽略；CNEBC不具可靠性，IDEA-042标记`rejected`。seed17 `80.817183%`只保留为观察。
