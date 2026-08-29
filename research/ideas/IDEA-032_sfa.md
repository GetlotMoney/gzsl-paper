# IDEA-032：SFA语义因子对齐

```yaml
idea_id: IDEA-032
source_type: compressed_description_generalization
evidence_refs: [V2-TRY-101, V2-INNOVATION-003]
base_commit: 371bbd539f12bc1511b8ae37dba93066109261c8
problem: DRA回归完整八角色描述会记住seen类；需要只保留跨类别共享的低维语义变化轴。
hypothesis: 对200类八角色描述做PCA得到64个跨类别语义因子，再训练图像ridge预测因子，可形成无需人工attributes的软属性残差并提高H。
core_change: DRA的6144维角色目标改为由全部类文本构造的64维中心化语义因子；CCGR、ridge、beta和训练seed保持不变。
success_condition: seed17 H达到78.0%，U/S任一下降不超过2个百分点，beta不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过CCGR父条件。
status: rejected
paper_core_innovation: false
parent_condition: V2-TRY-078 / TG-VPR + TST + NTR + CCGR
current_attempt: none
last_attempt: V2-TRY-103
last_decision: drop
```

SFA的因子只由200类允许使用的文本描述构造，ridge和beta只使用seen图像；true-unseen图像不进入梯度。

## V2-TRY-103结果与止损

第1轮beta=`4.42`时official H已降到`76.320452%`，随后稳定在约`73.8–74.0%`，最终只能选择epoch 0的`77.572682%`。低维PCA因子仍重复TG-VPR/CCGR已有描述语义并导致seen过拟合，IDEA-032提前止损。
