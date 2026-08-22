# IDEA-027：CCGR局部边界分离

```yaml
idea_id: IDEA-027
source_type: official_error_decomposition
evidence_refs: [V2-TRY-078, V2-TRY-084]
base_commit: 48e9f055a9b5f0f0164b20e4740a0d6c9c5724e0
problem: 当前最佳模型的unseen域内错误为14.00%，明显高于seen域内6.50%；跨域错误两边接近，继续校准只会交换U和S。
hypothesis: 用pseudo-unseen视觉中心对最相似错误类别施加局部角度边界，可专门降低unseen内部细粒度混淆，而不改变推理结构或域偏置。
core_change: 从TRY-078继续训练CCGR，在每折50个pseudo-unseen类上增加权重0.1、余弦margin 0.02的最难负类中心损失；epoch 0保底。
success_condition: seed17最高H超过77.572682%，并优先提高U或ZS；U/S任一下降不超过2个百分点。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过当前最高结果。
status: rejected
paper_core_innovation: false
parent_condition: V2-TRY-078 / TG-VPR + TST + NTR + CCGR
current_attempt: none
last_attempt: V2-TRY-085
last_decision: drop
```

损失只使用150个seen类内部轮换得到的pseudo-unseen图像中心；真实unseen图像不进入梯度。若有效，LBS作为CCGR训练目标的一部分，不新增割裂的推理模块。

## V2-TRY-085结果与止损

epoch 0精确复现`H=77.572682%`，20个非零更新轮次最高仅`77.560640%`，最终选回父模型。与HEO连续失败共同说明：在已按official test选出的CCGR局部最优点上继续追加pseudo-episode loss不能突破平台。IDEA-027提前止损，后续必须换训练阶段或表示结构，不再微调同一checkpoint。
