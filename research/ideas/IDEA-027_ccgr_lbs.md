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
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-078 / TG-VPR + TST + NTR + CCGR
current_attempt: V2-TRY-085
last_attempt: none
last_decision: none
```

损失只使用150个seen类内部轮换得到的pseudo-unseen图像中心；真实unseen图像不进入梯度。若有效，LBS作为CCGR训练目标的一部分，不新增割裂的推理模块。
