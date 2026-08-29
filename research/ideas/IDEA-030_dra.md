# IDEA-030：DRA描述残差对齐

```yaml
idea_id: IDEA-030
source_type: attribute_free_generalization
evidence_refs: [V2-INNOVATION-004, V2-TRY-096]
base_commit: f8c8085c4a11a5824103cf1f4576edfbafbae282
problem: ARA稳定超过79%，但依赖CUB专属人工attributes；需要检验同一残差对齐机制能否只依赖项目已有八角色GPT描述。
hypothesis: 用seen图像ridge预测每类八角色描述嵌入，再以训练式beta融合角色平均相似度，可得到不依赖人工attributes的可泛化增益。
core_change: 将ARA的312维属性目标替换为8×768角色描述目标；CCGR父模型、ridge、beta边界、训练seed和评估协议保持不变。
success_condition: seed17最高H达到78.0%，U/S任一下降不超过2个百分点，beta不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过CCGR父条件。
status: rejected
paper_core_innovation: false
parent_condition: V2-TRY-078 / TG-VPR + TST + NTR + CCGR
current_attempt: none
last_attempt: V2-TRY-101
last_decision: drop
```

DRA只使用当前仓库`CUB_gpt55_8role_sentence_embeds.pt`与seen图像；true-unseen图像不进入ridge或beta梯度。若有效，可作为ARA的跨数据集替代；若失败，不影响已支持的ARA结果。

## V2-TRY-101结果与止损

beta从第1轮`5.02`升到第20轮`17.38`，seen训练loss持续下降，但official H从`77.109114%`持续降到`71.199668%`，最终只能选择epoch 0的CCGR父结果`77.572682%`。同一GPT描述已被TG-VPR/CCGR使用，ridge再预测这些描述只造成seen过拟合；IDEA-030提前止损，不做参数补救。
