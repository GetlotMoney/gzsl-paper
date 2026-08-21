# IDEA-002：共享Value变换迁移到unseen

```yaml
idea_id: IDEA-002
source_type: local_observation
evidence_refs:
  - model/tg_vpr_h1/module.py
  - V2-CONFIRM-001/RUN-001
  - LEGACY-H1-EVIDENCE-001
base_commit: 3dc078c0d52bf358bf24a26e48346c97de9e99ca
problem: H1只适配seen原型，unseen仍保持Mean8，存在语义迁移断层。
hypothesis: 将seen训练得到的共享Value变换直接用于unseen三组语义，可以提高U和H。
core_change: 仅在评估时对unseen原型应用共享Value重参数化，训练权重与seen原型保持不变。
success_condition: H高于74.023182且U提高，S不出现严重下降。
failure_condition: H不提升、U不提升或S下降导致H变差。
status: revised
paper_core_innovation: false
last_attempt: V2-TRY-005
last_decision: promote
experiment_ref: V2-INNOVATION-001
next_training_attempt: V2-TRY-006
```

## TRY-001结果

直接把完整Value变换应用到unseen失败：`U=36.994851%`、`S=88.998413%`、`H=52.264429%`、`ZS=75.146323%`，相对基线`ΔH=-21.758753`个百分点。

修订结论：unseen不能接受完整幅度改写；若继续该方向，只允许尝试受约束的小残差迁移，不再重复全量迁移。

## TRY-002结果

保留90% Mean8、注入10%共享Value原型后：`U=76.842153%`、`S=74.372214%`、`H=75.587012%`、`ZS=83.266521%`，相对基线`ΔH=+1.563830`个百分点。

结论：该条件标记`promote`，停止继续搜索迁移强度；下一步建立正式`V2-INNOVATION-001`验证。当前仍是test-exposed单checkpoint结果，尚未标记`supported`。

## 四seed证明

固定10%条件在当前仓库从头训练的seed 5/6/7/8 checkpoint上全部提升H：`+1.304404 / +1.346984 / +1.563830 / +1.321295`。候选H mean=`75.237222`，相对基线平均`+1.384128`个百分点。

结论：固定10%在4/4 seed有效，但它只在测试时生效，且比例由official test下的快速TRY选择。该结果保留为动机观察，不作为论文核心创新。

## 训练式修订

下一方案改为ELPT：把150个seen类拆成3折pseudo-seen/pseudo-unseen任务，训练类别级迁移gate，由训练权重学习每类迁移强度，不在测试时人工指定10%。
