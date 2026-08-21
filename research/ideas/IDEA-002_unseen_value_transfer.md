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
status: rejected
paper_core_innovation: false
last_attempt: V2-TRY-009
last_decision: drop
experiment_ref: V2-INNOVATION-001
next_training_attempt: none
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

### V2-TRY-006

训练式ELPT得到`U=73.932368%`、`S=79.905742%`、`H=76.803085%`、`ZS=81.145829%`，`ΔH=+2.779903`。但gate alpha mean=`0.582589`超过预注册上限`0.50`，因此不直接通过，按规则进入补救1：限制alpha上限并增加保守约束。

### V2-TRY-007 / 补救1

限制alpha≤0.25并增加0.01保守约束后，`H=76.010388%`、`ΔH=+1.987206`，但S下降`2.071208`个百分点；alpha mean=`0.249867`、std=`0.000038`，gate几乎全部顶在上限。补救1未通过，进入补救2：增加完整top-5邻域几何输入。

### V2-TRY-008 / 补救2

增加完整top-5邻域输入后，指标仍为`H=76.010388%`，alpha mean=`0.249886`、std=`0.000053`，共享gate继续顶在上限。补救2未通过，进入最后一次补救3：三折独立gate并在推理时平均。

### V2-TRY-009 / 补救3与止损

三折独立gate并平均后，`U=78.845793%`、`S=73.371834%`、`H=76.010388%`、`ZS=83.562106%`，相对基线`ΔH=+1.987206`，但`S`下降`2.071208`个百分点，仍超过预注册的2个百分点上限；alpha mean=`0.249664`、std=`0.000126`，三个gate仍共同饱和在0.25附近。

最终结论：ELPT已使用首次TRY和全部3次方法级补救，仍未通过预注册门槛，训练式IDEA-002标记为`rejected`并强制止损。固定10%四seed结果继续作为`test_time_observation`保留，不计为论文核心创新。
