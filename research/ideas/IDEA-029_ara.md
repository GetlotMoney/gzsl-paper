# IDEA-029：ARA属性残差对齐

```yaml
idea_id: IDEA-029
source_type: complementary_semantic_evidence
evidence_refs: [V2-TRY-086, V2-TRY-091]
base_commit: f5bb00266ec5e3771775db50ac73c3b42074970c
problem: GPT描述原型与SDM已把H推到77.61，但仍有unseen域内细粒度错误；CUB标准attributes提供与描述提示不同的显式部位语义。
hypothesis: 用seen图像闭式学习CLIP到属性空间的ridge映射，再把属性相似度作为主logit的训练式残差，可提供与TG-VPR/CCGR互补的细粒度证据。
core_change: 冻结TG-VPR/TST/NTR/CCGR/SDM；ridge只用7057张seen训练图像拟合312维属性，单一有界beta再用seen CE训练，推理时与主logit相加。
success_condition: seed17最高H达到78.0%，U/S任一下降不超过2个百分点，beta不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过当前最高结果。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-086 / TG-VPR + TST + NTR + CCGR + SDM
current_attempt: none
last_attempt: V2-TRY-092
last_decision: run_reliability_seeds
```

属性来自当前CUB `att_splits.mat`，不是旧仓库知识；ridge和beta训练都只使用seen图像。只读official-test上界扫描曾达到`H=79.385635%`，只作为动机，不计正式结果；正式TRY必须由训练得到beta并继续披露`test_used_for_selection=true`。

## V2-TRY-092结果

训练式ARA在第7轮得到`U=73.954368%`、`S=85.495055%`、`H=79.307063%`、`ZS=86.089158%`，相对SDM父条件的`Delta H=+1.694075`并首次超过78%；learned beta=`13.706591`，未触及20的边界。当前仅为seed17成立，必须继续运行父CCGR/SDM seed7/27/37并完成module-off消融后才能标记supported。
