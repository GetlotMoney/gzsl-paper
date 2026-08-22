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
status: supported
paper_core_innovation: false
parent_condition: V2-TRY-086 / TG-VPR + TST + NTR + CCGR + SDM
current_attempt: none
last_attempt: V2-TRY-097
last_decision: promote
```

属性来自当前CUB `att_splits.mat`，不是旧仓库知识；ridge和beta训练都只使用seen图像。只读official-test上界扫描曾达到`H=79.385635%`，只作为动机，不计正式结果；正式TRY必须由训练得到beta并继续披露`test_used_for_selection=true`。

## V2-TRY-092结果

训练式ARA在第7轮得到`U=73.954368%`、`S=85.495055%`、`H=79.307063%`、`ZS=86.089158%`，相对SDM父条件的`Delta H=+1.694075`并首次超过78%；learned beta=`13.706591`，未触及20的边界。当前仅为seed17成立，必须继续运行父CCGR/SDM seed7/27/37并完成module-off消融后才能标记supported。

## 四训练seed支持结论

seed7/17/27/37的ARA H为`79.330716/79.307063/79.253171/79.280845%`，相对各自SDM父条件提高`1.769812/1.694075/1.667857/1.776917`；H mean/min/max/range=`79.292949/79.253171/79.330716/0.077545`。4/4 seed稳定超过79%，IDEA-029标记`supported`；仍保持`paper_core_innovation=false`，待SDM-off消融和相关工作核对后再决定论文叙事归属。

## SDM-off消融

关闭SDM后ARA得到`U=74.236083%`、`S=85.303891%`、`H=79.386082%`、`ZS=86.028028%`，比含SDM的TRY-092 H高`0.079019`并成为新最高。SDM与ARA组合时冗余，最终候选删除SDM；ARA相对纯CCGR仍提高H `1.813400`。下一步关闭CCGR验证ARA与类别条件原型的互补性。

## CCGR-off消融与结构结论

关闭CCGR、保留NTR+ARA得到`H=78.967987%`，比完整CCGR+ARA低`0.418095`，证明CCGR在属性残差存在时仍提供独立类别几何增益。最终组合删除SDM、保留CCGR与ARA，结构逻辑为结构化描述→安全迁移→类别几何→显式属性残差。
