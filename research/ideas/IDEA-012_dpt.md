# IDEA-012：DPT分布式原型置信度

```yaml
idea_id: IDEA-012
source_type: first_principles_prototype_uncertainty
evidence_refs:
  - V2-INNOVATION-002
  - V2-TRY-037
  - V2-TRY-040
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: 当前200个类别都被当作同样可靠的单位点原型，但八条视觉描述的一致程度不同；单点余弦分类无法表达文本原型不确定性。
hypothesis: 八条描述归一化合向量的长度可以作为vMF式集中度代理；让类别logit按相对集中度缩放，可降低含糊文本原型对联合竞争的错误影响。
core_change: 在TG-VPR+TST原型上增加类别文本集中度置信因子，训练一个有界全局gamma控制其作用强度；gamma关闭时严格回到TST。
success_condition: seed7相对TG-VPR+TST最高H提高至少0.20个百分点，U和S各自下降不超过2个百分点，gamma与置信度比例不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: rejected
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: none
last_attempt: V2-TRY-043
last_decision: drop
```

DPT的类别置信度只来自八条文本描述，gamma只用seen训练图像学习；true-unseen图像在训练结束后才加载。关闭DPT时逐位回到TST原型logits。

## V2-TRY-041-R1结果

工程重跑后全局gamma从`0.05`降到`0.007784`，类别置信范围仅`[0.999173,1.000290]`，U/S/H/ZS与TST逐位一致。补救1改为共享类别置信Gate，输入每类文本描述的合向量长度、句间余弦均值、标准差和最小值，输出有界类别尺度。

## V2-TRY-042结果

自适应Gate把所有类别置信度学成几乎相同的`1.105`，ratio仅`1.000049`，指标仍逐位等于TST。补救2对log尺度做seen均值中心化，强制seen置信度几何平均为1，删除统一放大捷径，只保留类别相对差异。

## V2-TRY-043结果与止损

中心化Gate产生`[0.991933,1.021476]`的类别差异，但相对TST `Delta H=-0.060319`、ZS下降`0.193882`。八描述内部离散度不能可靠预测视觉原型置信度；继续做多分量将与MPR重复，因此IDEA-012提前止损并标记`rejected`。
