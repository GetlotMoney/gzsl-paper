# IDEA-041：CNRA类名残差对齐

```yaml
idea_id: IDEA-041
source_type: independent_class_name_semantics
evidence_refs: [V2-INNOVATION-008, V2-TRY-131]
base_commit: 4bf093a2279143f17d4f10b1a0671cfb2d313f15
problem: TG-VPR使用长视觉描述，标准CLIP类名文本原型未进入最终logits，可能保留独立的类别身份线索。
hypothesis: 在冻结JBEC上训练一个类名CLIP logit残差beta，可提高ZS与H且不依赖人工attributes的新权重。
core_change: 加载200类CLIP类名embedding，只训练范围+-5的单一beta；beta=0严格回到JBEC。
success_condition: seed17 H超过80.482768%，U/S任一下降不超过2个百分点，beta不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过JBEC父条件。
status: supported
paper_core_innovation: false
parent_condition: V2-TRY-131 / TG-VPR + TST + NTR + CCGR + CRA + VPA + JBEC
current_attempt: none
last_attempt: V2-TRY-141
last_decision: promote_auxiliary
```

CNRA的类名embedding可用于全部类别，beta梯度只来自seen图像；true-unseen图像不进入训练。类名CLIP融合已有广泛先例，当前只检验互补性。

## V2-TRY-138结果

第5轮得到`U=77.406234%`、`S=84.313953%`、`H=80.712565%`、`ZS=87.423056%`，相对JBEC四项全部提高，learned beta=`2.707372`未饱和。继续运行父JBEC seed7/27/37可靠性。

## 四训练seed支持结论

seed7/17/27/37的H为`80.288043/80.712565/80.519916/80.530291%`，H mean/min/max/range=`80.512704/80.288043/80.712565/0.424522`；四个seed相对JBEC均提高H `0.082553–0.303164`，beta均未饱和。IDEA-041标记`supported`辅助类名语义分支，不增加论文核心创新数量。
