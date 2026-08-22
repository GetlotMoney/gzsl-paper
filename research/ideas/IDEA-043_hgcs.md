# IDEA-043：HGCS层级公共语义抑制

```yaml
idea_id: IDEA-043
source_type: class_name_hierarchy_diagnostic
evidence_refs: [V2-INNOVATION-009, V2-TRY-138]
base_commit: 8658e18fa79876d464d495ca9c067f13c4dd0840
problem: 细粒度鸟类共享大量粗粒度类名语义，直接增强组相似度可能掩盖类间差异。
hypothesis: 对类名embedding做20组球面聚类，并学习负组级残差以减去公共模式，可突出组内类别身份并提高H/ZS。
core_change: 冻结CNRA，只训练范围+-10的组级logit beta；20组由全部类名embedding固定球面k-means得到。
success_condition: seed17 H超过80.712565%，U/S任一下降不超过2个百分点，beta位于(-9.8,-0.1)。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过CNRA父条件。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-138 / TG-VPR + TST + NTR + CCGR + CRA + VPA + JBEC + CNRA
current_attempt: V2-TRY-147
last_attempt: V2-TRY-146
last_decision: rescue
```

HGCS聚类允许使用全部类名文本，beta梯度只来自seen图像；true-unseen图像不进入训练。层级标签思想已有先例，本实验只检验公共模式抑制。

## V2-TRY-146结果

seen CE把beta推向正值`0.3–1.25`，所有非零epoch H均低于CNRA，最终选回epoch 0。失败模式与只读负beta上界方向相反；补救1只把beta训练改为三折pseudo-unseen episode，不改变20组聚类或beta范围。
