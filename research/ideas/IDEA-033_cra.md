# IDEA-033：CRA类别中心属性对齐

```yaml
idea_id: IDEA-033
source_type: attribute_alignment_noise_reduction
evidence_refs: [V2-INNOVATION-004, V2-TRY-096]
base_commit: dc072b4e2db142630c1560203e83047a57f1daf1
problem: ARA用7057张图像重复拟合同一类别属性，可能让ridge吸收类内噪声与seen样本频率，而属性监督本质上是类别级。
hypothesis: 用150个seen视觉中心等权拟合属性ridge，可去除类内噪声并提高unseen属性预测与最终H。
core_change: ARA的ridge训练输入从全部seen图像改为每类一个归一化视觉中心；CCGR、属性、beta训练和评估保持不变。
success_condition: seed17 H超过79.386082%，U/S任一下降不超过2个百分点，beta不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过ARA最终父条件。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-096 / TG-VPR + TST + NTR + CCGR + ARA
current_attempt: none
last_attempt: V2-TRY-104
last_decision: run_reliability_seeds
```

CRA的视觉中心和beta训练只使用seen图像；true-unseen图像不进入梯度。该实验只检验ARA的训练统计，不新增论文核心模块。

## V2-TRY-104结果

类别中心ridge在第8轮得到`U=75.319785%`、`S=84.055454%`、`H=79.448210%`、`ZS=86.219549%`，相对CCGR四项全部提高，并超过普通ARA最终最高。learned beta=`10.494793`，未饱和。当前需在父CCGR Gate seed7/27/37上复现后才能标记supported。
