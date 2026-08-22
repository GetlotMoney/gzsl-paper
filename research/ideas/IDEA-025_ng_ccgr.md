# IDEA-025：NG-CCGR邻域几何生成

```yaml
idea_id: IDEA-025
source_type: stable_plateau_diagnosis
evidence_refs: [V2-TRY-028, V2-TRY-077, V2-TRY-078, V2-TRY-079, V2-TRY-080]
base_commit: c75198ce6bfb84b8e3eb2cd4fc47fda83572543b
problem: CCGR只使用top-5平均相似度，无法区分一个极强邻居与多个中等邻居；四个训练seed已证明约77.55的平台不是随机性造成。
hypothesis: 保留完整top-5相似度分布，可让类别Gate针对不同混淆结构选择更准确的文本切向方向和幅度。
core_change: CCGR类别输入从4维的三个几何量加top-5均值，改为8维的三个几何量加完整top-5向量；训练目标、父原型、幅度上限和训练seed均不变。
success_condition: seed17的official-test逐epoch最高H超过77.572682%，优先目标H达到78.0%，U/S任一下降不超过2个百分点。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过当前最高结果。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-078 / TG-VPR + TST + NTR + CCGR
current_attempt: V2-TRY-081
last_attempt: none
last_decision: none
```

训练仍只使用seen图像和三折pseudo-unseen；true-unseen图像只用于项目允许的official-test逐epoch选择，不进入梯度。若未超过当前CCGR，不晋级、不建立正式Experiment或HTML图。
