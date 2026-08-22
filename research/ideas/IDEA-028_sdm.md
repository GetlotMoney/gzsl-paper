# IDEA-028：SDM对称对角度量

```yaml
idea_id: IDEA-028
source_type: representation_metric_reframing
evidence_refs: [V2-TRY-069, V2-TRY-075, V2-TRY-078, V2-TRY-085]
base_commit: 00d6151934e186c979d69bc86a1fa257fdddc42b
problem: 当前余弦距离等权使用768个CLIP维度，且unseen域内错误14.00%；只适配图像会产生seen偏置，继续微调CCGR也已到局部最优。
hypothesis: 对图像和原型同步学习有界正对角度量，可重标定细粒度判别维度，同时避免图像单边适配造成的跨模态漂移。
core_change: 冻结TG-VPR/TST/NTR/CCGR，训练中心化log权重范围+-0.1的768维对角度量；同一权重同时作用于图像和原型，epoch 0严格回到TRY-078。
success_condition: seed17最高H超过77.572682%，U/S任一下降不超过2个百分点，权重不全部顶到边界。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过当前最高结果。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-078 / TG-VPR + TST + NTR + CCGR
current_attempt: none
last_attempt: V2-TRY-086
last_decision: rescue
```

SDM只用150个seen类的三折pseudo-unseen episode训练；true-unseen图像仅用于项目允许的逐epoch选择，不进入梯度。若有效，它作为最终共享度量层连接图像与CCGR原型，不改变三项核心创新逻辑。

## V2-TRY-086结果

第2轮得到`U=74.652368%`、`S=80.818135%`、`H=77.612988%`、`ZS=82.173079%`，相对TRY-078的`Delta H=+0.040306`并成为新最高；ZS提高`0.335044`，说明共享度量确实增强了域内细粒度判别。权重范围`[0.955329,1.021291]`远未触及边界，瓶颈是纯对角度量无法表达维度交互。补救1将保持对称变换和零初始化，增加受控低秩交互。
