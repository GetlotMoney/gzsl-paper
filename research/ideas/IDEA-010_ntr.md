# IDEA-010：NTR邻域感知切空间路由

```yaml
idea_id: IDEA-010
source_type: local_model_analysis
evidence_refs:
  - V2-INNOVATION-002
  - model/innovations/elpt.py
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: TST gate只接收top-5相似度的均值与最大值，无法区分单一近邻和多个近邻形成的不同语义局部结构。
hypothesis: 输入完整top-5邻域相似度分布，可让gate更准确地决定每个类的切空间迁移步长并提高H。
core_change: gate输入由4维摘要扩展为8维：Mean8-Value余弦、位移、top-5均值及完整5个邻居相似度；其余TST不变。
success_condition: seed7相对TG-VPR+TST的DeltaH不低于0.05个百分点，U和S各自下降不超过2个百分点，并保持TST步长与角位移安全门槛。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: testing
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: V2-TRY-032
```

NTR只改变训练式类别gate的语义几何输入；true-unseen图像不进入梯度。改回4维摘要时严格回到TST。

## V2-TRY-028结果

seed7得到`U=74.319029%`、`S=80.068129%`、`H=77.086536%`、`ZS=81.430238%`，相对TST四项全部提高，`ΔH=+0.101991`；相对原V2 `ΔH=+3.063354`，通过项目seed7目标。结构现已冻结，下一步运行seed5/6/8独立三折训练。

## 四seed诊断

相对TST的seed 5/6/7/8 `ΔH=-0.114005 / +0.055576 / +0.101991 / -0.001984`，只有2/4为正，触发跨seed不稳定。补救1不再用8维gate整体替换TST，而是冻结4维TST gate，只训练由完整top-5邻域驱动、范围±0.1且零初始化的残差步长。
