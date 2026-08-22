# IDEA-011：BMR双层元路由

```yaml
idea_id: IDEA-011
source_type: first_principles_training_analysis
evidence_refs:
  - V2-INNOVATION-002
  - V2-TRY-026
  - V2-TRY-027
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: TST把pseudo-seen和pseudo-unseen放在同一联合CE中，PURL又证明简单增加pseudo-unseen权重会过度纠偏，说明两个目标需要参数更新层级上的分离。
hypothesis: 先用pseudo-seen对Gate做临时内层更新，再只用pseudo-unseen外层损失更新原始Gate，可以直接优化“内层未见类”的泛化，同时保留seen竞争约束。
core_change: Gate训练改为一阶双层元更新；原型公式、TST切空间迁移、数据划分和推理路径保持不变。
success_condition: seed7相对TG-VPR+TST的最高H提高至少0.20个百分点，U和S各自下降不超过2个百分点，步长与角位移不越过TST安全门槛。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: testing
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: V2-TRY-040
```

BMR内外层都只使用150个seen训练类构造的episode；true-unseen图像在所有Gate训练结束后才加载。关闭双层更新时回到TST的联合Gate训练。

## V2-TRY-037结果

首次双层更新使`U`相对TST提高`3.296584`、`ZS`提高`1.388496`，但`S`下降`4.392850`，最终`H=76.530440%`、`Delta H=-0.454105`。诊断为外层pseudo-unseen元梯度压过seen保留目标。补救1显式计算seen与outer梯度，冲突时使用PCGrad投影后再合并。

## V2-TRY-038结果

PCGrad条件从第2轮起冲突率为`1.0`，最终步长均值`1.499258`并饱和到上限，`H=63.138895%`。补救2冻结可靠TST Gate，只允许双层元更新学习范围`+-0.1`的残差步长，避免离开TST安全邻域。

## V2-TRY-039-R1结果

修复工程错误后的残差BMR使`U`相对TST提高`1.494062`、`ZS`提高`0.722259`，但`S`下降`1.682091`，`H=76.975867%`、`Delta H=-0.008678`。最后一次补救把固定mod-3 episode改为按文本语义主方向形成的三个连续困难簇，并从头训练对应fold模型。
