# IDEA-007：CATA中心对齐切空间适配

```yaml
idea_id: IDEA-007
source_type: local_result_analysis
evidence_refs:
  - V2-INNOVATION-002
  - V2-TRY-015
  - V2-TRY-020
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: TST依靠分类CE间接学习迁移步长，seed7 ZS略降且H距离单点目标仍差0.038637个百分点；EPC证明事后logit校准不能可靠迁移。
hypothesis: 在pseudo-unseen episode中直接对齐迁移原型与视觉中心，可从训练阶段改善切空间方向，而不依赖测试时校准。
core_change: TST gate loss增加0.1倍pseudo-unseen迁移原型与其seen训练图像中心的余弦对齐损失。
success_condition: seed7相对TG-VPR+TST的DeltaH不低于0.05个百分点，U和S各自下降不超过2个百分点，并保持TST步长与角位移安全门槛。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: rejected
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: none
last_attempt: V2-TRY-024
last_decision: drop
```

CATA只在三折pseudo-unseen类上使用视觉中心；这些类属于150个seen训练类。true-unseen图像不加载、不进入梯度。对齐权重为0时严格回到TST训练目标。

## V2-TRY-021结果

一对一余弦对齐得到`H=76.900608%`，相对TST `ΔH=-0.083937`，U/S/ZS均轻微下降。失败原因是只拉近自己的视觉中心，没有显式保持类间判别性。补救1把该项改为50个迁移原型对50个视觉中心的对比分类loss，权重仍固定0.1。

## V2-TRY-022结果

单向中心对比得到`H=77.031094%`，已超过项目seed7目标`77.023182%`；但相对TST `ΔH=+0.046550`，比预注册创新门槛少`0.003450`，因此不直接通过。补救2把原型到中心的单向分类改为双向对比匹配，其他条件不变。

## V2-TRY-023结果

双向对比得到`H=77.009297%`、相对TST `ΔH=+0.024752`，弱于单向条件。最后一次补救回到单向中心对比，训练3个不同初始化的完整gate并平均步长，以降低优化方差；这是CATA最后一次方法预算。

## V2-TRY-024结果与止损

三gate平均得到`H=76.961977%`、相对TST `ΔH=-0.022568`，没有通过。CATA已用完3次方法级补救，IDEA-007标记`rejected`。最佳V2-TRY-022虽达到项目seed7目标，但相对TST增量未过预注册创新门槛，因此只保留为失败条件，不晋级。
