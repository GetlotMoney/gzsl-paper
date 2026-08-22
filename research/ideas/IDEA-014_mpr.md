# IDEA-014：MPR多角色原型判别

```yaml
idea_id: IDEA-014
source_type: first_principles_multi_prototype
evidence_refs: [V2-INNOVATION-002, V2-TRY-045]
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: 当前每类最终仍压缩为一个原型，图像只要与local/unique/overall中的某一组高度吻合，该局部证据也会在平均时被稀释。
hypothesis: 保留三组完整角色原型，并把最匹配角色相对三组平均的证据加到TST父logit，可提高细粒度类别判别。
core_change: 200类均保留local/unique/overall三个角色原型；训练一个有界强度融合soft-best角色证据，关闭时严格回到TST。
success_condition: seed7相对TG-VPR+TST最高H提高至少0.20个百分点，U和S各自下降不超过2个百分点，融合强度不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: testing
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: V2-TRY-047
```

MPR只使用seen图像训练融合强度；三组unseen角色原型完全来自文本，true-unseen图像在训练结束后才加载。

## V2-TRY-046结果

soft-best多角色证据相对TST仅`Delta H=+0.001218`，但ZS提高`0.040007`且U/S未受损，说明方向存在但信号过小。补救1增加local/unique/overall三个零均值全局可靠性偏置，让模型学习角色相对可信度。
