# IDEA-022：EAML pseudo-unseen角度间隔学习

```yaml
idea_id: IDEA-022
source_type: discriminative_boundary_analysis
evidence_refs: [V2-INNOVATION-003, V2-TRY-060, V2-TRY-067]
base_commit: 5e2038ef963c0a55d7c53580bbb1cdf33aa77267
problem: CCGR类别条件方向有效但在幅度0.15至0.20后平台化，说明需要提高方向的判别性，而不是继续扩大移动距离。
hypothesis: 在pseudo-unseen正确类logit上施加训练期角度间隔，可迫使CCGR Gate选择更具类间分离性的文本切向组合，并提高U/ZS/H。
core_change: CCGR结构、幅度0.2和0.25 pseudo-unseen权重保持不变，只在pseudo-unseen辅助CE中对正确类减去0.1 margin；推理无新增模块。
success_condition: seed7相对NTR最高H提高并超过当前CCGR最高77.459931，U/S任一下降不超过2个百分点。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: rejected
paper_core_innovation: false
parent_condition: TG-VPR + TST + NTR / CCGR训练目标增强
current_attempt: none
last_attempt: V2-TRY-074
last_decision: drop
```

EAML只在seen类构造的pseudo-unseen episode中使用角度间隔；true-unseen图像在训练结束后才加载。

## V2-TRY-074结果与止损

EAML得到`H=77.417950%`，低于CCGR 0.20的`77.459931%`。CCGR 0.10/0.15/0.20与EAML固定原型ensemble最高也只有`77.459608%`，没有互补增益；IDEA-022止损。
