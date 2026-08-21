# IDEA-008：SPA seen原型视觉锚定

```yaml
idea_id: IDEA-008
source_type: local_model_analysis
evidence_refs:
  - V2-INNOVATION-002
  - model/tg_vpr_h1/module.py
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: TG-VPR接收seen视觉中心但正式forward未使用该buffer，seen语义原型仍可能偏离真实视觉簇。
hypothesis: 用训练得到的有界强度把seen原型轻微锚向seen视觉中心，可在保持TST unseen原型不变的前提下提高整体H。
core_change: p_seen=Norm((1-lambda)p_TST + lambda v_centroid)，只训练全局lambda，范围(0,0.1)。
success_condition: seed7相对TG-VPR+TST的DeltaH不低于0.05个百分点，U和S各自下降不超过2个百分点，lambda不饱和到上限98%。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: rejected
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: none
last_attempt: V2-TRY-025
last_decision: drop
```

SPA只使用7057张seen训练图像形成视觉中心并训练lambda；TST unseen原型冻结，true-unseen图像不进入梯度。关闭SPA时严格回到TG-VPR+TST。

## V2-TRY-025结果与止损

训练得到`lambda=0.012965`，但`U`下降`4.046488`、`S`提高`3.840590`，最终`H=76.385853%`、相对TST `ΔH=-0.598692`。极小seen锚定已破坏联合竞争，因此IDEA-008提前止损并标记`rejected`，不做强度搜索。
