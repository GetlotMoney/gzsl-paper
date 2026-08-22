# IDEA-037：VEBC视觉原型偏置校准

```yaml
idea_id: IDEA-037
source_type: complementary_supported_auxiliaries
evidence_refs: [V2-TRY-118, V2-INNOVATION-006]
base_commit: 3ca778fbc09808c564f7c0d86ceb2b4f6fa9e0d8
problem: VPA显著提高ZS但降低U，EBC能提高U且不改变ZS，两者具有正交作用。
hypothesis: 在重建VPA的pseudo-unseen episode中训练seen偏置gamma，可同时保留VPA类内判别和EBC域间平衡，使H超过79.79。
core_change: EBC每折父logit从CRA改为CRA+属性视觉原型VPA；gamma上限0.25，其余episode与边界保持不变。
success_condition: seed17 H超过79.791176%，U/S任一下降不超过2个百分点，gamma不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过EBC最高结果。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-118 / TG-VPR + TST + NTR + CCGR + CRA + VPA
current_attempt: V2-TRY-125..127
last_attempt: V2-TRY-121
last_decision: run_parent_and_combination_reliability
```

VEBC的fold正反ridge只使用pseudo-seen视觉中心，gamma梯度只来自seen图像；true-unseen图像不进入训练。VPA与EBC均为辅助组合，不增加论文核心创新数量。

## V2-TRY-119结果

第6轮得到`U=76.674461%`、`S=84.529251%`、`H=80.410490%`、`ZS=87.125778%`，首次超过80并证明VPA/EBC互补；但gamma=`0.247979`接近0.25上限。补救1保持结构与边界，仅将gamma学习率降到0.0025以细化最优区间。

V2-TRY-120降低学习率后在第18轮仍达到gamma=`0.248014`并复现同一H，说明0.25边界压在有效最优区间。补救2保持细学习率，将max_gamma扩大到0.30，使约0.2425的诊断最优落入边界内部。

V2-TRY-121在第12轮得到`U=77.077311%`、`S=84.184039%`、`H=80.474080%`、`ZS=87.125778%`，gamma=`0.293854`低于0.30的98%门槛，成功条件通过。下一步先补齐VPA父模型seed7/27/37，再逐一训练VEBC可靠性。
