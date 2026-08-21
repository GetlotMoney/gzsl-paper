# IDEA-002：共享Value变换迁移到unseen

```yaml
idea_id: IDEA-002
source_type: local_observation
evidence_refs:
  - model/tg_vpr_h1/module.py
  - V2-CONFIRM-001/RUN-001
  - LEGACY-H1-EVIDENCE-001
base_commit: 3dc078c0d52bf358bf24a26e48346c97de9e99ca
problem: H1只适配seen原型，unseen仍保持Mean8，存在语义迁移断层。
hypothesis: 将seen训练得到的共享Value变换直接用于unseen三组语义，可以提高U和H。
core_change: 仅在评估时对unseen原型应用共享Value重参数化，训练权重与seen原型保持不变。
success_condition: H高于74.023182且U提高，S不出现严重下降。
failure_condition: H不提升、U不提升或S下降导致H变差。
status: proposed
```

