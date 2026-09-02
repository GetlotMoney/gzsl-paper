# IDEA-227 / Hybrid Attribute Evidence Likelihood (HAEL)

- status: `rejected_at_gate1a`
- problem_category: `expert_attribute_candidate_verification`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: `IDEA-226 / AELI`
- performance_status: `proof_of_path_failed`

## 假设

HAEL以global三态属性为主证据，patch为零初始化有界残差，并用三态CE＋固定class-attribute likelihood CE端到端联合训练，试图把IDEA-226的高LLR AUC转成可用的150类属性分类。

- old path: 三态属性检测后事后做class likelihood。
- new path: global+patch tri-state evidence与class likelihood联合优化。
- current advantage: class CE使纯属性分类CBA从5%跃升到26%，但patch贡献不足且unknown监督被分类目标压垮。
- failure boundary: flat三态softmax让“可见性”和“存在/不存在”争夺同一概率质量；class CE倾向消灭unknown。

## 身份

- 最终Idea SHA：`baf416f2f9524eed98172a9ddf4466a9f255ebbccb4c18dbaefe9fc2c5420f0c`；A/B最终复核SHA：`d5c419c7b4894f4f6e5e3175474a475319187b30aef7d6dc2b093cb750b087d` / `382d4371494281a0fbac594bb722a39287e8476e697d9b93ac520667d246a861`，均P0/P1/P2=0。
- 最终运行脚本SHA：`5784401f5d86f1b6cbbba4317bcfe9755cd062b450da916726732449a6db27e8`；runtime修复A/B复核SHA：`0ce78d0fa982742140881fc8e05770c94553752be0884d661c361ccef0f1fb89` / `c070c1bda714d88f919725afd43c18e8e4adaa44149f998698c8aa6bf07d9ba2`。

## 500步OOF结果

结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/hael/IDEA-227-GATE1A/result.json@sha256:f6206e2d79de4286e02d61c9834cb9f231a70d0b93a2d897c01407378c14dc66`；日志SHA：`21ba75c1ab90aaf9352c92b4d8e769e78489dffd8c20528c303f0393bef0a49b`。首次启动在update0前因CPU/GPU类别映射错误停止，保留日志SHA`26e0e2176bad3dc0e8b6798e2cc8ac30dda7f3602327913b70712aa74626f9a0`。

| 条件 | tri-state F1 | unknown/support/refute recall | 150-way CBA | Top17 recall | LLR AUC |
|---|---:|---|---:|---:|---:|
| global joint | 34.30% | 0.92/57.96/85.92% | 26.266% | 79.396% | .624 |
| hybrid | 34.86% | 1.37/60.25/85.00% | 25.358% | 78.830% | .662 |
| hybrid patch-off | 34.86% | 1.40/60.22/84.95% | 25.132% | 78.504% | .667 |

class likelihood CE让global CBA从IDEA-226的5.002%提升到26.266%，证明目标对齐有效；但Hybrid比global低.908pp，patch-off只损失.226pp，虽bootstrap lower=.096pp却未达1pp。所有fold Hybrid均比global低约.8–1.0pp。

unknown recall从tri-only global的19.32%塌到0.92%，说明class CE通过flat三态softmax把不可见属性重新解释成support/refute。Rescue2采用hurdle factorization：先判断visible，再在visible条件下判断present/refute；class CE不得反向推动visibility造假。

披露：纯human attribute路线、无8句、无official/unseen梯度。
