# IDEA-226 / Attribute Evidence Likelihood-Ratio Intervention (AELI)

- status: `revised_after_gate1a`
- problem_category: `expert_attribute_candidate_verification`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- performance_status: `proof_of_path`
- data_setting: `human/expert attributes; not attribute-free V5`

## 方法假设

AELI从核心路径删除8句生成文本。S使用200×312人类class attribute分布；V用逐图像MTurk属性的`support/refute/not-visible`三态监督；I只在候选差异属性上形成反对称likelihood ratio，并要求删除注册证据后对应决定减弱。

- old_solution_path: 图像/文本原型或绝对属性兼容度 -> 全类score。
- new_solution_path: class attribute hypotheses -> image-level tri-state evidence -> candidate-difference LLR -> causal deletion。
- principle_difference: 属性从语义向量变成可支持、可反驳、可未知的证据状态。
- non_equivalence_test: 必须胜ARA/dot-product/APN-like/absolute likelihood与删除对照。
- current_advantage: 三态证据与LLR方向可学习，但patch-only和绝对分类尚未成熟。
- failure_boundary: patch-only缺少全局上下文；三态CE未自动校准为150类判别分布。
- paper_level_claim: none before strong-control and causal Gate success。

## Idea审核

- 最终草稿SHA：`2e84e4b16ce93b31451d8ab89a61eaf4168e3c3063c5387eedc66782d81e6266`。
- A/B最终审核SHA：`f4215fff819d04cb69e7559625718cf1ab68d2f278a9fe7be0577c436e8f6c2f` / `862aec327137d206621294f867258a8f3c8693f34c632b108d28fe3203c690f9`，均P0/P1/P2=0并通过。

## Gate0属性资产

- 生成脚本SHA：`7380844d9616218c6e747ee27b9c270b200b702f58791965f9175e28eb1c9113`；Linux资产链A/B复核SHA：`97e9fd7d23a642f5aa753eb35e0a57758bbb934ab771280bad4d240b0d95defd` / `de62cf7ec6902215419e65ed495f6336cc6f69d38456887bffc2ec8733ad794c`，均P0/P1/P2=0。
- 资产：`/data/lby/projects/cv_project/GZSL_Warehouse/assets/v6/aeli/CUB_human_attribute_v1/asset_manifest.json@sha256:314921e4ac1e2ffc539cdad99d099aced4c655147d5149d136da56896667334d`。
- 3,677,856条image×attribute标注全部唯一；训练导出`[7057,312]`。
- train状态数：unknown235,096、support205,680、refute1,623,142、guess-skip137,866。
- 正式Linux projected576/coarse36 manifest=`d096087c.../1d60f9a1...`；全7057行4×4 pooling位级一致，raw image fingerprint=`97a8fd55...`。

## 500步OOF Gate1a

脚本SHA：`aa3c9f987016a8fdae538585f499e561536007b4cda8e1318cbe71e5986004d4`；A/B代码复核SHA：`fe9e67383c8d30bf922a14714fd062d2c5c2cabaebfcf34f96fc7059cfc5a6c1` / `4550af9a50f64033b56f5ab487654df46e45c6aa7b08ec6c0a9dd606349ff0d1`，均P0/P1/P2=0。

结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/aeli/IDEA-226-GATE1A/result.json@sha256:6c98f14b9d08f7d61edaff3c439cd87165194e0ca8ec97cd41b0668b394ca72e`；日志SHA：`f851f716ec9b297207347cd5eab6062d49c32322fa0bab329d7e81bf15b85432`。

| 证据头 | macro-F1 | unknown/support/refute recall | absolute 150-way CBA | LLR AUC (lower) |
|---|---:|---|---:|---:|
| prior | 29.15% | 6.44/10.09/98.17% | 0.667% | 0.000 |
| patch-only | 35.41% | 11.16/65.28/74.58% | 0.738% | 0.726 (0.653) |
| global-only | 39.91% | 19.32/66.84/75.33% | 5.002% | 0.829 (0.773) |

Patch显著胜prior `+6.26pp`，且patch/global的LLR方向均强，证明新三态信号和差异推理可学习；但patch比global低4.50pp、unknown recall未过.20，absolute likelihood分类远低于预注册10.67%。Gate1a失败，不执行原Gate1b。

Rescue1改为global主证据＋patch局部残差，并在同一训练中加入class-attribute likelihood CE，使三态检测与类别判别联合对齐；继续保留纯attribute、无8句、强ARA/APN控制和causal deletion边界。

披露：`human_annotations_used=true`、`expert_attributes_used=true`、`image_level_attribute_annotations_used=true`、`eight_role_sentences_used=false`、`unseen_images_used_for_gradient=false`、`official_eval_loaded=false`。
