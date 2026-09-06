# IDEA-199：Zero-Crop Semantic-Visual Risk Arbitration（SVRA）

idea_id: IDEA-199
status: revised_gate0_supported_but_fixed_official_diagnostic_failed
base_framework: FRAMEWORK-V6-DEVELOPMENT
source_code_parent: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
metric_parent: frozen CLIP ViT-L/14@336 class-name Parent, H=66.69292303042761
predecessor_evidence: IDEA-196 EAAC, IDEA-197 RoleTriPool and IDEA-198 SEAV are rejected evidence only and are not code parents
problem_category: reliability_robustness
mechanism_tags: [eight_role_natural_language, counterfactual_spatial_trigger, trigger_conditioned_parent_risk, zero_crop_deployment, sequential_seen_training]
diagnostic_receipt: /data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/diagnostics/IDEA-199-trigger-risk/receipt.json@sha256:129434686e6472db715fa43d03def27f0d245fd89ab992598ec4400facbf2f70
implementation_branch: exp/v6/innovation/v6-try-003-svra
current_run: V6-TRY-003 / Gate0 completed

problem: The exact class-name Parent has useful Top2 candidates but cannot identify which rows are safely correctable. EAAC showed that a semantic-local 26-way policy can isolate a small correction-opportunity cohort, while every deployment-time crop verifier tested so far either damages Parent-correct rows or is unnecessary. IDEA-198's No-crop control outperformed its crop-dependent Full, proving that another raw crop is the wrong causal object.

hypothesis: A seen-trained semantic-local trigger can identify the subset on which correction is eligible, and a separate four-dimensional Parent-risk arbiter can decide whether Top2 should replace Top1. Their conjunction should transfer to class-disjoint evaluation without opening any raw crop.

old_solution_path: The exact metric Parent maps one frozen 336 CLS feature directly to class-name cosine logits and returns Top1. It has no role-conditioned local state, no learned correction-opportunity signal and no conditional keep/swap process. The rejected EAAC predecessor added role-window action selection and then physically re-read one raw crop for a fixed margin verifier; that failed path is evidence, not the code parent.

new_solution_path: Frozen class-name Parent -> Top1/Top2 pair -> eight class-specific role-question differences -> low-resolution CLS+576-patch explicit-abstention spatial policy -> binary correction-opportunity trigger -> four-dimensional Parent-risk arbiter -> trigger AND risk-controlled Top1/Top2 keep/swap. The 25-way location is a training-structured latent state and diagnostic output, not a deployed crop command. Deployment opens and encodes zero raw crops.

principle_difference: The Parent solves classification by one direct similarity ranking. SVRA instead learns a conditional correction process with an explicit intermediate state: a seen-supervised semantic-local correction opportunity must exist, and the Parent pair must simultaneously be risky. This conjunction cannot be reproduced by only changing a similarity temperature, calibration weight or head because removing either intermediate state changes which rows are eligible.

old_signal_or_primitive: class-name similarity logits only.
new_signal_or_primitive: seen-only counterfactual 25-location correction targets plus explicit abstention train a latent spatially factored trigger; deployment represents each row as Parent pair state plus trigger eligibility rather than another visual prototype or crop score.
paradigm_shift: Replace direct similarity or acquire-then-verify inference with a zero-crop trigger-then-risk correction path.
why_not_module: The eight-role encoder, attention/policy and small MLP are individually established components and are not claimed as separately novel. The candidate claim is the whole non-equivalent solver path and its new training signal/intermediate trigger state; if the no-trigger or always-swap controls match Full, this framework claim is withdrawn.

closest_paradigm_work: GapSight / Learning to Look Again (arXiv:2608.21762v2) learns when and where to re-read; Selective Classification (arXiv:1705.08500) and SelectiveNet (PMLR 97, 2019) learn rejection/coverage. SVRA does not claim to invent crop routing, abstention or risk heads. Its narrow distinction is training a spatially factored correction-opportunity trigger but using it for zero-crop GZSL pair arbitration rather than executing a new observation.

exact_three_modules:

1. S — Eight-Role Natural-Language Pair Questions. Input is not one class name. Every class has eight frozen complete English sentences in order `[beak, head_features, body_plumage, wings, tail, legs, overall_appearance, unique_discriminative_features]`, encoded once by frozen OpenAI CLIP. The real Laysan Albatross example begins `A photo of a Laysan Albatross, showing a long pink hooked bill with a dark tip.` and includes seven analogous class-specific sentences, including whole-body and distinctive descriptions. The source is the frozen role-text asset, not CUB expert attributes or human attribute answers. For the Parent leader/challenger pair, S outputs eight learned 64-D role questions from their role-sentence differences and name difference. S-off sets the complete learned question tensor to zero while preserving the name-only Parent, names and downstream shapes.
2. V — Counterfactual Spatial Opportunity Trigger. Input is frozen 336 CLS,576 projected 24x24 patch tokens, the eight S questions and Parent pair statistics. A shared role-to-window policy emits `[fixed zero abstain,25 window logits]`; `argmax=0` means keep/ineligible and any other class means trigger. The location classes come from seen-only all25 counterfactual correction targets and provide structured training supervision. At deployment the selected location is logged but never opened, cropped or encoded. V-off replaces all local patches by broadcast global CLS while retaining S and I.
3. I — Trigger-Conditioned Parent-Risk Arbiter. Input is exactly four frozen-Parent values `[leader-minus-challenger margin, entropy, logit mean, logit std]` on every row; the output is one keep/swap logit from `Linear(4,32)->GELU->Linear(32,1)` with zero-initialized output head. Final swap is `V_trigger AND sigmoid(logit)>0.5`; otherwise return Parent. I-off preserves the Full S/V policy and trigger computation but forces keep/Parent.

training_contract:

- Stage1 independently reproduces the reviewed EAAC S/V policy from the formal source identity: seen-only all25 target generation, exact natural target histogram4107 abstain/595 action, fixed4:4 group sampler, seed7, batch8,1000 updates, AdamW1e-3/wd1e-4 and no eval data. The all25 table is a training target generator from the same frozen CLIP backbone and seen truth; no teacher model or distillation is used.
- Freeze every Stage1 S/V parameter and all train action/trigger decisions.
- Stage2 reads no crop feature. On dev_train, only the574 frozen triggered rows enter I training; target1 iff truth is Parent challenger, target0 for leader/outside. The preregistered split is300 positive/274 negative. Train1000 updates with a fixed16+16 sampler, seed7, BCE, no class weight and no threshold search.
- Save one combined checkpoint and receipts for code/config/assets, Stage1 natural and sampled target histograms, train/eval action and trigger SHAs, Stage1 and Stage2 gradient gates, Stage2 batch-trace SHA and probability/group summaries.

deployment_contract: Use only frozen text embeddings and the existing336 CLS+projected-patch tensor. Freeze Parent, S/V policy, action and trigger before labels/metrics. Run the4D arbiter, apply trigger AND risk, and output final logits. `raw_image_open_count=0`, `raw_crop_encode_count=0`, `eval_all25_opened=false`, `B=0`, no PCLR online inference, no teacher, no crop fusion and no Top3.

module_off_contract: S-off, V-off and I-off use the single Full-trained combined checkpoint; no off path retrains, changes threshold or changes the Parent. S-off physically zeros all eight learned questions. V-off broadcasts CLS in place of576 patches. I-off retains Full logical S/V computation and returns Parent. Observed `Full-Parent`, `Full-Soff`, `Full-Voff` and `Full-Ioff` must each be at least+1.0pp; paired class-bootstrap CIs are reported, and a CI crossing zero is marked as formal/multi-seed risk rather than hidden.

non_equivalence_test: Full must beat same-trigger Always-swap by at least0.5pp with paired CI lower>0, proving I is not redundant. The same triggered-row-trained4D arbiter deployed on all rows, and a separately all-row-trained4D arbiter deployed on all rows, must each lose to Full by at least0.5pp with CI lower>0, proving V trigger is not a decorative mask. All three module-off point gates must pass. Any match withdraws the framework-path claim.

Gate0_controls:

- Hard: Parent, S-off, V-off, I-off, same-Full-trigger Always-swap, triggered-trained4D/no-trigger deployment, all-row-trained4D/no-trigger deployment, trigger/abstain, action occupancy, group safety, positive net, corrections>damages and physical B0 receipts.
- Complexity ceiling: train the prior13D no-crop arbiter on the same frozen Stage1 trigger rows, target rows, initialization seed,16+16 batch trace and1000 updates. It is not the method. If it beats4D by>=0.5pp with paired CI lower>0, the4D minimality statement is revised and13D cannot be adopted without a new Idea; this does not retroactively turn unproven action metadata into the current contribution.
- Report-only: selected action/confidence histograms and Center/Static/Random/TextHeatmap. SVRA does not claim that a deployed crop location beats heuristics because no location is executed.

minimal_viability: exact Stage1/Stage2 counts and SHAs; nonzero gradients in all enabled S/V/I paths; both trigger/abstain and keep/swap; at least two occupied actions and no one action above the preregistered70% trigger occupancy bound; zero raw crop/eval-all25 physical access; nonconstant I scores.

minimal_falsification: Run exactly Full, S-off, V-off, I-off, Always-swap, triggered-trained no-trigger, all-row-trained no-trigger and13D ceiling under one frozen Gate0. Drop if Parent or any module point gap is<1pp; Full-Always-swap or either Full-no-trigger control is<0.5pp or has CI lower<=0; net<=0; corrections<=damages; challenger trigger<=leader trigger; trigger/abstain or action occupancy collapses; or any raw crop/eval-all25/eval-label-before-decision boundary fails. No threshold, width, feature, prompt, policy, sampler, geometry, B>0 or Top3 rescue inside this TRY.

current_advantage: Frozen Gate0 gives4D Full68.335831 versus Parent66.692923 (+1.642908pp), S-off66.571724 (+1.764108pp), V-off66.851359 (+1.484473pp) and I-off=Parent (+1.642908pp), while opening zero raw crops. Always-swap and both no-trigger controls lose by statistically positive margins. Formal/multi-seed confirmation remains pending.
performance_status: above_parent_at_frozen_gate0_with_speed_or_cost_advantage
problem_family: class-disjoint GZSL Top1/Top2 correction under frozen vision-language features; broader coverage unknown.
shared_bottleneck: unsafe correction requires both a localized correction opportunity and an uncertain Parent pair; either alone over-corrects.
reusable_capability: unknown until a second dataset or formal protocol confirms the trigger-risk conjunction.
coverage_and_transfer: current evidence only on disclosed CUB dev seen/unseen split.
frontier_shift: potentially removes all deployment-time high-resolution crop reads while improving the exact Parent.
downstream_effects: lower inference cost and a deterministic keep path; no claim beyond measured GZSL correction.
failure_boundary: The4D risk state may be a dev-specific group prior; 4:4 Stage1 sampling may be essential but attribution-unclean; S-zero tests role questions rather than all semantics; action location itself is not claimed necessary; the single diagnostic lacks frozen-code and multi-seed evidence; truth outside Parent Top2 is unreachable.
paper_level_claim: none until formal protocol and multi-seed evidence. If those pass, the narrow candidate claim is zero-crop semantic-visual trigger-conditioned risk arbitration for GZSL pair correction, not first-ever routing or selective prediction.


## 2026-09-02 范式 Idea 双 Agent 对抗定稿

- 审核草稿 SHA256：`950339da7ea21d5dcddaf33a924cab49c7849f239b11aa97290f8cfb240578c3`
- Agent A 独立报告 SHA256：`5759d7b48e5ff0519554f291e5db48b4d6f404135879354c4437d3f083956873`
- Agent B 独立报告 SHA256：`53f549bebd2e6a7f868bd56fb6ac4717ffcef19c066695de16875154aefccef0`
- A/B 先独立审完同一准确草稿，再直接交换完整清单并逐项回应；交叉回应 SHA256 分别为 `97c4f5964abf442c18591853764497fb3b0f63a8639e746377e69da108f1499e` 与 `08067ddfe0177c22a8ce9816a578a1980b251ce9512783bba7157886fcb73e6f`。
- 双方最终均为 `P0=0 / P1=0 / P2=0 / pass`。
- 共同结论：`范式Idea双Agent对抗审核通过`。该通过只允许建立可证伪候选，不代表冻结 Gate、统计稳定性或论文 claim 已成立。

## 2026-09-02 冻结代码交叉审查

- 最终训练代码 commit：`8c50f02fc1f7dded7971b39cdac0b5d6b31c3c3a`
- 最终训练代码 tree：`62027dbc94df1ca0af49d8e3852a181a9a48f6b9`
- train config SHA256：`182d61455467f0f88594b7b56a21a916deb1aaa17c0213309a9fcfc4a389b581`
- eval placeholder config SHA256：`671e61e979ca08605a96859ca5cdda7288be1cd7bbfd2f466626788273dbb600`
- Agent A 最终复核 SHA256：`ca3a32dc5c4331e5463cabcc34c7ebca8eab8169ca53882980d572048c2349d1`
- Agent B 最终复核 SHA256：`6a0fe9882e85c368db00afd5750be926fb91c1e7f9a12df1056a862e0b93a801`
- A/B 交叉回应 SHA256：`49be5ca691121ce4b0d54f04f024045d0199a0c91bfeefe299566329481a22b0` / `43d261a6956b0b83a8980692beaf12370bb42a0b6bd6dfeba3bf459aa729daa6`
- 双方最终均为 `P0=0 / P1=0 / P2=0 / pass`；结论：`双Agent交叉审查通过`。本地最小相关测试为 `32 passed`。

## 2026-09-02 Gate0 训练收据

- checkpoint：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/svra/V6-TRY-003-GATE0-FULL/svra_gate0_combined.pt@sha256:89b485d58b1cc00c0bd1d248d4e1a472c5368bbfff32eaf3b6cc5c59e1d5eceb`
- train history：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/svra/V6-TRY-003-GATE0-FULL/train_history.json@sha256:d44473b0d182fcdc6c9da2a17bb4db040bead6bd101d6f9dcb1c6524175d9068`
- Stage1 目标计数严格复现：`4107` abstain / `595` action；loss `3.258096 -> 0.828329`。
- 冻结 policy 触发训练行严格复现：`574` rows，`300` challenger-positive / `274` leader-or-outside-negative。
- Stage1 step2 的 S 与 V upstream 梯度、Stage2 三个 arbiter 的 step2 hidden/output 梯度均为 finite/nonzero。
- 状态：训练完成，尚未运行 Gate0 eval；上述内容不是性能结果。

## 2026-09-02 冻结 Gate0 结果

- eval code/config commit：`bb4fae5b4da3df6dbcc2f733f8a46b64eeba71dd`
- final eval config SHA256：`e37f1997a85906020501fb124a918b3047335222230d39e72db423d8511477e3`
- result：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/svra/V6-TRY-003-GATE0-EVAL/result.json@sha256:0a979486e8c90596cd57e6ff51e556d9c49c5303caac4045a52dd268cb27ac70`

主要结果：

- Parent=`66.692923`；Full=`68.335831`；`Full-Parent=+1.642908pp`。
- S-off=`66.571724`，语义模块贡献 `+1.764108pp`。
- V-off=`66.851359`，视觉模块贡献 `+1.484473pp`。
- I-off=`66.692923`，交互模块贡献 `+1.642908pp`。
- 同一 Full trigger 的 Always-swap=`66.144910`，Full 优势 `+2.190921pp`，CI95=`[+0.614185,+3.971078]`。
- triggered-trained 4D 去掉 trigger 后=`55.758799`；all-row 4D 去掉 trigger 后=`56.390374`，均远低于 Full，证明 V trigger 不是装饰。
- 13D ceiling=`68.701935`，仅比4D高 `0.366104pp`；`Full-ceiling` CI95=`[-0.861985,+0.069248]`，没有达到预注册的稳定 `+0.5pp` 必要性门，因此保留更简单的4D主方法。
- Full 纠正84张、破坏46张、净增38张；280/2355触发，25个动作全部被使用，最大动作占用16.07%。
- `raw_image_open_count=0`、`raw_crop_encode_count=0`、`eval_all25_opened=false`、labels在全部logits冻结后才加载；所有预注册 Gate0 hard gates 为true。

判断：SVRA 达到当前 owner 合同的冻结 Gate0 成功条件：整模型高于准确 Parent，S/V/I 三模块各自同checkpoint关闭均下降至少1.0点，并且强控制与零裁剪边界通过。`H=80`仍只是未达到的追求目标。Parent与三个module point gap的单次class-bootstrap CI仍跨0，必须在formal/multi-seed确认中继续披露，不能写成统计稳定或正式框架已晋级。

## 2026-09-02 固定 checkpoint official 诊断

- result：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/svra/V6-TRY-003-OFFICIAL-DIAGNOSTIC/result.json@sha256:8816993a123f55eebb6230b425e693ea6d0bbe15d8889ec5346c1a2728a776d2`
- 该诊断不训练、不调参、不做official checkpoint选择，只把Gate0 checkpoint加载到完整200类轴并冻结全部logits后读取official labels。
- Parent：`U=62.210931 / S=64.205810 / H=63.192631 / ZS=79.580780`。
- Full：`U=61.265407 / S=64.252305 / H=62.723317 / ZS=79.546317`；`Full-Parent=-0.469314 H`。
- S/V/I gaps分别为 `+0.012295 / -0.600558 / -0.469314 H`；纠正85张、破坏112张、net=-27。
- 零裁剪和label-after-logits边界继续成立，但正式性能与三模块合同均失败。

root_cause_update: Gate0 policy/risk在100类active train axis上学习并在150类轴验证；official部署改为200类竞争轴，Parent entropy/mean/std与trigger分布继续漂移，冻结cohort的I无法向S/V回传纠正信号。IDEA-200按owner要求改为full200轴端到端联合训练。


