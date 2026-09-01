# IDEA-206 draft / Anchored Role-Relation Alignment (ARRA)

idea_id: IDEA-206
status: contingent_rescue_proof_of_path_draft
source_type: IDEA-205 interim failure evidence + exact affine diagnostic + first_principles
problem_category: class_competition
mechanism_tags: [anchor_preservation, role_patch_attention, compiled_relation_residual, simultaneous_parameter_groups, graph_free_inference]
base_framework: FRAMEWORK-V5
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
candidate_framework: FRAMEWORK-V6-DEVELOPMENT
method_name: Anchored Role-Relation Alignment
method_acronym: ARRA
predecessor: IDEA-205/RGRA; not a code parent

problem: RGRA is genuinely one-stage, but its formal run was stopped at update22,983 after proving that `P_v5` was materialized while the source model remained in train mode: TG dropout changed the claimed fixed anchor by max-abs0.145699 across two seeds. Before stopping, best H was77.911 at update423 and then seen overfitting dominated. Same-checkpoint diagnostics at that best state show S-off gap+9.549 and V-off gap+2.250, but I-off gap only+0.065; additive is0.022 H above Full and shuffled differs by only0.035. Failure receipt: `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/rgra/V6-TRY-008/failure.json@sha256:7637294196a5fe00da91ff2550a2bd11e8ed9068f7c2541d319ebe5e09e9d5d7`.

hypothesis: Preserve the accepted V5-R2 class prototype table exactly as the semantic anchor and express 6+1+1 roles as additive trainable classifier directions rather than mixing them into the anchor. Initialize the compiled relation field at the empirically nontrivial inherited scale and make role visibility a zero-initialized modulation around that relation residual. Then one simultaneous optimizer can learn patch grounding without first destroying the strong class boundary, allowing Full to exceed V5 H=81.068777 while S/V/I each retain at least1 H same-checkpoint contribution.

old_solution_path: RGRA computes `p=normalize((1-rho)P_v5+rho*grouped_roles)` with rho=0.1, adds a patch term, and uses `alpha*g_v*r` with alpha=0.05 and fixed gamma0.91. The anchor, role geometry and relation strength are all rescaled at initialization.

new_solution_path: Keep `P_v5` untouched; add all eight role directions linearly with inherited role0/role6 initialization; compute role-conditioned coarse-patch evidence; add a compiled relation residual with nonzero alpha from step0; let the patch support gate modulate around identity rather than suppressing the base relation residual. All trainable S/V/I parameters update in every step from the same Full loss.

principle_difference: The rescue changes the optimization state, not merely a scalar: validated class weights become an invariant anchor and new evidence is learned as residual directions around it. Visual grounding controls how an already-active relation field is modulated, instead of being the only path through which relation evidence can exist.

old_signal_or_primitive: A convex mixture replaces the anchor prototype, while relation evidence is a near-zero gate-scaled add-on.

new_signal_or_primitive: A fixed semantic anchor plus separately identifiable role, patch and relation residual coordinates; the interaction primitive is an active relation field with a learned visibility-conditioned deviation around identity.

paradigm_shift: This remains the same broad graph-free joint-classifier family as IDEA-205, but it is a new falsifiable rescue because the representation formula and interaction state change. It is not yet Innovation.

why_not_module: Role attention and additive residuals have prior art. The only candidate contribution is the unified anchored residual coordinate system with visibility-conditioned compiled relations. If additive relation alone matches the conditional term, retain at most an engineering framework and withdraw the conditional-interaction claim.

## Exact inputs and three deployed modules

Inputs remain exactly the audited assets from IDEA-205: V5 `role_sentence_embeds[200,8,768]` containing six visual roles plus overall and unique descriptions; exact V5 CLS `[B,768]` and 36 coarse patches `[B,36,768]`; 438 legal directional relation-text pairs. Frozen OpenAI CLIP is outside the trainable boundary. No teacher, distillation, expert attributes, unseen-image gradients, PCLR online inference or sequential training.

S / Anchored Role Classifier (ARC): let `T_r[c]=normalize(role_sentence_embeds[c,r])`, `r=0..7`, and fixed `P=P_v5` extracted only after `source.eval()`. There is no semantic adapter. Train exactly eight bounded coefficients `w_r=0.75*tanh(raw_w_r)`, initialized `w0=0.16,w6=0.36`, all others0. `Q_s[c]=P[c]+sum_r w_r*T_r[c]`; `l_s=scale*z*Q_s^T`, with source scale frozen. S-off is the required `P_v5 only` semantic-increment control: set all `w=0`, keep P and the raw role queries used independently by V, and change nothing in V/I. Mean8 is reported only as a reference and is not the S module-off.

V / Role Patch Aligner (RPA): use one exact bottleneck residual `A_v:768->64->768` with zero-initialized output layer; query/key are exact `768->64` bias-free Xavier projections. The raw local6/overall/unique queries attend over the exact36 patches and produce `z=normalize(x+A_v(x))`, `l_v[B,200]` and `g=sigmoid(clamp(zscore_class(l_v),-5,5))`. `l_v` enters Full through `beta=sigmoid(raw_beta)` with max1 and init0.10. V-off uses raw normalized x, `l_v=0`, and neutral `g=0.5`; RFM remains active.

I / Anchored Relation Residual (ARR): compile `G=normalize(solve(B^T B+0.3I,B^T)D)` once; `z_r=normalize(z+A_r(z))` uses the exact R2 Reader initialization and `r=z_r G^T/0.2`. Train `alpha=2*sigmoid(raw_alpha)` initialized1.0 and `delta=tanh(raw_delta)` initialized0. The exact interaction is `l_i=alpha*(1+delta*(2g-1))*r`. Thus update0 has a full relation residual, while conditional grounding begins as an identity-preserving zero residual and can learn in either direction. I-off sets all of `l_i=0`. Additive control fixes delta=0 using the same alpha/r. Shuffled control permutes g independently per image before the conditional factor.

Final logits: `l_full=l_s+beta*l_v+l_i-gamma*1_seen`, with fixed inherited R3 gamma0.575 in Full and every off/control. Main CE uses only the150 seen columns and labels; evaluation uses200-class GZSL and50-class ZS.

## End-to-end optimization

All S/V/I groups are present from update1 in one AdamW and one `zero_grad -> Full forward -> L_total.backward -> step`. Two simultaneous parameter groups are fixed: role coefficients plus relation reader/alpha use3e-6; patch adapter/query/key/beta and conditional delta use3e-5; weight decay1e-3 and cosine minima one tenth of each rate. P and source scale are frozen inputs, not trainable modules. This is not stagewise because no trainable group is frozen or activated later and every step updates all groups. One global Full-H checkpoint is selected.

`L_total=L_cls(full_seen,y)+0.3*L_topology+0.1*L_direction`. Let `Qn=normalize(Q_s)` and fixed `Pn=normalize(P)`. `L_topology=1-corr(offdiag(Qn Qn^T),stopgrad(offdiag(Pn Pn^T)))`, using centered vectors and epsilon1e-8. For sample label y, only incident edges `E_y={e:y=a_e or y=b_e}` enter `L_direction`; with the same Full visual feature z and R2-initialized reader `z_r=normalize(z+A_r(z))`, edge logits are `[z_r dot text(a rather than b)/0.07,z_r dot text(b rather than a)/0.07]`, and the endpoint matching y is the target. Nonincident edges are ignored. L_cls-only CUDA micro must give finite nonzero gradient norms to role coefficients `raw_w`; visual `A_v/query/key/beta`; relation reader `A_r`, alpha and delta. Unseen rows receive no direct label gradient and rely on shared text/field structure.

## Evidence and falsification

accepted_parent: V5 `U/S/H/ZS=80.694097/81.446952/81.068777/88.785273`.

IDEA-205 failure evidence: stopped engineering-invalid run best update423 `U/S/H/ZS=75.371444/80.628538/77.911411/84.670228`; S/V/I gaps `+9.548995/+2.250195/+0.064736`; Full-additive `-0.021937`; Full-shuffled `+0.035142`; anchor replay max-abs0.145699. These values are failure diagnostics, not valid RGRA performance evidence.

exact affine diagnostic: after explicitly setting the R2 source to eval mode, use `x=normalize(CLS)`, `Q=P+0.16*T0+0.36*T6`, the exact R2 Reader `z_r=normalize(x+A_r_R2(x))`, `r=z_r G^T/0.2`, and logits `scale*xQ^T+1.0*r-0.575*1_seen`. This gives `U/S/H/ZS=77.396578/83.640134/80.397321/88.425398` before any patch residual or joint training. Same-formula controls are P-only H78.493750, P+roles H80.241347 and P+relation H79.581197. Receipt: `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/rgra/IDEA-206-ARRA-AFFINE-DIAG-R1/result.json@sha256:0d5323edc6881b703818a9d103da9447919833cde98f626035783ba649c18a24`; generating script SHA `ff72b80ec2ba8ca53f8652f96a59b912bf820706ab2af0930e598e9270246d56`. Root/main verified the remote file and SHA after generation. This official-test diagnostic freezes alpha/gamma/role initialization and therefore sets `test_used_for_hyperparameter_selection=true`, `nested_official_test_selection=true`, `strict_blind_claim=false` for ARRA.

minimal_viability: before enabling patch logits, an explicit `beta_override=0,delta_override=0` affine subpath must reproduce the receipt formula logits within1e-5; this is not the ARRA Full update0 metric. ARRA Full update0 includes beta0.10 and must be separately reported without an equality target. Batch50 CUDA micro must show L_cls-only gradients to role coefficients, V adapter/query/key/beta, relation reader/alpha/delta, nonuniform attention, alpha0==I-off, and graph-free export parity.

minimal_falsification: fixed seed7/batch50/28,228 updates/eval141. Successful framework requires selected Full H>81.068777 and same-checkpoint S/V/I gaps each>=1.0. H80 remains target only. If any fails, ARRA is not a successful framework. No post-run gamma/alpha grid expansion; alpha1/gamma0.575 are frozen from the disclosed pre-run diagnostic.

near_threshold_policy: if Full exceeds81.068777 by<0.2 H or any S/V/I gap exceeds1.0 by<0.2 H, the result is only tentative proof-of-path and cannot support a strong paper claim until a fixed additional seed or owner-approved confirmation reproduces it.

non_equivalence_test: graph-free deployment opens no relation/edge assets; alpha0 exactly equals I-off; delta0 additive control uses same checkpoint; per-image shuffled-g support control preserves each row's values. Conditional Full must beat additive and shuffled by at least0.5 H with net corrections>=20 for the conditional interaction claim. If performance passes but these fail, keep only engineering framework status.

current_advantage: none. The audited H80.397321 affine diagnostic is below V5 and only indicates a better rescue starting state than RGRA. Performance status remains proof_of_path.

performance_status: proof_of_path

failure_boundary: direct role logits may double-count P semantics; patch attention may still overfit seen classes; alpha1 can dominate and harm ZS; identity-centered modulation can remain delta≈0; fixed low base LR may preserve but not improve the anchor; official diagnostic selection is nested/non-blind; CUB-only evidence gives no generality claim.

problem_family: fine-grained GZSL with strong pretrained class weights, local role cues and pairwise relative descriptions.
shared_bottleneck: new end-to-end evidence can destroy a validated boundary faster than it learns transferable corrections.
reusable_capability: if supported, anchor-preserving simultaneous residual learning for semantic/visual/relation evidence.
coverage_and_transfer: CUB seed7 only; cross-seed/dataset untested.
frontier_shift: possible graph-free replacement of online PCLR while retaining a strong starting boundary; no speed claim before measurement.
downstream_effects: attention/support traces are diagnostics, not causal localization.
paper_level_claim: only after performance, non-equivalence and near-threshold confirmation gates: an anchor-preserving graph-free classifier jointly learns role, patch and compiled relative evidence without sequential module training. No first/paradigm claim. If promoted toward paper-core, TaskRes/Tip-Adapter/MMA and relation-classifier synthesis boundaries require a fuller systematic comparison; current review is only sufficient for proof-of-path.

closest_paradigm_work: inherit IDEA-205 boundaries—Dense Attribute Attention CVPR2020, PSVMA CVPR2023, AdaptCLIPZS CVPR2024, VSPCN CVPR2025, and DGP CVPR2019. ARRA additionally overlaps TaskRes (CVPR2023), which freezes a pretrained text classifier and learns task residual classifier weights; Tip-Adapter (ECCV2022), which preserves CLIP logits and adds a residual cache score; and MMA (CVPR2024), which adapts visual/language signals jointly. Therefore anchor preservation, residual logits, adapters and multimodal fusion are explicitly not claimed as new. The only conditional claim remains role visibility modulating a compiled relation residual inside one GZSL training graph, subject to additive/shuffle controls.

- TaskRes primary page: https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Task_Residual_for_Tuning_Vision-Language_Models_CVPR_2023_paper.html
- Tip-Adapter primary page: https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/154_ECCV_2022_paper.php
- MMA primary paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_MMA_Multi-Modal_Adapter_for_Vision-Language_Models_CVPR_2024_paper.pdf

owner_requirement: RGRA failure is now recorded; this rescue remains authorized by the owner's standing instruction to keep trying after failure. No branch or RUN is authorized until this revised draft passes the required dual-Agent adversarial finalization.

## 2026-09-02 范式Idea双Agent对抗定稿

- final_draft_sha256: `12935af213f6d7150251e2fb715f6e32f81ecff5d96ca9df7e086ee10ef220bc`
- Round1独立：A=`c34c4daea908d561d29264ffa0d537ff66af387d61e3bfa19c096441f7ad241f`；B=`f4683a455ae58c2dbdedbb7e4d281eeb256b485901ce4f3639ad707a2fec89b4`。
- Round1交叉：A=`b109338bd53776cc4a3ee8b38bc418b42374505161d812603aad2313d3ab18fc`；B=`f64c8217eb4563f869421e798403424b41e54afdc3b8e38cce4e2ab82348ae84`。
- Round2独立：A=`53392a4852fe009e7e8a411e89f084324400a58bd1a528f699b14a81db750698`；B=`83ed43b4ea6a9ad5fac448879df0c680159c51d1ebc8fe412f981ddd15ab2272`。
- Round2交叉：A=`3fb4bab0489feed9d9a80a98f4b6451d07bd1cc76a5aa7601991fba0d23defbf`；B=`784c99921fd55dcd3c6c9db9019e1a3198051d5be3662e4caa1efe6d976b52f6`。
- Round3独立：A=`fae6cf7c3aa6bb89d0390a595a0f878054b3380b9b36a042f721fb174765c08b`；B=`90187355d6a77e6f9d306f4b3c80dedfeffdd5e32e1b545903f166dcfff25bb0`。
- Round3最终交叉：A=`4261c9c84ebbd0d4fe7c2c7050d50bf55fb624a2a861a56522982810feeb81ee`；B=`89e0d2ca142e750024a067305dc7b7bbfa720d83fff10d4485454fb1c791d028`。
- 共同结论：`P0=0/P1=0/P2=0`，**范式Idea双Agent对抗审核通过**。仅授权contingent rescue proof-of-path实现；不等于Innovation、性能或代码审核通过。
- root已核验远端failure receipt与R1 affine receipt的真实路径、内容和SHA。
- owner_confirmation_basis: owner明确要求端到端并授权失败后继续补救；因此批准从准确父提交`52b511d77b4ad048f35b40dc3cbd9afd092167e9`独立实现V6-TRY-009。

## 2026-09-02 实现代码双Agent交叉审查

- reviewed_commit: `a3bebba64906a1979f549c48e77986461ac0bae6`
- reviewed_tree: `76eab3cdb220677fcf10d508768e24deedd0b0ae`
- config_sha256: `58f631a37f984c72818c30fe76f6f9b19e79b325179237241202e1902330b5e3`
- review_receipt: `experiments/v6/innovation/V6-TRY-009_ARRA_REVIEW.md`
- final_A: `0069bfe645ea3b20146df4f538725a54d1fd8837330199d97d2c0f414208b999`
- final_B: `ca785dae6d0f52ec023dc223a5b84209b8df2a08772f29e36aca4ec951283a49`
- result: `P0=0/P1=0`，**双Agent交叉审查通过**；只授权固定配置RUN。
