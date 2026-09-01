# IDEA-208 draft / Continuous Top-2 Patch Margin (CTPM)

idea_id: IDEA-208
status: proposed_proof_of_path_new_v6_framework
source_type: owner_correction + V6-TRY-001_to_005_failures + first_principles
problem_category: class_competition
mechanism_tags: [top1_top2_pair, role_difference_query, patch_attention, continuous_margin, one_stage_training]
base_framework: frozen_CLIP_class_name_baseline
code_parent_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
candidate_framework: FRAMEWORK-V6-DEVELOPMENT
method_name: Continuous Top-2 Patch Margin
method_acronym: CTPM

problem: The intended V6 path asks whether 36 local patches can improve the Parent Top1/Top2 decision. Earlier RoleTriPool/SEAV/SVRA/DESC implementations converted this into staged action labels or hard keep/swap decisions. They collapsed to neutral/single actions on development data and caused severe official-test damage; patch-action AUC fell from about0.925 train to0.481 official. The research question should remain Top1/Top2 patch evidence, but the discrete action formulation should be removed.

hypothesis: A single end-to-end classifier can keep the frozen class-name CLIP logits as a neutral base, use 6+1+1 role differences to query the36 patches for signed Top1-vs-Top2 evidence, and learn a bounded continuous antisymmetric margin instead of a hard swap. Semantic, visual and interaction residuals can then each contribute at least1 H at the same checkpoint without requiring H80.

old_solution_path: `Parent Top1/Top2 -> hand-defined crop/action utility labels -> separate action/safety models -> threshold -> hard keep or swap`.

new_solution_path: `frozen class-name logits -> detached Top1/Top2 pair -> role-difference semantic evidence + role-conditioned36-patch evidence -> continuous semantic, visual and joint margin residuals -> scatter antisymmetrically into the same200 logits -> one Full CE`.

principle_difference: The model no longer predicts a discrete action outcome from surrogate crop labels. It directly optimizes how much continuous logit margin to transfer between the two live candidates under the final200-class classification loss.

old_signal_or_primitive: three-state action/crop correctness labels and thresholded swap decisions.

new_signal_or_primitive: differentiable signed candidate-margin residual learned from final classification error.

paradigm_shift: The basic learning object changes from “which action should I execute?” to “how much evidence-supported margin should move from one candidate to the other?”

why_not_module: A generic MLP head is not the claim. The candidate path is defined by the detached live Top1/Top2 pair, eight role-difference queries, 36-patch signed evidence, antisymmetric conservation of pair logit mass and direct Full CE. If class-name-only or global-CLS controls match it, the new framework fails.

## Inputs and exact three modules

Base input: row-normalized frozen OpenAI CLIP ViT-L/14@336 CLS `x[B,768]`, row-normalized class-name text `N[200,768]`, row-normalized role descriptions `T[200,8,768]`, and row-normalized36 coarse patches `P[B,36,768]` from the same accepted asset. Base logits are `l0=(1/0.07)*x*N^T`; the fixed scale14.285714 is used for all training conditions and does not affect parent argmax. The base has no trainable V5/TG/GTD/PCLR weights and no relation graph.

The candidate pair `(c1,c2)=top2(stopgrad(l0))` is selected once from base logits and held identical for Full and every S/V/I-off. Top2 uses stable descending argsort with class-id ascending tie break. `m0=l0[c1]-l0[c2]`.

S / Role-Difference Semantics (RDS): define positive evidence as support for currentTop2. Build eight signed queries `q_r=normalize(T[c2,r]-T[c1,r])`. Train bounded role classifier weights `w_r=0.75*tanh(raw_w_r)`, initialized0, giving all-class residual `l_role=(1/0.07)*x*(sum_r w_r T_r)^T`. For the live pair, `e_s[r]=x dot q_r`; a fixed `9->32->1` GELU MLP receives the eight e_s values plus base margin m0 and outputs `d_s=2*tanh(raw_s)`. Its final weight uses deterministic Xavier values multiplied by1e-3 and zero bias, so update0 perturbation is tiny but Full CE reaches the hidden path. Positive d_s moves mass towardTop2. Output `(q,d_s,l_role)`. Official S-off preserves the same Full q_r for downstream V/I and sets only `d_s=0,l_role=0`; candidate pair remains fixed. A separate named `S-query-off` diagnostic replaces q_r by `q_name=normalize(N[c2]-N[c1])` replicated eight times, but it is not the module-success S-off.

V / Signed Patch Comparator (SPC): project q_r and each of36 patches to64 dimensions; softmax over patches independently for each role; compute signed role support `e_v[r]=sum_n attention_rn*(P_n dot q_r)`. A fixed `8->32->1` GELU MLP outputs `d_v=2*tanh(raw_v)`. Its final weight uses deterministic Xavier times1e-3 and zero bias, ensuring update0 Full CE reaches query/key and attention. Positive d_v supportsTop2. Output `(d_v,attention,e_v)`. V-off sets `d_v=0` and sets the CMI input d_v and product d_s*d_v to0; S and I remain executable.

I / Continuous Margin Interaction (CMI): input `[m0,d_s,d_v,d_s*d_v,h0]`, where `h0=stopgrad(entropy(softmax_200(l0))/log(200))` lies in[0,1]. A fixed `5->32->1` GELU MLP outputs `d_i=2*tanh(raw_i)`; final weight is deterministic Xavier times1e-3 with zero bias. I-off sets d_i=0; S and V margins remain.

Final correction `d=d_s+d_v+d_i`. Scatter `-d/2` to c1 and `+d/2` to c2 so pair logit sum is conserved. Full logits are `l=l0+l_role+scatter(d)`. Positive d therefore moves evidence from currentTop1 toward currentTop2; learned signs may reverse that transfer. All outputs remain `[B,200]`.

## End-to-end training

One AdamW updates RDS, SPC and CMI every step. `L_total=CE(l_seen,y)+0.1*L_pair+0.01*L_attention_diversity`.

For truth-in-pair samples, define `pair_logits=[l[c1],l[c2]]` from the final logits after l_role and scatter(d); target is0 when y=c1 and1 when y=c2; `L_pair=mean CE(pair_logits,target)`. Pairs may include one unseen endpoint because unseen class text is legal, while y always remains a seen training label. Samples whose truth is outside the pair contribute exact zero pair loss, not a neutral action label. `L_attention_diversity` is computed per sample from the eight36-dimensional attention rows: epsilon-normalize each row, square all off-diagonal row cosines, then average across ordered off-diagonal pairs and batch. It never uses part annotations.

The final200-class CE has a nonzero update0 gradient path to every S/V/I MLP and the V query/key projections because final weights use fixed1e-3 initialization. Top2 indices are detached discrete routing, but every score after routing is differentiable. Pair CE supplies additional label-bearing gradients only on truth-in-pair rows; diversity is a non-label regularizer and cannot substitute for the Full CE micro gate. No teacher, distillation, stage, hard action threshold, crop generation, expert attributes, unseen-image gradients or PCLR online inference.

## Off paths and success goal

At one Full-selected checkpoint:

- S-off: `d_s=0,l_role=0`, while V/I keep the exact Full role-difference queries q_r; candidate pair is unchanged. `S-query-off` is reported separately and is not the success S-off.
- V-off: `d_v=0`, zero role evidence to I.
- I-off: `d_i=0`.

Owner-updated module evidence success requires all three `H_full-H_off>=1.0`. H80 and V5 H81.069 are contextual numbers only, not gates. To prevent trivial success by damaging Full/off semantics, Full must also exceed the exact frozen class-name parent H under the same asset/protocol.

current_advantage: none.
performance_status: proof_of_path

minimal_viability: first run a frozen parent diagnostic on train/test splits. Record counts for truth=c1, truth=c2 and truth outsideTop2; compute true-label Top2 coverage and oracle pair-flip U/S/H/ZS by keeping c1 when correct and replacing c1 by c2 exactly when truth=c2. Abort before implementation/formal training if combined train truth-in-pair coverage<60%, train truth=c2 rows<100, or official oracle H-parent H<1.0. If the gate passes, a real batch50 CUDA micro must prove finite nonuniform8x36 attention, exact pair identity across all offs, exact antisymmetric correction sum0, Full CE nonzero gradients into role weights/S MLP, V MLP/query/key, and I MLP, and no crop/graph/search runtime.

minimal_falsification: after the frozen gate passes, run one fixed seed7/batch50/28,228-update run with Full-H selection. Success requires Full>parent and each same-checkpoint S/V/I gap>=1. Margin bound2.0 is fixed and no post-run expansion/grid is allowed. Report gains separately for base-truth insideTop2 and outsideTop2, plus `margin_only/no_l_role` diagnostics. Any failure triggers one concentrated rescue based on which branch gap failed; no H80 gate.

non_equivalence_test: after module success, compare hard-swap using the same predicted sign, global-CLS comparator without patches, `S-query-off`, and `margin_only/no_l_role`. CTPM must improve the minimum S/V/I gap by0.5 over each relevant control for a method claim.

failure_boundary: base truth may often lie outsideTop2; coarse36 patches can miss small parts; continuous margin can still overfit seen pairs; role difference descriptions may be noisy; all-class role residual may dominate S contribution; tiny-initialized CMI may remain functionally silent; fixed base pair prevents a corrected class outsideTop2 from entering the pair.

problem_family: fine-grained recognition dominated by two confusing live candidates.
shared_bottleneck: global logits know the candidate pair but not the local signed evidence separating it.
reusable_capability: continuous evidence-conserving pair correction with text-conditioned patch comparison.
coverage_and_transfer: CUB seed7 only initially.
frontier_shift: module evidence, not an absolute H threshold.
downstream_effects: signed role/patch evidence traces, not causal localization.
paper_level_claim: none before real S/V/I gaps and controls. No first attention/Top2/reranking claim.

closest_paradigm_work: metric learning, pairwise ranking, learning-to-rank, top-k reranking, attention-based ZSL and differentiable routing are established. Local nearest failures/controls include GWPS Top1/Top2 pair training, RDSS role-difference scaling, MHPS hard-pair matching, and V6 RoleTriPool/SEAV/SVRA/DESC discrete action paths. The only narrow candidate claim is replacing staged crop-action supervision with one-stage role-patch continuous margin learning under fixed candidate routing; concrete external paper comparison is required only if proof-of-path and controls pass.

selection_disclosure: Chen-style official-test-selected; `test_used_for_selection=true`, `test_used_for_hyperparameter_selection=true`, `strict_blind_claim=false`, `unseen_images_used_for_gradient=false`.

owner_requirement: returns to the owner-specified Top1/Top2 x36patch new-framework line. Owner explicitly accepted the minimal inference relaxation: CLS classification plus one Top1/Top2 role×36patch continuous correction; no crop/search/graph/hard-swap/multiple rounds. No branch or RUN before dual-Agent Idea finalization and ARRA auxiliary result recording.

## 2026-09-02 范式Idea双Agent对抗定稿

- final_draft_sha256: `b125748c2a44ee078535ce8c65fb2ab8bbfc33f8598ba05f03da4c928dd16aa9`
- Round1独立：A=`f5db82cb7fdb73bf35b4e145743acf0ae8de9180d13520fa184397e8395f81ee`；B=`e3c813e16f698df82830dd02ea3fa41e2585a90f2db91124776e4025cb60e25e`。
- Round1交叉：A=`1279d6fa760887ec039f7c3de9a55276bbda68b6c5418c3a1a8626775b1be6a5`；B=`70787dc60a09d898514d037ff1eff8ce25ce675d0813837656c9d3cbab8a68f9`。
- Round2独立：A=`def9c386749b0e045bdc8ec56d12a9f944d575138f925e039a79a6c6308a6e68`；B=`1d71fedbfc5adb885546c36f193fcd1265b0b98f5b71e937fd203d7a15967470`。
- Round2最终交叉：A=`e5fa720e7c75093119c723412dbc4bc50e635714e015fde209b4d553aa5af911`；B=`d6c3583a578c91a002e3ddad6041dff93e33d3b2f3ee7a68f2374c0d1b44bd46`。
- 共同结论：`P0=0/P1=0/P2=0`，**范式Idea双Agent对抗审核通过**。仅授权proof-of-path实现。

## Top2可达性Gate

- script_sha256: `965e90705ffcd09aab37b2e91a40d226427f610a545820e6657b8101feb4445d`
- result: `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/ctpm/V6-TRY-010-TOP2-GATE/result.json@sha256:64d12884994ea44c30945662aaa1781fde8622c08f0526c473a59b043c181bfb`
- class-name Parent: `U/S/H/ZS=62.210363/64.205813/63.192339/79.681343`。
- Top2 oracle: `U/S/H/ZS=74.678022/78.614962/76.595937/90.382707`，`ΔH=+13.403597`。
- train：Top2覆盖`0.780360`，truth=c2共`1,023`张；test-seen/unseen覆盖`0.777211/0.748905`。
- 三项预注册硬门全部通过，允许进入实现。
- owner_confirmation_basis: owner接受最小在线修正推理，并要求S/V/I同checkpoint各>=1、不强制80。

