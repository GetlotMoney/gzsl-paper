# IDEA-202： Direct Evidence-Conditioned Swap Competition（DESC）

idea_id: IDEA-202
status: proposed_v6_e2e_rescue_precheck
implementation_branch: exp/v6/innovation/v6-try-005-desc
current_run: V6-TRY-005 / official precheck
base_framework: FRAMEWORK-V6-DEVELOPMENT
source_code_parent: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
predecessor_evidence: IDEA-200 J-SVRA rejected at official precheck; not a code parent
problem_category: reliability_robustness
mechanism_tags: [direct_keep_swap_competition, soft_spatial_evidence_pooling, auxiliary_counterfactual_localization, full_200_class_axis, zero_crop_deployment]

problem: J-SVRA's weighted opportunity-risk product triggered41.5% of official images, with leader trigger42.69% greater than challenger39.97%, producing238 corrections and570 damages. Full H56.250432 was below Parent62.441058 and No-joint60.285100. Balanced latent-event probabilities are not calibrated final correction utility.

hypothesis: A single directly supervised keep-versus-swap logit conditioned on a differentiable pool of the25 local semantic-visual evidence states will align training with the deployed decision and remain conservative under the natural 14.5% challenger rate, while an unweighted action auxiliary retains spatial grounding.

old_solution_path: S/V predicts an opportunity probability, I predicts a risk probability, and the product is thresholded. Separate positive reweighting makes both factors broad, so their product does not minimize final leader damage.

new_solution_path: Full200 Parent pair -> S role questions -> V25 local evidence hidden states and action logits -> softmax-weighted evidence pool plus max action logit -> I direct keep/swap logit -> one hard `swap_logit>0` decision. All modules update jointly from the same natural batches; no probability multiplication or separate trigger exists.

principle_difference: The deployed action is binary, so the primary learning object should be the binary correction decision itself. Spatial opportunity remains an auxiliary structured explanation and an input feature, not an independently thresholded latent event.

old_signal_or_primitive: two separately balanced latent probabilities combined multiplicatively.
new_signal_or_primitive: one final evidence-conditioned decision logit with a differentiable64-D spatial evidence summary;25-way counterfactual localization is auxiliary supervision.
paradigm_shift: Replace factorized probability gating with direct evidence-conditioned decision competition.
why_not_module: Softmax pooling, BCE and auxiliary CE are established. The candidate claim is the non-equivalent direct decision formulation and its end-to-end spatial evidence path; if Parent-only or No-action-aux controls match, the claim is withdrawn.

closest_paradigm_work: Selective Classification (arXiv:1705.08500) and SelectiveNet (PMLR97,2019) establish learned keep/reject decisions; attention-based multiple-instance learning establishes differentiable evidence pooling; GapSight / Learning to Look Again (arXiv:2608.21762v2) learns crop-review utility. DESC claims none of those generic components. Its narrow boundary is a zero-crop GZSL Top1/Top2 keep-swap logit that directly consumes a role-conditioned spatial evidence field while counterfactual25-location supervision remains auxiliary rather than a deployed crop action.

exact_three_modules:

1. S — Eight-Role Natural-Language Pair Questions. Input is200 classes ×8 frozen complete sentences `[beak, head, body, wings, tail, legs, overall, unique]`, not class-name-only and not expert attributes. Output is eight64-D pair questions. S-off zeros them.
2. V — Spatial Evidence Field. Input is336 CLS,576 projected patches, S questions and Parent statistics. It emits25 hidden64-D action evidence states and25 action logits. `a=softmax(action_logits)` and `e=sum_a a_a*h_a`; `m=amax(action_logits)` records whether localized evidence competes with fixed abstain. V-off broadcasts CLS instead of patches.
3. I — Direct Swap Competition. Input is `[Parent stats4, max action logit1, pooled evidence64]` =69 values. `Linear(69,64)->GELU->Linear(64,1)` with zero output head emits `swap_logit`; deployment swaps Parent Top1/Top2 iff `swap_logit>0`. I-off returns Parent.

full_axis_training_contract: All7,057 trainval images, labels from150 seen classes, frozen full200 text competition. The SHA-bound4702+2355 all25 tables provide train-only action targets. Official test images never provide gradients.

targets_and_loss:

- `y_swap=1` iff truth is Parent challenger; leader/outside are0. Census is1022 positive/6035 negative.
- `y_action26` is strongest corrective action or fixed abstain, census992 action/6065 abstain with the frozen26-class histogram.
- `L_swap=BCEWithLogits(swap_logit,y_swap)` on the natural batch, no class/positive weight.
- `L_action=CrossEntropy([fixed zero abstain,action_logits25],y_action26)` on the natural batch, no class/row weight.
- `L_total=L_swap+L_action`, coefficients exactly1. No detach, freezing, alternating optimizer, probability product, focal loss, margin or threshold search.

end_to_end_contract: `L_swap` gradients must reach I, V evidence pooling/action hidden and S role questions; `L_action` also trains S/V. Step2 and final receipts must show finite nonzero S/V/I gradients. The exact deployed `swap_logit>0` is the same logit optimized by BCE.

precheck_contract: One fixed seed7 batch50/1000-update natural trace (`randperm(7057)[:50]` each step). Train three conditions with the same initialization and trace: Full; No-action-aux (`L_swap` only); Parent-only (I receives Parent4 plus65 zeros, `L_swap` only). Evaluate fixed final checkpoints once on official test after all logits freeze. Full must beat Parent and same-checkpoint S/V/I-off by>=1.0 H, and beat No-action-aux and Parent-only by>=0.5 H with paired class-bootstrap CI lower>0; net corrections positive, both keep/swap, zero raw crop. Failure creates a new Idea; no tuning inside DESC.

formal_contract_after_precheck: If successful, run28,228 updates, batch50, official Full evaluation every141 updates and select one global best only by Full H. Module-offs use that exact checkpoint. Disclose `test_used_for_selection:true`, `test_used_for_hyperparameter_selection:false`, `nested_official_test_selection:true`, `strict_blind_claim:false`, `unseen_images_used_for_gradient:false`.

deployment_contract: Full200 frozen text/CLS/patch inputs; zero raw image/crop, no eval all25, teacher, distillation, PCLR online inference or Top3.

module_off_contract: Same Full checkpoint. S-off zero questions; V-off CLS broadcast; I-off Parent. Each Full-off H gap>=1; H80 target only.

non_equivalence_test: Full must beat same-trace No-action-aux and Parent-only hard controls. Full `swap_logit` must change under S/V-off, and step2 gradients must reach S/V/I. Otherwise DESC reduces to a Parent MLP or ungrounded direct head.

minimal_viability: full200 target census and SHA;25 hidden states/action logits;69-D interaction input; finite/nonzero gradients; both keep/swap; action diversity; logits frozen before official labels; per-condition logits/action/swap SHA; zero crop.
minimal_falsification: fixed1000-update official precheck above. No loss coefficient, positive weight, threshold, pooling temperature, width, prompt, sampler or feature-set rescue in this Idea.

identity_and_receipts: Before optimization save `initialization_sha256`, `batch_trace_sha256`, target-census SHA and action-target histogram. For Full, No-action-aux and Parent-only, record step1/step2/final raw and total loss components plus S/V/I gradient norms. Each final checkpoint and official condition must record `swap_logit_sha256`, `action_logits_sha256`, `evidence_pool_sha256`, final logits/action/swap SHA, checkpoint/config/code identities and exact same-trace assertion.

current_advantage: none yet. It is motivated by J-SVRA Full H56.250432, No-joint60.285100 and Parent62.441058.
performance_status: proof_not_yet_run
failure_boundary: Natural BCE may collapse to keep; auxiliary action CE may still dominate; soft action pooling may ignore absolute abstain strength despite including max logit; official precheck is nested test use, not blind validation; Parent Top2 limits reachable corrections.
paper_level_claim: none before official precheck and formal confirmation.


## 2026-09-02 范式 Idea 双 Agent 对抗定稿

- 最终草稿 SHA256：`82e7d4001325877c72af63896307c83dcb3fd289e47dedc388e66bc3ecdc07ca`
- Agent A/B 复核 SHA256：`502d1618f7ecd7e2f18180f7816be6a9ac33377527e126d99a4e08aa9947e2d4` / `e36984646f3923ac835ebfd5a281ba095b6248ff95cbdfd297ffb7f3d04d1b25`
- A/B 交叉回应 SHA256：`cda61f8c5065f32919a5d5b01b7170d4db99ff4ec8cb8023817d8407e8b811ce` / `d39b92805848cf1e5aba3ad9823934a6cb523733935e26fccb2d88907315377a`。
- 双方最终均为 `P0=0 / P1=0 / P2=0 / pass`；共同结论：`范式Idea双Agent对抗审核通过`。
