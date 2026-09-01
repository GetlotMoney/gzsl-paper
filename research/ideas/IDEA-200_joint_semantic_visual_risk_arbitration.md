# IDEA-200： Joint Semantic-Visual Risk Arbitration（J-SVRA）

idea_id: IDEA-200
status: proposed_v6_e2e_rescue_precheck
implementation_branch: exp/v6/innovation/v6-try-004-joint-svra
current_run: V6-TRY-004 / official precheck
base_framework: FRAMEWORK-V6-DEVELOPMENT
source_code_parent: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
predecessor_evidence: IDEA-199 SVRA Gate0 supported, but its fixed checkpoint failed official diagnostic; predecessor code is evidence, not an automatically accepted formal parent
problem_category: reliability_robustness
mechanism_tags: [end_to_end_joint_training, counterfactual_spatial_opportunity, differentiable_trigger_risk_conjunction, full_200_class_axis, zero_crop_deployment]

problem: Sequential SVRA passed the 150-axis class-disjoint Gate0 but its fixed checkpoint failed on official 200-axis GZSL: Full H62.723317 versus Parent63.192631, with negative Parent/S/V/I gaps and net -27. Training used a 100-class active axis and a frozen trigger cohort, while official deployment uses a 200-class axis. This shifts Parent entropy/statistics and prevents I loss from correcting S/V jointly.

hypothesis: Training S, V and I simultaneously on the same full 200-class competition axis, with an explicit differentiable conjunction between correction opportunity and Parent risk, will make the trigger and arbiter co-adapt without raw-crop inference and restore official U/S/H module contribution.

old_solution_path: Sequential SVRA first trains the S/V 26-way abstain/action policy, freezes its hard trigger rows, then trains I only on that fixed cohort. I cannot send gradients back to S/V, and the Gate0 train/eval class axes differ.

new_solution_path: Full200 Parent pair -> S eight role questions -> V 25 action logits plus fixed abstain -> differentiable opportunity probability -> I four-dimensional Parent-risk probability -> differentiable conjunction loss; all S/V/I parameters update in every optimizer step. Deployment retains the hard zero-crop rule `max_action_logit>0 AND risk_prob>0.5`.

principle_difference: A safe correction is the intersection of two latent events, not a post-hoc filter: local semantic-visual evidence says correction is available, and Parent state says Top2 is plausible. Joint training makes errors in the final conjunction update both event estimators, unlike frozen stagewise routing.

old_signal_or_primitive: frozen predicted trigger cohort followed by a separate BCE risk fit.
new_signal_or_primitive: three simultaneous seen-only targets on the full200 axis: a 26-way counterfactual action target, a Parent-Top2 truth target and a differentiable opportunity-risk conjunction target.
paradigm_shift: Replace train-then-freeze routing with one end-to-end probabilistic conjunction whose deployment remains a deterministic keep/swap path.
why_not_module: The individual attention, CE, BCE and MLP pieces are established. The candidate is the non-equivalent joint learning problem and its explicit intermediate opportunity/risk factorization; if gradients do not reach all S/V/I paths or no-joint controls match, the claim is withdrawn.

exact_three_modules:

1. S — Eight-Role Natural-Language Pair Questions. Each of200 classes has eight frozen complete English sentences `[beak, head, body, wings, tail, legs, overall, unique]`, not a class-name-only prompt and not expert attributes. S maps Parent leader/challenger role differences to eight64-D questions. S-off zeros the final question tensor.
2. V — Explicit-Abstention Spatial Opportunity Policy. Input is336 CLS,576 projected patches, S questions and Parent statistics. Output is25 action logits against one fixed zero abstain logit. Hard deployment trigger is `max(action_logits)>0`; no action is executed and no crop is opened.
3. I — Four-Dimensional Parent-Risk Arbiter. Input is `[leader-challenger margin, entropy, logit mean, logit std]`; output is one risk logit. Hard deployment swaps only when V triggers and `sigmoid(risk)>0.5`.

full_axis_training_contract:

- Train on all7,057 official trainval images with labels from150 seen classes, but compute Parent/S/V/I on the frozen full200 text axis. Unseen images never provide gradients; legal unseen class text participates only in competition.
- Training all25 targets are the SHA-bound union of dev_train4,702 and dev_eval_oracle2,355. Their raw indices are disjoint, union exactly trainval_loc, and reordered labels exactly match the official train cache.
- Fixed target census on the full200 axis:6065 abstain/992 action;4485 leader/1022 challenger/1550 outside. Exactly30 challenger rows have no corrective action target; these are intentional factorization-conflict rows where I should learn Parent risk but V should learn no evidence opportunity. The precheck must save this census as `target_census.json` with its own SHA before the first optimizer step.

joint_objective:

- `L_action`: weighted 26-way CE on `[0, action_logits25]`, target strongest corrective action or abstain. Action-positive row weight is fixed `6065/992 = 6.113911151885986`; abstain weight1.
- `L_risk`: BCEWithLogits on I, target1 iff truth is Parent challenger; positive weight fixed `(4485+1550)/1022 = 5.905087947845459`.
- `p_opportunity = sigmoid(amax(action_logits, dim=1))`. Because the fixed abstain logit is zero, `p_opportunity>0.5` is mathematically identical to the deployed hard rule `max(action_logits)>0`; all-zero logits give exactly0.5 and therefore hard abstain under the strict `>` rule. `torch.amax` supplies a subgradient to tied maxima.
- `p_risk = sigmoid(risk_logit)` and `p_joint = p_opportunity * p_risk`.
- `L_joint`: weighted binary cross-entropy between `p_joint` and `1[target26>0]`, using positive weight `6.113911151885986`.
- `L_total = L_action + L_risk + L_joint`; all coefficients exactly1. No detach, freezing, alternating optimizer or threshold search. Receipts must report raw and weighted means of all three losses plus finite/nonzero per-module gradient norms for S, V and I at step1, step2 and the final step.

precheck_contract: Before the expensive formal run, use one frozen1000-update batch trace (`seed7`, batch50, each step independent `randperm(7057)[:50]`) to train three fixed final-checkpoint conditions with no checkpoint selection or tuning: (1) Full joint objective; (2) No-joint control `L_action+L_risk` with identical initialization/batches/optimizer; (3) same-total-budget sequential control with500 policy updates followed by500 frozen-policy risk updates on the final hard-trigger cohort. Evaluate all final checkpoints once on official test. Full must beat Parent and same-checkpoint S/V/I-off by at least1.0 H point, have positive net corrections and zero raw-crop access. Full must also beat No-joint and Sequential by at least0.5 H point with paired class-bootstrap CI lower>0; otherwise the end-to-end conjunction claim fails. Any failure is recorded and triggers a new Idea/rescue rather than tuning this objective in place.

formal_contract_after_precheck: If precheck passes, create a formal confirmation Experiment with28,228 updates, batch50, independent randperm per step, official Full evaluation every141 updates, and one global best checkpoint selected only by Full H. U/S/H use200-class competition and ZS uses50 unseen classes. Module-offs run only at the selected Full checkpoint and cannot select it. Because the official precheck decides whether this method proceeds and the formal run later selects a checkpoint on official H, every receipt must disclose `test_used_for_selection:true`, `test_used_for_hyperparameter_selection:false`, `nested_official_test_selection:true`, `strict_blind_claim:false`, and `unseen_images_used_for_gradient:false`.

deployment_contract: One forward path, zero raw images/crops, zero eval all25, no teacher/distillation, no PCLR online inference, no Top3 and no threshold tuning.

module_off_contract: Same checkpoint and thresholds. S-off zeros role questions; V-off broadcasts CLS instead of patches; I-off retains S/V computation but returns Parent. `H_full-H_parent/Soff/Voff/Ioff` must each be at least+1.0 point; H80 is a target, not a hard line.

non_equivalence_test: Joint loss must produce finite nonzero step2 gradients in semantic parameters, visual upstream parameters and I hidden/output parameters. On every train/eval row, the receipt must assert exact equality between `(p_opportunity>0.5)` and `(max_action_logit>0)`. The same-trace No-joint and same-budget Sequential controls are precheck hard controls; either matching Full withdraws the joint-learning claim.

minimal_viability: SHA-bound full-axis target census including the30 conflict rows; full200 logits; nonzero S/V/I gradients; exact soft/hard trigger equivalence; both hard trigger/abstain and keep/swap; all25 actions train-only; zero-crop official logits frozen before labels. The precheck receipt must save per-condition logits/action/trigger/swap SHAs for Full and all same-checkpoint module-offs and controls.
minimal_falsification: fixed1000-update official precheck above. No loss-weight, coefficient, soft-trigger, threshold, width, prompt, sampler or class-axis rescue within this Idea.

current_advantage: none yet. Sequential predecessor Gate0 Full68.335831 exceeded its development Parent, but fixed official Full H62.723317 was below official Parent63.192631.
performance_status: proof_not_yet_run
failure_boundary: weighted joint losses may over-trigger; `amax` supplies sparse/tied subgradients even though its threshold exactly matches deployment; the30 challenger-without-action rows deliberately give I-positive/V-negative supervision; official precheck plus formal selection is nested official-test use and not blind validation; Parent Top2 reachability remains a ceiling; formal comparison to FRAMEWORK-V5 H81.068777 is cross-framework rather than the same CLIP parent.
paper_level_claim: none before official precheck and formal confirmation.


## 2026-09-02 范式 Idea 双 Agent 对抗定稿

- 最终草稿 SHA256：`6266d686f761865b10253e0a2ec6ac341eee69f75624939e69b267850b66135b`
- Agent A 复核 SHA256：`542c1126a597d9435e754c7ce0ccf152ee9fca4a72c7213a11c9cdd3875dac1a`
- Agent B 复核 SHA256：`258fd9127b80e71e40f7098e93d07f972588c063dea39472a784b033d4083740`
- A/B 逐项回应 SHA256：`6a8b403450e2345d75b6873a4098b249a4b8324cd2a3a6275096908a3fa2c563` / `1a6a34b731fa6db88d069b355a6c1dbd4e2f840286f01380da9ab8879f23c163`。
- 已关闭 soft/hard trigger 同义、No-joint/Sequential 硬控制、nested official-test披露、30条冲突行和完整收据要求。
- 双方最终均为 `P0=0 / P1=0 / P2=0 / pass`；共同结论：`范式Idea双Agent对抗审核通过`。
