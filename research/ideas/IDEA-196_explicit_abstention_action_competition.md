# IDEA-196：Explicit Abstention Action Competition（EAAC）

idea_id: IDEA-196
status: rejected
source_type: IDEA-195_failure + first_principles + owner_continuous_rescue_authorization
method_name: Explicit Abstention Action Competition
method_acronym: EAAC
implementation_branch: exp/v5/innovation/v5-try-010-eaac
current_run: V5-TRY-010 / Gate0 explicit-abstain Full
train_config_sha256: `97b1cf6206b0f6081e24dd8b735afef9ddc7e00c70bccbec3f9263f9e7a7a326`
implementation_code_commit: `3366c1de468154dfb415433fe15f02e7295ae5ee`
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
predecessor: IDEA-195 SCAA rejected at Gate0
problem_category: visual_grounding
mechanism_tags: [explicit_abstention, normalized_action_competition, strongest_corrective_action, role_window_attention, pair_verification]

problem: SCAA learned signed damage but independent regression plus max over25 actions made at least one noisy score positive on60.6% of eval rows, causing239 leader damages and Full4.04 points below Parent. No-action was only an external threshold, not a competitor.

hypothesis: Putting abstention and all25 actions into one normalized 26-class decision can suppress multiple-comparison noise and learn a class-disjoint corrective policy while retaining role-window localization and B<=1 verification.

core_change: Reuse the shared per-action raw scores `s[25]`, prepend a fixed zero abstain logit, and form `policy_logits=[0,s_0...s_24]`. Class0 is abstain; classes1..25 execute action0..24. Argmax uses smallest ID, so the zero-initialized model initially abstains. Train cross-entropy with fixed4 challenger+4 non-challenger batches. For a train challenger with at least one correcting crop, target the action with the most negative leader-minus-challenger crop margin—equivalently the strongest challenger-over-leader support and strongest executable swap correction—tie smallest action ID; all leader/outside/uncorrectable challenger rows target abstain. No threshold exists.

old_solution_path: independent action values -> max then external threshold -> 25 chances for noise to trigger.
new_solution_path: abstain and actions share one categorical competition -> normalized choice -> at most one crop -> fixed pair verification.

principle_difference: The learned object is a mutually exclusive action including no-action, not 25 independently thresholded utilities. Abstention receives direct supervised probability mass.

novelty_boundary: Explicit abstention, categorical action policies and crop routing are standard; no component-level novelty is claimed. Under the owner's reduced requirement, this is a performance-oriented supporting path. Any later framework claim must rely on the whole 6+1+1 role-window GZSL combination and honest controls.
closest_paradigm_work: GapSight (arXiv:2608.21762v2) already learns when/where to re-read with a crop router; Selective Classification (arXiv:1705.08500), SelectiveNet (arXiv:1901.09192) and recurrent visual attention establish reject/abstain and glimpse-action precedents. EAAC claims none of these generic mechanisms.

exact_evidence:
- SCAA Parent66.692923, Full62.655040, trigger60.552%, corrections144/damages239/net-95.
- Natural train target: abstain4107 rows; executable action595; uncorrectable challenger39. Strongest-action class histogram `[4107,11,17,29,9,11,18,32,35,32,17,22,34,46,24,23,19,32,28,26,17,19,16,29,19,30]` where index0 abstains and index1..25 map action0..24.
- 4:4 sampling yields approximately balanced abstain/action labels without weighting; exact sampled class histogram is recorded.

modules_assets_offs: Same reviewed S eight-role questions, V role-window action features, I fixed one-crop verifier, assets, geometry, B1 and S/V/I-off definitions, reimplemented on a new formal-parent branch. Only policy normalization, target and CE loss change.

training_contract:
- target/group/margin only from100 dev-seen labels and all25 train crops;
- assert exact natural histogram above before optimizer;
- fixed4:4 sampler, seed7, batch8,1000 updates; report sampled26-class histogram;
- zero output head, CE over26 logits, step1/step2 gradient gates;
- no class weights, label smoothing, focal loss, threshold, temperature or schedule search.
- receipt reports for every correctable challenger: correcting-action count, strongest-vs-second correction-margin gap and chosen-action histogram; these diagnose strongest-target ambiguity but are not post-result tuning gates.

Gate0: fixed50 dev-unseen classes/150-axis macro Top1. Full-Parent and Full-S/V/I-off >=1pp/CI>0; Full-triggered Center/natural strongest-action StaticBest/HashRandom/TextHeatmap >=0.5pp/CI>0; challenger trigger>leader; leader damage<challenger corrections; net>0; trigger+abstain/action occupancy/B1 gates. StaticBest is defined only as the most frequent non-abstain action in the natural 4,702-row 26-class train target, tie smallest, and is executed only where Full triggers.

Gate1_after_Gate0: Natural-sampler is attribution only. Hard controls: CLS-only26, Image-only26, No-Glimpse Pair and Dense multi-label correction. Full beats each0.5pp/CI>0. Matching controls narrow/withdraw corresponding framework claims; performance may still be retained as engineering evidence under owner policy.

non_equivalence_test: Must reduce SCAA leader damage and multiple-comparison overtrigger through explicit abstention, not by changing crop budget or threshold. Fails if all-abstain/Parent, overtrigger, module-off gaps fail, or no-glimpse/image-only controls match.

minimal_viability: exact target histogram, both abstain and action labels occur over the complete frozen seed7/1000-step sampler trace (not necessarily every batch), all gradients, nontrivial26-way predictions, trigger+abstain, B1 clean.
minimal_falsification: Train only Full and run Gate0. Any hard gate fails -> drop; no temperature/label smoothing/class weight/threshold/prompt/window/B2 rescue.

current_advantage: none; standard explicit-abstain rescue derived from three measured failures.
performance_status: below_parent
failure_boundary: action label may be unlearnable from low-res features; strongest-margin labels may be noisy; policy may all-abstain or confuse spatial actions; S/V contributions may remain weak; class-disjoint transfer may fail.
paper_level_claim: no standalone novelty claim; only if the whole framework succeeds, report explicit-abstention role-window verification with proper attribution.

## Design review

review_date: 2026-09-01
review_agents: [`/root/rwdg_review_a`, `/root/rwdg_review_b`]
review_subject_sha256: `b6cf0e2aa528aab7313f14fa4e7777968a4f1831335d0d8e64d232027a9b048c`
review_result: `P0=0 / P1=0 / P2=0 / PASS`

- Independent review found no P0/P1 and requested precise strongest-margin, trace, StaticBest, ambiguity diagnostics and prior-work boundaries.
- The final draft closed every P2 and both agents independently returned `0/0/0`. Owner's standing authorization permits the proof Gate but not promotion or a novelty claim.

## Code review

review_subject_commit: `3366c1de468154dfb415433fe15f02e7295ae5ee`
review_subject_tree: `c0f7be7c37b4cfe62c2a397569f514df383e6740`
review_result: `P0=0 / P1=0 / 双Agent交叉审查通过`

- Both agents independently reviewed policy-logit offset, explicit abstain tie, strongest-margin targets/counts, CE, 4:4 trace, natural StaticBest, checkpoint/eval, B1 and group gates.
- They exchanged SHA-bound reports through shared files, replied to the other report and both concluded `P0=0/P1=0`. Local tests: `22 passed`.

## 2026-09-01 Gate0 result

train_checkpoint: `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/eaac/V5-TRY-010-GATE0-FULL/eaac_gate0_full.pt@sha256:481f757397937c9b4e37f9daffc676e215321c1d9a8c3e0fa06887059a62d188`
failure_receipt: `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/eaac/V5-TRY-010-GATE0-EVAL/failure.json@sha256:740731ad26cf2eee3cb968a3ba2d080b04096f48a1d459a0893fe85ad11efb0e`

- Parent=`66.692923%`; Full=`66.400172%`, Full-Parent=`-0.292751pp`; Full-Soff=`-0.166912pp`; Full-Voff=`-0.332856pp`.
- Trigger=`280/2355=11.889597%`; challenger corrections=53, leader damages=59, net=`-6`. Explicit abstention removed max-noise overtrigger but selected actions remained unreliable.
- All B1/data boundaries passed. The result rejects recovery through the current 36-patch-mean/indirect-attention representation.

root_cause: Each 6x6 window is averaged before role interaction and the learned attention has no direct per-role location target. Small discriminative peaks can be diluted; normalized abstention fixes calibration but not evidence localization.

decision: Drop EAAC. IDEA-197 starts the V6 development line with direct mean/max/min role evidence and tri-state action supervision; EAAC is evidence only and not its code parent.


