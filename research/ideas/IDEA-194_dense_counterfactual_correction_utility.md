# IDEA-194：Dense Counterfactual Correction Utility（D-CCU）

idea_id: IDEA-194
status: rejected_at_gate0
source_type: IDEA-193_failure + first_principles + owner_continuous_rescue_authorization
method_name: Dense Counterfactual Correction Utility
method_acronym: D-CCU
implementation_branch: exp/v5/innovation/v5-try-008-dccu
current_run: V5-TRY-008 / Gate0 balanced Full
train_config_sha256: `1c73cf95bcebdba91a288fcfb9a39dd3141eb2857f956dd11376a16053475f09`
implementation_code_commit: `161d8ff7c34823ab98699488289dfa4f3460a087`
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
predecessor: IDEA-193 RWDG rejected at Gate0
problem_category: visual_grounding
mechanism_tags: [baseline_aware_action_gain, correction_only_dense_field, balanced_pair_sampling, role_window_attention, one_shot_verification]

problem: IDEA-193 trained crop correctness rather than improvement. It rewarded Parent-correct leader rows for opening crops that merely retained an already-correct answer, producing 91.59% trigger, 151 corrections, 284 damages, net -133 and Full 5.54 points below Parent.

hypothesis: A dense field that labels only executable corrections relative to Parent, combined with a preregistered 4:4 challenger/non-challenger training sampler, can learn transferable role-window corrective evidence without rewarding unnecessary actions on Parent-correct or unreachable rows.

core_change: For each seen training row/action, define `g*_a=1` iff Parent is wrong, truth is its challenger and the fixed selected-crop I rule swaps to truth. All Parent-correct leader rows, outside-pair rows and non-correcting actions are zero. Fixed batch8 draws 4 challenger rows and 4 non-challenger rows each update. Architecture, 6+1+1 texts, role-to-window attention, 25 windows, threshold 0.5, I rule, seed7, AdamW and 1000 updates remain identical to IDEA-193.

old_solution_path: dense crop-correctness target -> act on both already-correct leaders and correctable challengers -> high trigger and avoidable damage.

new_solution_path: Parent outcome establishes the no-action baseline -> dense per-action correctness gain labels only actions that can turn a current error into the truth -> attention predicts corrective evidence -> abstain or acquire one crop -> fixed pair verification.

principle_difference: Utility is marginal executable improvement over abstention, not absolute correctness after acting. The 4:4 sampler is an explicit method component that makes this sparse signal trainable; outputs are not interpreted as deployment-distribution posterior probabilities and 0.5 remains a fixed decision threshold, not a calibrated probability claim.

novelty_boundary: Baseline-aware gain/advantage supervision is established, especially by GapSight (arXiv:2608.21762v2). No novelty is claimed for improvement utility, attention or active cropping. The only possible narrow framework-level claim is a complete 25-action binary correction field for class-disjoint GZSL under fixed Top1/Top2, seen-only training and B<=1. The owner allows established support methods; this candidate need only provide an honest useful framework component.

exact_evidence:

- IDEA-193 Parent=66.692923%, Full=61.151459%, Full-Parent=-5.541464pp.
- Trigger=2157/2355=91.592357%; corrected=151; damaged=284; net=-133.
- Train groups leader/challenger/outside=3478/634/590.
- Correction-only positives=6279 actions; all-row density=5.341557%; challenger density=39.615142%; 595/634 challenger rows have at least one positive, 39 are all-zero.
- Eval groups leader/challenger/outside=1584/381/390; current pair oracle corrects 375 challenger rows with zero damage.

exact_modules: Reuse the three reviewed computational modules by reimplementing them on a new branch from the formal parent: S eight-role pair questions; V role-to-window 25-action attention; I fixed one-crop keep/swap. S/V/I-off definitions and B1 physical contract remain unchanged.

training_contract:

- Build groups/targets only from 100 dev-seen labels and crop table.
- Every update uses independent seeded random permutations: 4 challenger + 4 from leader/outside; sampling with wrap/reshuffle, no eval data.
- Loss is ordinary unweighted BCE over the balanced batch and 25 actions. No class weight, focal loss, entropy loss or threshold tuning.
- Report natural train distribution and sampled distribution separately.

Gate0: Same Parent, S/V/I-off, Triggered Center/StaticBest/HashRandom/TextHeatmap, B1 and paired bootstrap contracts as IDEA-193. All Gate numbers are percentage-point differences in macro Top1 over the fixed 50 dev-unseen class vector under the 150-class axis, not formal Chen-style H. StaticBest is computed once as `argmax_a mean_train_natural(g*[:,a])` over all 4,702 natural-distribution training rows, tie smallest action ID; it never uses the 4:4 sampled stream. Additionally require challenger trigger rate > leader trigger rate, leader damage < challenger corrections, net correction >0 and both trigger/abstain. Full must exceed Parent and each off by 1.0 point and triggered controls by 0.5 point. All-abstain is exactly Parent and is already the Parent hard control.

Gate1_after_Gate0_only:

1. Natural-sampler D-CCU: identical correction-only target/architecture/updates but natural random batches; tests whether 4:4 sampling is necessary.
2. CLS-only Dense, Sparse-26, Image-only Window Dense and No-Glimpse Pair controls retained from the registered framework contract.

Full must beat CLS-only, Sparse-26, Image-only and No-Glimpse by 0.5 point with paired CI lower >0. Natural-sampler D-CCU is an attribution control, not a hard performance gate: if it matches or exceeds balanced Full, 4:4 sampling is unnecessary and must be removed from the final method/claim, while the correction-target evidence is judged against the four hard controls. If natural sampling falls behind, the final method may retain 4:4 only as a disclosed training aid, not as a novelty claim. Matching Sparse/No-Glimpse withdraws the possible core contribution; matching CLS/Image-only withdraws the role-window component.

non_equivalence_test: D-CCU must reduce leader damage and yield positive net corrections while changing predictions beyond Parent. It fails if it is all-abstain, if leader/outside over-trigger persists, or if no-glimpse/image-only/margin-like controls match.

minimal_viability: On a real train micro-batch, correction targets have the registered group/density counts and all projections receive gradients after the zero head moves. Frozen eval must show nonzero actions and both trigger/abstain without pre-action crop reads.

minimal_falsification: Train only balanced D-CCU Full and run Gate0. Any Parent/off/control/group-safety/B1 failure immediately drops it. No threshold, prompt, role, geometry, entropy, B2 or architecture rescue.

current_advantage: none; this is a failure-driven target correction with an existing +16.07pp pair-oracle opportunity.
performance_status: matched_parent_not_significant

failure_boundary: 4:4 oversampling changes the prior and may still over-trigger at eval; low-resolution evidence may not predict correctable crops; model may learn only Parent margin; class-disjoint transfer may fail; all-zero rows may still dominate; module contributions may remain below one point.

paper_level_claim: Only after Gate0/Gate1/formal/multi-seed evidence: “Dense executable correction labels can train one-shot pair verification in class-disjoint GZSL.” No first/attention/active-crop/advantage claim.

## Design review

review_date: 2026-09-01
review_agents: [`/root/rwdg_review_a`, `/root/rwdg_review_b`]
review_subject_sha256: `3d9b2d4eb2d9428f5107bf3d8554a50f718fb49c753db3ce4b06314bc84be3b3`
review_result: `P0=0 / P1=0 / P2=0 / PASS`

- Independent review first identified the 4:4 sampling-prior attribution and GapSight boundary as core issues.
- The unified draft made balancing an explicit non-posterior training aid, added a natural-sampler attribution control, narrowed novelty, fixed group-safety and all-abstain gates, and froze StaticBest on the natural 4,702-row distribution.
- Both agents independently re-read the final SHA and returned `0/0/0`. Owner previously authorized all subsequent in-scope rescue attempts without repeated approval; this authorizes the proof Gate but not promotion or a success claim.

## Code review

review_subject_commit: `161d8ff7c34823ab98699488289dfa4f3460a087`
review_subject_tree: `aac5f7f47b3899895c6fac879ee07fc6b1176dd2`
review_result: `P0=0 / P1=0 / 双Agent交叉审查通过`

- Two agents independently reviewed the frozen D-CCU target, target-count assertions, 4:4 sampler, natural StaticBest, checkpoint/eval interface and group-safety gates.
- Both found one shared eval-receipt `NameError` P0. The main agent fixed only that affected path plus method/alias identity; both agents restricted recheck to `aa4d4c5..161d8ff` and returned `P0=0/P1=0 / 受影响diff复核通过`.
- Local contract tests: `22 passed`; post-fix runner/model subset: `17 passed`.

## 2026-09-01 Gate0 result and experience

train_checkpoint: `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/dccu/V5-TRY-008-GATE0-FULL/dccu_gate0_full.pt@sha256:69dd27b19fbd6618a469d373cb2de91fa05cc0cb0f325a7619f31c7957c7bbab`
eval_config_sha256: `a3ceda138a7762769b2c43afdfbbd28a0bfee6c2a97c5251256d982717de670d`
failure_receipt: `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/dccu/V5-TRY-008-GATE0-EVAL/failure.json@sha256:3f2fac53d935d0552c822fa67f27ffa89f98abbc2494cda3d93fbe8ee0fa1da0`

- Training target statistics matched preregistration exactly; loss fell from `0.693147` to `0.216979`, and all projection gradients passed.
- Parent=`66.692923%`; Full=`66.740131%`, gain=`+0.047208pp`, CI=`[-1.977020,+2.118232]`. Full-Soff=`+0.443188pp`; Full-Voff=`+1.028243pp` but its CI crossed zero. No module contract passed jointly.
- Trigger fell from IDEA-193's `91.59%` to `11.97%`. Challenger trigger rate=`27.82%` exceeded leader=`6.69%`; corrections=66, leader damages=62, net=`+4`. The target repair therefore solved high-trigger damage but did not produce enough transferable action discrimination.
- Full used all 25 actions without collapse and all B1/data boundaries passed. Fixed-control advantages were small and nonsignificant.

root_cause: Correction-only binary targets distinguish beneficial actions from everything else, but collapse neutral actions and harmful leader-swapping actions into the same zero label. The model learned when correction may be possible, yet received no direct signal for avoiding action-specific damage; 66 corrections were nearly cancelled by 62 damages.

decision: Drop D-CCU at Gate0. Do not tune threshold/sampling ratio/prompt/attention/windows. A further rescue must be a new Idea with a signed action-advantage target: correction `+1`, no decision change `0`, damage `-1`, with abstention as the exact zero baseline.


