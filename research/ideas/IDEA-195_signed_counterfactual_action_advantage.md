# IDEA-195：Signed Counterfactual Action Advantage（SCAA）

idea_id: IDEA-195
status: testing_owner_preapproved_rescue
source_type: IDEA-194_failure + first_principles + owner_continuous_rescue_authorization
method_name: Signed Counterfactual Action Advantage
method_acronym: SCAA
implementation_branch: exp/v5/innovation/v5-try-009-scaa
current_run: V5-TRY-009 / Gate0 signed Full
train_config_sha256: `f25330ed5b7f8a67f012d40a2dc36e0155b3e88f82afc7d94b09dde7b28bc7f4`
implementation_code_commit: `8b390408235fd5012d70f161cfd47bc0914ab332`
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
predecessor: IDEA-194 D-CCU rejected at Gate0
problem_category: visual_grounding
mechanism_tags: [signed_action_advantage, damage_aware_glimpse, zero_baseline_abstention, role_window_attention, pair_verification]

problem: D-CCU reduced trigger from 91.59% to 11.97% and achieved 66 corrections, but 62 leader damages cancelled nearly all gain. Its binary target labels both harmless no-change crops and harmful leader-swapping crops as zero, so it cannot learn action-specific damage.

hypothesis: A signed dense action advantage field relative to abstention can teach the same role-window reader both which crops correct challenger errors and which crops damage correct leaders, yielding positive transferable net gain under B<=1.

core_change: For each seen row/action, target `A*_a=+1` if Parent is wrong, truth is challenger and fixed I swaps to truth; `A*_a=-1` if Parent is correct leader and fixed I would swap away; otherwise `A*_a=0`. The shared scalar head remains 25 actions but deployment value is `tanh(raw_logit)`. Choose smallest-ID max and trigger iff `max advantage > 0`; zero is the exact abstention baseline. Train `MSE(tanh(raw_logits),A*)` with the same fixed 4 challenger +4 non-challenger sampler, seed7, batch8, 1000 updates. No threshold, architecture, text, window or I-rule tuning.

old_solution_path: binary corrective/not-corrective target -> no distinction between neutral and damaging actions -> weak net gain.
new_solution_path: Parent defines zero action value -> each counterfactual action receives signed executable gain -> attention predicts benefit and harm -> act only when predicted best gain is positive -> fixed one-crop pair verification.

principle_difference: The learning object is action advantage over abstention, not correctness or correction probability. Negative supervision is attached to the exact leader/action combinations that caused IDEA-194 damage.

novelty_boundary: Signed advantage and baseline-relative utility are established ideas, and GapSight already learns crop improvement signals. No novelty is claimed for advantage, attention or active cropping. The candidate only tests a narrow signed 25-action field for class-disjoint GZSL Top1/Top2 verification; owner permits established supporting mechanisms.

exact_evidence:
- IDEA-194 Parent=66.692923%, Full=66.740131%, gain=+0.047208pp CI crosses zero; 66 corrections/62 damages/net+4; trigger11.97%; S gap0.443; V gap1.028 CI crosses.
- Natural train groups=3478/634/590.
- Signed action counts over117,550 actions: +1=6,279 (5.341557%), -1=35,262 (29.997448%), 0=76,009 (64.661%).
- Fixed 4:4 sampling balances challenger/non-challenger groups. It does not guarantee both signs in every batch because some challenger/outside/leader rows can be all-neutral; the complete seed7/1000-step sampler trace must report natural and sampled `-1/0/+1` counts and must contain nonzero positive and negative totals.

modules_and_assets: Same reviewed S eight-role questions, V role-to-window attention, I fixed B1 verifier, assets, geometry and off definitions, reimplemented on a new branch from the formal parent. Only target, scalar activation/loss and zero decision boundary change.

training_contract:
- targets/groups only from100 dev-seen labels/all25 train crop table;
- exact target-count assertions above before optimizer;
- output head zero init; step1/step2 gradient gates retained;
- ordinary unweighted MSE, no focal/class weight/entropy/calibration;
- report target counts per group/action and natural/sampled sign frequencies; no per-batch sign-presence claim.

Gate0: fixed50 dev-unseen class vector/150-axis macro Top1. Full-Parent and Full-S/V/I-off >=1.0pp with paired CI lower>0; Full-triggered Center/natural Signed-StaticBest/HashRandom/TextHeatmap >=0.5pp; challenger trigger rate>leader; leader damage<challenger corrections; net>0; trigger+abstain with explicit per-group counts; action occupancy/B1 gates unchanged. Signed-StaticBest=`argmax_a mean_natural(A*[:,a])` using the signed sum/mean over all natural 4,702 rows (not positive-only count), tie smallest.

Gate1_after_Gate0: Natural-sampler is non-hard attribution. Hard controls are (1) Unsigned-calibrated: same architecture, 4:4, `tanh+MSE` and trigger `>0`, but target is `{0,+1}` with every damage `-1` replaced by `0`; (2) CLS-only Signed; (3) Sparse single-action; (4) Image-only Signed; (5) No-Glimpse Pair. Full must beat every hard control by0.5pp/CI>0. Unsigned matching Full proves any gain came from activation/zero-threshold calibration rather than signed damage evidence and withdraws the SCAA core claim. Matching natural removes4:4 from method/claim; matching Sparse/No-Glimpse withdraws core claim; matching CLS/Image-only withdraws role-window component.

non_equivalence_test: Must improve net correction beyond D-CCU and lower leader damage through action-specific negative evidence. Fails if all-abstain, only margin/challenger prior, overtrigger, or hard controls match.

minimal_viability: exact signed counts, nonzero positive and negative totals over the frozen sampler trace, all gradients after zero head, finite nonconstant signed predictions, trigger+abstain, no pre-action crop read.
minimal_falsification: train only balanced Full and run Gate0; any parent/off/control/group/B1 failure drops. No threshold/activation/loss-weight/prompt/window/B2 rescue.

current_advantage: none; failure-driven signed reformulation with +16.07pp pair-oracle opportunity.
performance_status: proof_of_path
failure_boundary: low-res evidence may not predict high-res advantage; MSE may regress toward zero/all-abstain; the maximum over25 actions may turn tiny near-zero positive noise into over-trigger; 4:4 prior may overtrigger; damage patterns may not transfer; S/V contributions may remain weak.
closest_paradigm_work: GapSight / Learning to Look Again (arXiv:2608.21762v2) already mines global-vs-crop answer loss/margin improvements and trains review/utility/box prediction. SCAA does not claim baseline-relative utility or crop routing originality; only the fixed signed discrete GZSL experiment boundary is tested.
paper_level_claim: only after all evidence: “Signed counterfactual crop advantages support damage-aware one-shot pair verification in class-disjoint GZSL.” No first claim.

## Design review

review_date: 2026-09-01
review_agents: [`/root/rwdg_review_a`, `/root/rwdg_review_b`]
review_subject_sha256: `8aa0793f08f6a222604e13ae5dc6ad655c73837b9ccb5dafe2b5ebe970a13816`
review_result: `P0=0 / P1=0 / P2=0 / PASS`

- Initial independent review identified the false per-batch sign guarantee and the need to isolate signed damage evidence from activation/trigger calibration.
- The final draft limits 4:4 to group balance, records actual sign frequencies, fixes Signed-StaticBest to the natural signed mean, adds the Unsigned-calibrated hard control and narrows the GapSight boundary.
- Both agents independently checked the final SHA and returned `0/0/0`. Owner's standing authorization permits the proof Gate but not promotion or a success claim.

## Code review

review_subject_commit: `8b390408235fd5012d70f161cfd47bc0914ab332`
review_subject_tree: `ddedff2bad608e761c754bd89af5039e6ee24f25`
review_result: `P0=0 / P1=0 / 双Agent交叉审查通过`

- Both agents independently reviewed signed target mapping/counts, tanh/MSE/zero trigger, 4:4 trace statistics, SignedStaticBest, schema/checkpoint, group/B1 gates and tests.
- They exchanged SHA-bound full reports through shared files, responded to the other report and both concluded `P0=0/P1=0`. Local tests: `23 passed`; reviewer subsets: `18/11 passed`.
