# IDEA-197：Role Tri-Pool Signed Window Utility

idea_id: IDEA-197
status: rejected_at_gate0
source_type: owner_hypothesis + IDEA-193/195 failure diagnosis
problem_category: visual_grounding
mechanism_tags: [role_patch_difference, mean_max_min_pooling, signed_action_target, natural_sampling, one_shot_verification]
method_name: Role Tri-Pool Signed Window Utility
method_acronym: RoleTriPool
base_framework: FRAMEWORK-V6-DEVELOPMENT
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
implementation_branch: exp/v6/innovation/v6-try-001-role-tripool
current_run: V6-TRY-001 / Gate0 Full-only
train_config_sha256: `80e564bc9f6636a377d9c4955ec27454cce17b96148c468877d7d164738ecea4`
implementation_code_commit: `5d1b316adbd482a4be7c2a4938f92c611e6348ae`
predecessor: IDEA-196 / EAAC rejected at Gate0; not a code parent

problem: The current role-window implementation first averages all 36 projected CLIP patches in a window and then learns an indirect 8x25 role-to-window attention. A small but decisive part can be diluted by the other 35 patches, while the attention allocation has no direct location supervision. Prior signed-action training also used a 4:4 row sampler that changed the natural train distribution.

hypothesis: For each candidate window, preserving every role's mean support, strongest Top2 support and strongest Top1 support is sufficient for a shared small network to predict the executable window outcome, while retaining small-part peaks that window averaging loses.

core_change: For patch `i` and role `k`, compute `s[i,k]=cos(patch_i,Top2_role_k)-cos(patch_i,Top1_role_k)`. Each of 25 fixed 6x6 windows emits `8 roles x [mean,max,min]=24` evidence values. Append 8 fixed position values, Parent Top1-Top2 margin and Parent entropy, yielding 34 values per action. A shared `34->64->3` MLP predicts `damage/neutral/correction` for every action. The action score is `P(correction)-P(damage)`; execute only when the maximum is positive. The zero output head initially abstains.

target_and_sampling: Every dev-train row keeps all 25 executable labels. Parent-correct plus crop-wrong is damage; Parent-wrong with truth=Top2 plus crop-correct is correction; unchanged correctness and truth outside Top2 are neutral. Training samples rows uniformly from the natural 4,702-row distribution, uses all 25 labels, plain cross-entropy, no class weights and no 4:4 oversampling.

unique_change: Replace 36-patch window mean plus learned role-to-window attention and indirect action supervision with direct per-window 24-dimensional role evidence and three-state executable supervision. Prompts, Top1/Top2 parent, 25 windows, crop geometry, fixed verifier, seed, update budget and B<=1 remain unchanged.

minimal_falsification: Train only Full for the fixed seed7/1000-update Gate0 contract. Full must exceed Parent and same-checkpoint S/V/I-off by at least 1.0 point with paired class-bootstrap lower bound above zero, beat triggered Center/StaticBest/HashRandom/TextHeatmap by at least 0.5 point, achieve positive net corrections, retain both trigger and abstain, use at least two actions, and satisfy physical B<=1. Any hard gate failure drops the candidate; no Top3, prompt, threshold, class-weight, focal-loss, resampling, window or B2 rescue in this TRY.

current_advantage: none; this is an owner-approved performance-oriented supporting path, not a standalone innovation claim.
performance_status: below_parent
failure_boundary: Low-resolution projected patches may not reveal high-resolution crop details; max/min may amplify noisy patches; invisible roles may generate false peaks; natural sampling may collapse to neutral; truth outside Parent Top2 is unreachable; the fixed class-name crop verifier may discard role information used for selection.
paper_level_claim: none before real Gate0 and required controls.

## V6 design and code review

review_subject_commit: `5d1b316adbd482a4be7c2a4938f92c611e6348ae`
review_subject_tree: `60cab901aa13359eab38f3a1cebe7e6e4936f42d`
review_agents: [`/root/rwdg_review_a`, `/root/rwdg_review_b`]
review_result: `Design P0=0/P1=0; Code P0=0/P1=0; 双Agent交叉审查通过`

- Both agents independently reviewed the V6 identity, RoleTriPool formula, tri-state targets, natural sampling, S/V/I-off, B1, assets, controls and receipts.
- A found one StaticBest P1: eval counted correction only while train selected correction-minus-damage. Commit `5d1b316` aligned eval to the natural 4,702x25 tri-state net histogram and added checkpoint-detail equality assertions. Both agents restricted recheck to the affected diff and returned `P0=0/P1=0`.
- They exchanged SHA-bound complete reports via shared temporary files, replied to the other report and jointly passed the development candidate. Local V6 tests: `18 passed`.

## 2026-09-02 Gate0 result

train_checkpoint: `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/role_tripool/V6-TRY-001-GATE0-FULL/role_tripool_gate0_full.pt@sha256:d68a27a7704beb4afd7246d9a6c91506809b112752d855fa4efe42534c9416a8`
eval_config_sha256: `b29e35659815491a65ecf9b67c885336e8b8e2f01fb6239071cbdf3d9d6f263f`
failure_receipt: `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/role_tripool/V6-TRY-001-GATE0-EVAL/failure.json@sha256:2af434be572587e16ce93456eb95b09ad8a9f0cfc45c95559e41aecfc7ff0ca5`

- Parent=`66.692923%`; Full=`66.554494%`, Full-Parent=`-0.138429pp`; Full-Soff=`+0.087403pp`; Full-Voff=`-0.094826pp`.
- Only22/2355 rows triggered and every triggered row selected action24; corrections6, damages9, net=-3. The natural tri-state objective converged to neutral-dominant near-total abstention and action collapse.
- All data/B1 boundaries passed. Direct no-crop pair classification using the same tri-pool evidence was also diagnosed under natural sampling and reached only`66.781195%` (`+0.088271pp`, CI crossing zero).

root_cause: Low-resolution role tri-pool evidence does not reliably identify the high-resolution corrective action. Natural damage/neutral/correction supervision makes abstention globally optimal; direct pair classification likewise carries too little transferable signal.

decision: Drop RoleTriPool. V6 remains a development line, not a formal framework. The next V6 candidate must use the only observed positive route: retain the explicit-abstention action policy but replace the fixed crop sign rule with a seen-trained crop safety verifier.


