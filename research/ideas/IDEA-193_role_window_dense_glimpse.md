# IDEA-193：Role-Window Dense Glimpse（RWDG，角色—窗口密集取证）

idea_id: IDEA-193
status: testing_owner_confirmed_method_candidate
source_type: owner_hypothesis + diagnostic_result + experiment_result + first_principles + nearest_work_boundary
problem_category: visual_grounding
mechanism_tags: [eight_role_text, role_to_window_attention, dense_counterfactual_action_supervision, one_shot_glimpse, pair_verification]
method_name: Role-Window Dense Glimpse
method_acronym: RWDG
base_framework: FRAMEWORK-V5
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
implementation_branch: exp/v5/innovation/v5-try-007-rwdg
current_run: V5-TRY-007 / Gate0 Full-only
implementation_code_commit: 0485445f1dcb83be201717a3c67cf31747782a53
train_config_sha256: df1b9bb6e9d9db49c186c420f266b31213b97f404f09b875e8a62142c7d62e20
reuse_refs: [IDEA-133, IDEA-172, IDEA-191, IDEA-192]
evidence_refs:
  - research/ideas/IDEA-133_visual_evidence_learning.md
  - research/ideas/IDEA-172_text_difference_active_evidence_acquisition.md
  - research/ideas/IDEA-191_counterfactual_utility_active_view.md
  - research/ideas/IDEA-192_dense_counterfactual_pairwise_glimpse_utility.md
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/diagnostics/IDEA-192-pair-oracle/receipt.json@sha256:8fe3b8e20bbf49e1188b40fb9fc07d1e73d48548a435fc151f8895d0b8b30c8f

problem: Current 2,355-row evidence contains more than 16 H points of Top1/Top2 high-resolution crop oracle headroom, but the pre-run IDEA-192 implementation reduced the policy input to one name-only CLS vector and omitted the owner-confirmed 6-part+overall+unique text as well as train-time attention. A correct candidate must learn which low-resolution window answers which visible role before paying for one high-resolution observation.

hypothesis: If every 100-class dev-seen image supervises the complete 25-dimensional vector of whether each crop makes a fixed Top1/Top2 verifier correct, then eight visible-role questions can learn transferable role-to-window attention, abstain outside the pair and acquire at most one useful high-resolution crop on class-disjoint dev classes.

core_change: Replace the CLS-only utility MLP with a complete 25-action pair-decision correctness field whose gradients train role-to-window attention over six part descriptions, one global description and one distinctive description. Outside-pair rows receive the all-zero vector.

success_condition: Gate 0 Full must exceed the exact projected-patch Parent by at least 1.0 H point and same-checkpoint S/V/I-off each by at least 1.0 H point, with paired class-bootstrap lower bounds above zero, while beating fixed triggered location controls by at least 0.5 point and satisfying B<=1. Gate 1 must beat CLS-only Dense, Sparse-26, Image-only Window Dense and No-Glimpse Pair by at least 0.5 point with lower bounds above zero. Formal success later uses the owner-selected Chen-style protocol; H=80 remains a target, not the Gate line.

failure_condition: Any Gate 0 parent/module/cost/collapse/net-correction condition fails, or any Gate 1 strong control matches Full. No rescue may change prompts, roles, window geometry, threshold 0.5, attention depth/head count, crop budget, add entropy loss, use PCLR online or introduce B>1/multi-crop fusion.

owner_decision: 2026-09-01 owner accepted the final v3-owner synthesis and authorized the continuing try->fail->rescue->experience loop. Owner also reduced the paper requirement: only the overall framework or at least one core point needs defensible novelty; the other modules may be established methods with accurate attribution. The performance/module contribution contract remains unchanged.

## Novelty and attribution boundary

The following are supporting techniques, not claimed contributions:

- eight-role CLIP text embeddings and Q/K/V cross-attention;
- coarse-to-fine or learned one-crop acquisition;
- the fixed selected-crop Top1/Top2 keep/swap rule.

Only possible core claim:

> Class-disjoint GZSL can train a role-conditioned one-shot verifier using the complete binary correctness field of all 25 crop actions, including an explicit all-zero outside-pair target.

This is deliberately narrow. It must be withdrawn if Sparse-26 or No-Glimpse Pair matches Full. It is not a claim of inventing attention, active vision, crop routing or utility supervision.

closest_work_boundary:

- AdaptVision, arXiv:2512.03794v3: low-resolution-to-crop adaptive acquisition and RL tool use.
- CropVLM, arXiv:2511.19820v2: learned single-crop zoom using downstream correctness/likelihood rewards without human boxes.
- GapSight / Learning to Look Again, arXiv:2608.21762v2: mines target-model answer-loss or option-margin crop gains and predicts review, expected utility and a free-form box from the global state.
- Local IDEA-133 Spatial-RGVE: existing patch attention; one-stage Visual removal contributed only +0.558258 H and role maps nearly overlapped.

RWDG can therefore only test the specific dense binary pair-correctness field + eight-role conditioning + class-disjoint GZSL combination.

## Exact assets

Text manifest:

- path: `/data/lby/projects/cv_project/GZSL_Warehouse/assets/clip_vitl14_336/CUB/69c9c6d82a755fe8/asset_manifest.json`
- manifest SHA256: `52c50c2f55250399bce360a218c30e70b66945953bec7e825e8dd8f20dddf91f`
- role source SHA256: `bd935b8a4ed42d59c3a39c3f30bb99552c717ef18dadbf3349422b1cef728985`
- role embedding SHA256: `f614a06cd93b071a4d8c7355f78a10588a0b954e46bc99c64b76399c8af5a889`, FP32 `[200,8,768]`, normalized
- name embedding SHA256: `c3a2f177f728621a56d1e972b91614346eee47a749e2902db0af33fac0543232`
- class-order SHA256: `7b6ffe26103bfeb73324f328fac499d6ea7cfadfb2b56448b0df295aca22df38`
- role order: `[beak, head_features, body_plumage, wings, tail, legs, overall_appearance, unique_discriminative_features]`

Projected-patch manifest:

- path: `/data/lby/projects/cv_project/GZSL_Warehouse/assets/rgve/CUB_openai_vitl14_336_projected_patch_final_v1/asset_manifest.json`
- manifest SHA256: `d096087c9bd37d90157688e21e79b8ba6a61f0ea9b1fa91f4f544f8bc1dd1ad0`
- CLS SHA256: `5c6e69fbfca4d41d73e133c6085e058e2c6f25237a34a7d00902c80b00b9db9a`
- patch SHA256: `937a906d18cc7acc556e75fe8b9822e47be8cc6b3d21c89e181a80a257940537`
- patch tensor: `[7057,576,768]`, float16, L2-normalized, 24x24 row-major, class token removed
- extraction: last visual transformer block 24 -> `ln_post` -> frozen `visual.proj` -> L2 normalization
- CLIP checkpoint SHA256: `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`

Gate bundle:

- bundle SHA256: `0b956bb4445033e14bb692dd725fbf894db1f8e2fc337d78cf4a1b3b63cd3450`
- dev_train SHA256: `0fc6df6b5babd8c8c2822f33ade9caa1fdad9284a424f028dd20991e7b50d20c`, 4,702 rows
- dev_eval SHA256: `2342392b5fb6f839c07a78922e2c3c59de63f68016e8d7cdbe9f6f11770d8af2`, 2,355 rows
- dev_eval_oracle SHA256: `2e4db3b918b7ea915e272d4266fd6241e0aa7624e9d5483a7ffdcbf9348b9fea`
- `raw_indices.pt` joins Gate rows to projected-patch `trainval_loc`; train/eval have zero overlap and cover all 7,057 rows.
- action geometry: 6x6 patches, starts `[0,4,9,14,18]`, row-major 25, SHA256 `4e64cb1fa0a24b3fd734d53dc60dadf94057bfadf36ff65fb0e0a063bfdb74cb`.

`projected_patch_pair_oracle_receipt: required_before_training`; before any optimizer step it must be replaced by a real path/SHA binding the exact projected-patch CLS, 2,355 rows, 150-axis, crop table, Parent, Oracle, gain, corrections and damage.

## Exactly three deployment modules

### S — Eight-Role Pair Questions

For Parent leader `l`, challenger `c` and role `k`:

`r_k=normalize(role[l,k]-role[c,k])`

`Q_k=LayerNorm(W_r r_k + W_n normalize(name_l-name_c) + role_id[k])`, giving `[B,8,64]`.

S-off does not open role embeddings; it repeats the name difference eight times through `W_r`, retaining role IDs, shapes, Parent and all downstream work.

### V — Role-to-Window Dense Utility

Pool each action's 36 projected patch tokens into `X_a`. The fixed normalized position is:

`p_a=[x/24,y/24,(x+6)/24,(y+6)/24,(x+3)/24,(y+3)/24,6/24,6/24]`.

`K_a=LayerNorm(W_x[X_a,X_a-z_full,p_a])`

`V_a=W_vx[X_a,X_a-z_full,p_a]`

`R_k=W_vr Q_k`

`score[k,a]=Q_k dot K_a/sqrt(64)`

`A[k,a]=softmax_a(score[k,:])`

`mass_a=sum_k A[k,a]/8`

`context_a=sum_k A[k,a]R_k/(sum_k A[k,a]+1e-6)`

The shared action-head input has 261 values: `[K_a,V_a,context_a,V_a*context_a,mass_a,leader_margin,entropy,logit_mean,logit_std]`.

`W_h=Linear(261,64,bias=False)`; `w_u=Linear(64,1,bias=False)`; `utility_logit[a]=w_u(GELU(W_h(input_a)))`.

All Full linear projections are bias-free. Every LayerNorm uses `elementwise_affine=True, eps=1e-5`. `w_u.weight` is zero-initialized; other weights use fixed seed-7 PyTorch defaults.

If `max(sigmoid(utility_logit))<=0.5`, abstain. Otherwise choose the smallest action ID among exact maxima, freeze it and only then open/encode one raw crop.

V-off does not open patch tokens; it broadcasts `z_full` to all 25 `X_a` slots, preserving positions, S, weights, threshold, action interface and possible B1 crop.

### I — Selected-Crop Pair Verifier

`m_crop=(z_crop dot name_l-z_crop dot name_c)/0.07`.

Keep leader for `m_crop>=0`; otherwise exchange only Parent Top1/Top2. I-off preserves Full attention/action/crop/cost but returns Parent logits.

## Training

Only 100 dev-seen classes/4,702 images enter gradients. For target construction only, their all25 frozen crop CLS are allowed:

- truth=leader: target 1 iff I keeps leader;
- truth=challenger: target 1 iff I swaps;
- truth outside pair: all 25 targets zero.

Loss: elementwise `BCEWithLogits` only. No attention target, part label, entropy/diversity loss, classification auxiliary, teacher or fusion weight.

Fixed Gate 0 budget: seed 7, AdamW, batch 8, 1,000 updates, lr `1e-3`, weight decay `1e-4`, final-update checkpoint. Step 1 requires nonzero finite `w_u.weight` gradient. After one optimizer step, replay requires independent finite nonzero gradients for `W_r,W_n,W_x,W_vx,W_vr,W_h,w_u`.

## Gate 0

All comparisons use one frozen 10,000x50 paired class-bootstrap matrix and require CI lower `>0`.

`module_contract_margin=1.0pp`; `support_control_margin=0.5pp`.

Evaluate Parent, Full, same-checkpoint S/V/I-off, Full-triggered Center, train-StaticBest, hash-Random and IDEA-172 TextHeatmap.

- `train-StaticBest=argmax_a mean_train(u*[:,a])`, tie smallest action ID.
- hash-Random action is SHA256 of `seed7|relative_path|leader_id|challenger_id` modulo 25; Full trigger is unchanged.
- TextHeatmap chooses the role with largest leader/challenger cosine distance, then the window with largest mean absolute patch response difference; ties use smallest ID and Full trigger is unchanged.

Hard pass:

- Full-Parent and Full-S/V/I-off each `>=1.0pp`;
- Full-triggered controls each `>=0.5pp`;
- corrections-damages `>0`;
- at least two actions, highest occupancy `<=70%`, both trigger and abstain;
- zero pre-action high-resolution/all25 opens, selected-crop forwards `<=trigger_count`.

## Gate 1 after Gate 0 only

Fixed separately trained controls:

1. CLS-only Dense: name/CLS/stats -> 25 BCE utilities.
2. Sparse-26: same S/V capacity, one 26-way action-or-abstain CE target.
3. Image-only Window Dense: same windows/positions/CLS/stats/BCE target, no role/name-pair query; hidden 128, zero output head.
4. No-Glimpse Pair: flatten Full's 25 `[K,V,context,V*context,mass]` tensors in action order, append stats, hidden 256, zero 3-way head; targets keep/swap/abstain; no raw/high-resolution crop.

Full must beat each by at least `0.5pp` with CI lower `>0`. A matching Sparse/No-Glimpse result withdraws the possible core contribution. Matching CLS/Image-only withdraws the necessity of the 6+1+1 role-window component.

## Information, cost and receipt contract

Eval may read only the already-paid 336 CLS and 576 patch tokens before action freeze. Raw image path, high-resolution crop and all25 crop table are prohibited before freeze; after freeze at most one crop CLIP forward is allowed.

The 5.9-GiB patch array is read-only memmap. Batch 8 gathers only eight rows and converts about 13.5 MiB of patch values to FP32 on device. Cached training does not load CLIP and must measure peak allocation; selected-crop eval loads CLIP once and records one forward per trigger, none per abstention.

Report group counts/density/all-zero rate, trigger/abstain, corrections/damage/net, histogram/highest occupancy, attention entropy/per-role mass/overlap, manifest/file/config/checkpoint/code SHAs, action/box SHA, opened keys, raw/forward counts, GPU/environment, peak memory and wall time.

## Falsification and claim

minimal_viability: Real seen-only micro-batch proves all listed gradients after the zero head moves, attention rows sum to one over 25 windows, nontrivial actions/trigger/abstain and at least one prediction change, with no pre-action crop read.

minimal_falsification: Gate 0 immediately rejects parent/module/control/cost/collapse/net failures. Gate 1 rejects the sole possible contribution claim if a strong control matches. No post-result architecture, threshold, prompt, role, window, entropy or crop-budget rescue is permitted.

current_advantage: none; the oracle is opportunity evidence only.
performance_status: proof_of_path_not_run

failure_boundary: Low-resolution tokens may not predict high-resolution crop usefulness; descriptions may be absent/wrong; attention may repeat RGVE overlap; outside all-zero rows may dominate; action may collapse; Sparse/No-Glimpse may match; the fixed verifier may damage Parent-correct rows; memory and crop forward add cost.

paper_level_claim: Only after Gate 0/1, formal Chen-style, same-checkpoint module-offs and multi-seed evidence: “A complete pair-decision crop-correctness field can supervise role-conditioned one-shot visual verification in class-disjoint GZSL.” No “first” claim.

## Three-round design review and owner resolution

review_date: 2026-09-01
review_agents: [`/root/idea189_a`, `/root/idea189_b`]
v0_sha256: `a6907cc86af7f034207c643fdfa21cca5738459aeb1b2a91f43fb51e25f38bdd`
v1_sha256: `e8b2c34e0468a5fcfe45f065e73d8639ab16498dfbafa0cd3365066341493e69`
v2_sha256: `8d22472a9984957a37f84bfc32b41d86c353481d6a3b503e82fea3f55aace5fb`
v3_owner_sha256: `876d864bf07ee404e39c1c3993d70afc793f1bceabcc5ac2a9a4ebb7393a0b67`
review_status: owner_accepted_after_three_round_cross_disagreement

- v0/v1 both returned REVISE and exchanged SHA-bound full reports/responses through shared temporary files because child-agent messaging was unavailable.
- In v2 independent review A reported one P1 (Full head shape) while B passed. In cross-response A accepted PASS while B accepted REVISE, so there is no common v2 signature and this card does not claim “双Agent对抗审核通过”.
- The main-agent v3 synthesis applied the stricter fix by freezing Full as `261->64->1`, linear biases, LayerNorm behavior and control-capacity intent.
- The owner explicitly accepted v3 and authorized implementation. This resolves the three-round decision gate but does not replace code review, the projected-patch oracle receipt or real Gate results.

## Gate0 code review

review_subject_commit: `0485445f1dcb83be201717a3c67cf31747782a53`
review_subject_tree: `6e4e731ad3d141e7e61fb233a826b1f2e96ecef4`
review_agents: [`/root/rwdg_review_a`, `/root/rwdg_review_b`]
review_result: `P0=0 / P1=0 / 双Agent交叉审查通过`

- Both fresh reviewers independently inspected frozen commit `8b760d4`, exchanged SHA-bound full reports directly through shared files, responded to the other report and concluded `P0=0/P1=0`.
- The main agent then fixed only the affected asset/receipt boundary diff. Both reviewers restricted their recheck to `8b760d4..0485445` and independently reported `P0=0/P1=0 / 受影响diff复核通过`.
- Local evidence: RWDG runner/model tests `17 passed`; affected runner/model subset `12 passed`; data subset `5 passed`; formal patch manifest helper passed; the 5.9-GiB patch file was independently rehashed to the registered SHA `937a906d18cc7acc556e75fe8b9822e47be8cc6b3d21c89e181a80a257940537`.
- Train config SHA is fixed above. Eval config is a schema-reviewed template; only checkpoint path/SHA/training commit are filled after training as a pure identity update. GPU/environment fingerprint remains a required runtime receipt field.
