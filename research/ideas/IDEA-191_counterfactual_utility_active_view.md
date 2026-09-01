# IDEA-191：Counterfactual Utility Active View（CUAV，反事实效用主动视图）

status: proposed_owner_confirmed_proof_of_path_candidate
idea_id: IDEA-191
source_type: experiment_result + code_analysis + first_principles + owner_hypothesis + nearest_work_boundary
method_name: Counterfactual Utility Active View
method_acronym: CUAV
current_run: none
problem: IDEA-172证明原始高分辨率crop存在巨大oracle上限但静态文本行动失败，OREF进一步否定低分辨率局部token；需要检验seen-only name歧义策略能否在严格B=1下主动请求一个真正有用的新观察。
hypothesis: 只用100 dev-seen标签对25个固定crop的反事实效用训练离散policy，若能在冻结class-disjoint类别上先选动作后只编码一个原图高清crop，并胜过全部同成本固定/图像-only/低分辨率控制，则主动观察路径成立。
core_change: 将“下一条视觉输入是什么”变成训练对象；部署先从低分辨率name歧义选择一个离散动作，再获取一个原图高清crop并固定更新全类logits。
success_condition: 先通过本文100/50 B=1 proof Gate；后续正式Chen-style Full高于name-only计算父基线，且S/V/I同checkpoint module-off分别至少降低1.0 H。
failure_condition: Parent/Center/StaticBest/Image-only/Low-resolution/TextHeatmap/Unrelated/三off/坍塌/成本任一硬门失败即drop；禁止B>1、多crop融合、改几何、文本heatmap/PCLR或policy架构搜索救活。
evidence_refs:
  - research/ideas/IDEA-160_full_resolution_concept_grounding.md
  - research/ideas/IDEA-172_text_difference_active_evidence_acquisition.md
  - research/ideas/IDEA-189_role_contrast_evidence_gain.md
  - research/ideas/IDEA-190_observable_role_entailment_field.md
problem_category: visual_grounding
mechanism_tags: [active_visual_acquisition, counterfactual_crop_utility, class_ambiguity_policy, single_glimpse]
base_framework: FRAMEWORK-V5
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
gate_computational_parent: frozen OpenAI CLIP ViT-L/14@336 CLS × one canonical class-name prompt
formal_reference: FRAMEWORK-V5 H=81.06877662507551 is reported only; CUAV prohibits PCLR online inference
owner_decision: 2026-09-01 owner明确接纳IDEA-191/CUAV为方案4 proof-of-path候选，并授权继续执行尝试→失败→合规补救→经验总结闭环；授权不允许核心反例后扩大B、改crop几何或调融合救活。
reuse_refs: [IDEA-160, IDEA-172, IDEA-189, IDEA-190]

## Evidence status

IDEA-172 is negative-plus-ceiling evidence: on its fixed 25 original-resolution crops, Parent macro Top-1=`81.0%`, oracle=`95.0%`, but real TextDifference Active=`80.8%`, Center=`81.6%`, correction=8, damage=9. OREF then showed low-resolution local witness Full=`58.290949%` below Parent=`66.695476%`, while V-off global CLS=`67.082538%`. CUAV therefore does not claim crop policy success; it asks whether seen-only counterfactual utility learning can turn the known high-resolution observation ceiling into a real B=1 action without using eval oracle labels.

## Mandatory admission fields

old_solution_path: `one fixed 336 global image → name-only similarity → all-class argmax`; IDEA-172 adds a hand-designed text-difference crop action; OREF reuses low-resolution patch tokens.

new_solution_path: `low-resolution global observation + name-parent class ambiguity → learned discrete action distribution over 25 fixed windows → request exactly one original-resolution crop → one additional frozen CLIP encoding → fixed all-class evidence update`.

principle_difference: The parent passively consumes a fixed input. CUAV learns which new raw observation to request before seeing it. Training treats the 25 seen-only crop outcomes as counterfactual action utilities; deployment executes only the selected action and obtains one new signal unavailable in the original 336 view.

old_signal_or_primitive: fixed global CLS, cached low-resolution patch, text heatmap, multi-crop average, or post-hoc crop oracle.

new_signal_or_primitive: discrete action `a∈{0..24}` selected from a pre-observation ambiguity state, plus the newly acquired original-resolution crop CLS `z_crop(a)`. The action is part of the inference state and has a same-cost B=1 contract.

paradigm_shift: GZSL changes from single-observation classification to one-step active perception. The system first diagnoses ambiguity, then chooses and acquires a new observation, then updates the hypothesis.

why_not_module: Crop networks and logit fusion are implementation tools. If Center, StaticBest, Random, IDEA-172 TextHeatmap, image-only policy, low-resolution crop, or any same-cost fixed action matches Full, learned ambiguity-conditioned acquisition is unnecessary and CUAV is only a crop selector/multi-view engineering trick.

closest_paradigm_work:
- AdaptVision already performs coarse-to-fine adaptive visual acquisition: https://arxiv.org/abs/2512.03794
- CropVLM already learns dynamic zoom without boxes: https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/html/Carvalho_CropVLM_Learning_to_Zoom_for_Fine-Grained_Vision-Language_Perception_CVPRW_2026_paper.html
- Recurrent Models of Visual Attention and end-to-end active visual categorization already learn glimpse policies: https://arxiv.org/abs/1406.6247 and https://doi.org/10.1109/TPAMI.2018.2840991

closest_work_conclusion: CUAV cannot claim active vision, zoom, crop policy, or fine-grained cropping itself. Its only possible narrow distinction is a class-disjoint GZSL proof that name-only parent ambiguity can train a seen-only B=1 discrete high-resolution observation policy that beats all same-cost crop controls and contributes independently in S/V/I off tests.

## Exactly three deployment modules

### S — Name-Ambiguity State Composer

Inputs:
1. frozen name embeddings `T[C,768]` from `allclasses_names → clean_class_name → "a photo of a {class name}."`;
2. low-resolution full-image normalized CLS `z_full[768]`;
3. parent logits `L_parent=z_full T^T /0.07` over the full active axis 100/150/200.

Stable-sort by `(-L_parent,global_class_id)`. Let leader/challenger be Top1/Top2. Output semantic ambiguity state:

`q=normalize(T_leader-T_challenger)`

`margin=L_top1-L_top2`

`entropy=-sum softmax(L_parent)_c log softmax(L_parent)_c`

`s=[q, margin, entropy, mean(L_parent), std(L_parent)]`.

No LLM role sentences, expert attributes, boxes, parts or PCLR assets enter CUAV.

S-off: policy receives `q=0` and the four logit statistics zeroed, while retaining `z_full`; classification still uses name embeddings. The loader records that policy ambiguity inputs were not computed/read.

### V — Counterfactual Glimpse Policy and Executor

Crop geometry is inherited exactly from IDEA-172 commit `1374835caf91a2ab1279a3f7c1c9c37bd9fe574f`: 24×24 CLIP patch grid, 6×6-patch windows, row/column starts `[0,4,9,14,18]`, ordered row-major into 25 actions `[(0,0),(0,4),(0,9),(0,14),(0,18),...,(18,18)]`. Compact-JSON action-list SHA256=`4e64cb1fa0a24b3fd734d53dc60dadf94057bfadf36ff65fb0e0a063bfdb74cb`. `raw_crop()` maps those windows back to the original RGB image before resize. Asset generation must store, for every image/action, the resulting original-pixel box; the manifest records the complete box tensor SHA and at least one 25-box sample. Changing window size/starts/order after results is forbidden.

Policy network:

`h=GELU(W_z z_full + W_q q + W_s[margin,entropy,mean,std])`

`policy_logits=W_a h`, with bias-free `W_z/W_q:768→64`, `W_s:4→64`, `W_a:64→25`; all parameters are shared, no class/action-specific table.

Initialization is fixed: seed7 default PyTorch initialization for `W_z/W_q/W_s`, and exact zero initialization for `W_a`, giving an initially uniform 25-action policy. Two-step gradient receipt: step1 requires nonzero finite `W_a` gradient and permits zero upstream gradients; after one optimizer step, step2 requires nonzero finite gradients for `W_z/W_q/W_s/W_a`. Initial/final state SHAs are saved.

Training may precompute all 25 original-resolution crop CLS only for 100 dev-seen images. Deployment/frozen eval first computes policy from `z_full,s`, chooses `a*=argmax policy_logits`, then opens the raw image and performs exactly one extra frozen CLIP crop forward. It is P0 if eval encodes all25 before action, uses dev-unseen labels to choose action, uses B>1, or selects after seeing crop features.

Output: action distribution, selected action ID, fixed crop geometry, newly acquired normalized `z_crop(a*)[768]`, policy entropy and action-occupancy receipt.

V-off: policy still chooses `a*`, but the executor is forbidden to open the original-resolution image. It applies the same window to the already resized 336×336 input, resizes that low-information crop to336 and runs the same one extra CLIP forward. This is high-resolution-new-signal off, not no-visual.

### I — Fixed Crop Evidence Update

For every training action or the one selected deployment action:

`L_crop(a)=z_crop(a) T^T /0.07`

`D(a)=standardize_axis(L_crop(a)-L_parent)`

`L_final(a)=L_parent+std_axis(L_parent)*tanh(D(a))`.

For any vector`x` over the current active 100/150/200 class axis, `standardize_axis(x)=(x-mean_axis(x))/sqrt(var_axis(x,unbiased=false)+1e-6)` and `std_axis(L_parent)=sqrt(var_axis(L_parent,unbiased=false)+1e-6)`.

No learned alpha, gamma, Top-K slice or class-specific parameter exists. All 100/150/200 classes are updated.

Seen-only policy training computes every action's fixed loss without constructing a hard oracle label. Let `n` be the highest parent-logit wrong class:

`loss_a=CE(L_final(a),y)+softplus(0.1-(L_final(a)_y-L_final(a)_n))`

`pi=softmax(policy_logits)`

`L_policy=sum_a pi_a*loss_a`.

Only policy parameters update. Fixed seed7, 1,000 updates, AdamW lr1e-3/weight_decay1e-4, batch8, last checkpoint only. Fifty dev-unseen images/text and official test are absent from gradients/checkpoint selection.

I-off: use the same selected original-resolution crop and same checkpoint/policy. Keep identical `L_parent`, `std_axis(L_parent)`, `tanh`, active axis and crop cost, but replace the counterfactual relative evidence by absolute crop evidence:

`D_off=standardize_axis(L_crop)`

`L_off=L_parent+std_axis(L_parent)*tanh(D_off)`.

This is explicitly the non-counterfactual absolute-crop solver off path; it never computes `L_crop-L_parent`.

Physical off receipt:
- S-off policy call receives only `z_full`; no ambiguity tensor is created.
- V-off raw original path is never opened; only the preprocessed336 tensor is cropped.
- I-off never calls the standardized-delta solver.
- Module call counts and opened asset keys are written; calculate-then-discard is failure.

## Strong controls and same-cost contract

All B=1 controls use exactly one extra crop encoding and identical I update unless their purpose is to disable I:
1. Center: fixed action `(9,9)`.
2. StaticBest: one global action selected once using only 100 dev-seen mean action loss; frozen before eval.
3. Random: seed7 path-hash selects one action per image without labels.
4. TextHeatmap: exact IDEA-172 B=1 text-difference action.
5. Image-only policy: separately trained policy with `q/statistics=0`, same `z_full`, parameters/budget.
6. Low-resolution policy: Full policy action, but same V-off 336-derived crop observation.
7. Unrelated action: IDEA-172 same-cost unrelated semantic pair action.
8. All25 Oracle and All25 Average are high-cost ceiling/diagnostics only and can never satisfy a B=1 gate.

StaticBest action, Random mapping and all geometries are frozen before dev-unseen evaluation. Any policy/control action uses low-resolution state only; reading crop features before action is forbidden.

Control identity details:
- StaticBest computes each action's mean Full loss once on all 100 dev-seen training images, chooses the lowest-loss action with smaller action ID as tie-break, and records the 25 losses, selected ID, whether it equals Center action12, and SHA.
- TextHeatmap and Unrelated reuse only IDEA-172 algorithms/geometries; both are rerun on CUAV's exact2,355 eval rows, 150-class axis and macro definition. IDEA-172's old500-row numbers cannot enter current gates.
- Random uses seed7 SHA256(relative path) modulo25 and records the full image→action mapping SHA.

## Gate and statistics

Gate train: CUB xlsa17 100 dev-seen classes/4,702 images. Frozen eval: 50 dev-unseen classes/2,355 images under the 150 active name-only axis. Official test absent.

One seed7 proof Gate passes only if:
- Full macro Top-1 ≥ Parent +1.0pp;
- Full relative to Center, StaticBest, Random, TextHeatmap, Image-only policy, Low-resolution policy and Unrelated each ≥+1.0pp;
- Full relative to S-off, V-off and I-off each ≥+1.0pp;
- every difference uses one paired seed7 10,000×50 class-bootstrap matrix and has 95% lower bound >0;
- correction-damage >0;
- selected-action highest occupancy ≤70%, at least10 actions selected on dev-unseen, and Center occupancy alone cannot explain gains;
- action-map shuffle, crop-feature cross-image cycle and name-ambiguity shuffle retain at most20% of Full positive gain.

The three content controls are unique:
- `action-map shuffle`: eval-only破坏 control using the frozen Full checkpoint. Full policy normally chooses`a*`; only the eval executor is forced to run `perm(a*)=(a*+1) mod25`. Training never sees/adapts to this derangement. The 25-entry array and SHA are frozen before eval. A train+eval relabeled-action condition cannot replace this control.
- `name-ambiguity shuffle`: within the 100 dev-seen and 50 dev-unseen class blocks, class IDs are sorted and mapped to the next ID by a no-fixed-point forward cycle. For every image, replace leader/challenger name embeddings by `T_perm(leader)/T_perm(challenger)` and recompute `q`; retain the original parent-logit margin/entropy/mean/std, image and label. The two class maps and SHA are frozen before eval; training uses only the 100-class map.
- `crop-feature cross-image cycle`: within each class and action, images are stable-sorted by repository-relative path and forward-cycled with no fixed point; `z_crop(a)` is replaced by the next same-class image's same-action feature while policy input/action/label remain. Every class must have at least2 images. Dev-unseen labels are used only offline to construct this diagnostic mapping after Full checkpoint freeze; they never enter training, checkpoint selection, Full action or Full logits. Mapping arrays and SHA are frozen before the diagnostic.

Gate passing is proof only. Formal success later requires Chen-style Full above the declared name-only computational parent and same-checkpoint `H_full-H_Soff/Voff/Ioff≥1.0pp`; H=80 remains a target, not a pass line.

minimal_falsification: First revalidate IDEA-172 25-action oracle on the exact rows/geometries. Then train only Full and Image-only policy and evaluate Parent, Center, StaticBest, Low-resolution, S/V/I-off. If Full does not beat Parent and every displayed control by+1pp with CI lower>0, net correction is nonpositive, or policy collapses to StaticBest/Center, immediately drop CUAV before Random/TextHeatmap/Unrelated/shuffles. No B=2, multi-crop fusion, crop-geometry change, text heatmap rescue, PCLR or policy-architecture search.

current_advantage: none. IDEA-172 oracle is only a ceiling; its real Active result is negative.
performance_status: proof_of_path_not_run.

failure_boundary: Seen action utility may learn center/body/background priors; name ambiguity may not localize morphology; expected-loss policy may be an ordinary selector; original high-resolution crop may distract; fixed I update may damage parent-correct images. Center/StaticBest/Image-only/Low-res/TextHeatmap/Unrelated matching Full, eval needing all25, or any module-off below1pp immediately rejects CUAV.

receipt_contract: Save train/eval action histograms, per-image policy entropy distribution, highest-action occupancy, number of used actions, center-action occupancy, StaticBest action overlap, Full↔Center/Static action agreement, initial/final policy state SHAs, geometry/box SHAs, opened asset keys and module call counts. Full eval must separately record `raw_original_open_count=N_eval`, `selected_crop_forward_count=N_eval`, `all25_eval_encoding_count=0`, and one action decision per row made before raw-image open. Highest occupancy>70% or fewer than10 eval actions is a hard failure, not a tuning prompt.

paper_level_claim: Only after proof Gate, formal H, three module-off gates and multi-seed evidence: “A seen-only name-ambiguity policy actively requests one original-resolution crop to resolve class-disjoint GZSL uncertainty under a strict B=1 observation budget.” No first-active-vision, first-zoom, first-crop-policy or first-fine-grained-glimpse claim.

## 范式Idea双Agent对抗定稿记录

review_date: 2026-09-01
review_agents: [`/root/idea189_a`, `/root/idea189_b`]
review_subject_sha256: `a14521ff152ad992bcc772553ef04ddb096f8a15af99191ccd4cdc1c32b174d9`
review_status: passed_for_proof_of_path_idea_only

- 前置独立论证：Agent A提出seen-only 25-action高清观察policy；Agent B否定“crop selector本身就是范式”，双方直接交叉后把候选收窄为class-disjoint GZSL下的严格B=1反事实观察效用学习。
- 第一准确草稿SHA256=`7a9469ae2038394f48d3c09a7627690c0a4c1bd253f0ba8e7cce29a072bbc8c2`。双方独立审查和直接交叉判`REVISE`；问题为I-off口径不公平、shuffle自然语言不唯一、控制rows与policy收据不足。
- 第一集中修订SHA256=`7566fa49eb314cf6da6426004dc9797b492f3101e3ee36f4c4ff5cc70f666cc0`。双方复核后继续发现action permutation若train/eval共同使用会退化为动作重命名。
- 最终准确审核对象为上述`review_subject_sha256`。两名Agent分别完整读取后均给出`P0=0 / P1=0 / P2=0 / PASS`，随后交换完整终审清单并直接回应；双方无补充、无异议、无遗漏。
- 最终共同结论：**范式Idea双Agent对抗审核通过**。
- 该通过只证明CUAV具备可证伪proof-of-path资格，不代表Gate成立、Innovation晋级或论文claim成立。任一同成本控制追平、eval编码all25、policy坍塌或三module-off不足1都必须drop。
