# IDEA-189：Role-Contrast Evidence Gain（RCEG，角色对比证据增益）

status: rejected_at_proof_gate
idea_id: IDEA-189
source_type: experiment_result + code_analysis + first_principles + owner_hypothesis + nearest_work_boundary
method_name: Role-Contrast Evidence Gain
method_acronym: RCEG
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
current_run: V5-TRY-003-G0
problem: CEC证明绝对补全误差可以下降却不改变类别决策；需要检验6+1+1可见角色描述相对同候选类别名，是否真正增加对当前图像隐藏局部证据的解释力。
hypothesis: 若丰富角色描述携带可迁移、可观察的类别信息，则同一共享预测器在角色条件下应比该候选自己的类别名条件更准确预测隐藏patch目标，且这一候选特异增益应在class-disjoint类别上超过真类难错类、父基线和三项module-off。
core_change: 用候选特异的name-only预测误差作为reference、6+1+1角色预测误差作为enriched hypothesis，学习并使用二者的嵌套证据增益，而不是绝对补全误差或文本直接重排。
success_condition: 先通过本文100/50 proof Gate；后续正式Chen-style同checkpoint评估中Full高于声明的name-only计算父基线，且S/V/I三项module-off分别至少降低1.0 H。
failure_condition: 任一方向门、shuffle、module-off、Absolute-role、Reference-difficulty或Target-free控制失败即drop；不得靠事后gamma、scale、mask、target层或prompt调整救活。
evidence_refs:
  - research/ideas/IDEA-104_rgwps.md
  - research/ideas/IDEA-105_crgwps.md
  - research/ideas/IDEA-110_pdrs.md
  - research/ideas/IDEA-171_hypothesis_conditioned_visual_completion.md
  - research/ideas/IDEA-188_cec.md
problem_category: visual_grounding
mechanism_tags: [name_anchored_role_gain, rolewise_rival_contrast, masked_visual_prediction, hypothesis_test]
base_framework: FRAMEWORK-V5
git_base_commit_candidate: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
git_parent: formal FRAMEWORK-V5 commit 52b511d77b4ad048f35b40dc3cbd9afd092167e9; this is code lineage only
gate_computational_parent: frozen OpenAI CLIP ViT-L/14@336 image CLS × the existing single canonical class-name prompt embedding, same split/checkpoint/calibration; no PCLR online inference
formal_reference: FRAMEWORK-V5@52b511d, method TG+GTD+PCLR-RSE, reported formal H=81.06877662507551; RCEG does not call PCLR online and does not inherit its logits
owner_decision: 2026-09-01 owner explicitly confirmed that scheme-2 proof and module gains use the declared name-only computational parent; FRAMEWORK-V5@52b511d and H=81.06877662507551 are reported as an additional formal benchmark; RCEG is prohibited from PCLR online inference
reuse_refs: [IDEA-104, IDEA-105, IDEA-110, IDEA-171, IDEA-188]

## 1. Problem and direct evidence

CEC/IDEA-188 showed that absolute completion can learn without becoming a decision signal: R2 completion error fell `2.0078→0.1208`, but Full=off, 1702 gate images had zero prediction changes, and unseen true-vs-hard-wrong direction was `47.14%`. IDEA-171/HCVC already proposed masked context plus class-conditioned target prediction and raw `-E_c` classification; RCEG cannot claim the first masked generative verifier.

Rich role text alone is also insufficient. IDEA-104 rejected directly concatenated eight-role differences; IDEA-105 only obtained tiny `+0.0248/+0.0620 H`; IDEA-110 rejected text-distance pair weighting. Therefore the role descriptions are only questions to be checked against hidden current-image evidence, not direct reranking scores.

## 2. Mandatory admission fields

old_solution_path: `frozen image/text representation → similarity/PCLR correction → calibrated argmax`; CEC/HCVC variants use absolute candidate completion energy.

new_solution_path: `name-only parent ambiguity → matched 6-part+overall+unique candidate-vs-rival questions → fixed masked visual context and original local patch target → the same shared predictor under name-only and role-rich conditions → candidate-specific role gain over its own name anchor → bounded 200-class update`.

principle_difference: The hypothesis test is nested and candidate-specific. For every candidate `c`, the reference hypothesis already knows `c` and its rival only through canonical class names, while the enriched hypothesis replaces those repeated name conditions with eight matched visible role descriptions. The score is whether the richer semantic hypothesis predicts hidden current-image local evidence better than the same candidate's name-only hypothesis. Unlike the old zero/null draft, both numerator and denominator vary with `c`, so the reference term cannot cancel as an imagewise constant.

old_signal_or_primitive: class point prototype, role-text direct score, PCLR relation score, or absolute completion error `E_role,c`.

new_signal_or_primitive: candidate-specific nested explanatory gain `G_c = mean_m log((E_name,c,m+1e-6)/(E_role,c,m+1e-6))` on original local patch-embedding targets hidden from the predictor context.

paradigm_shift: GZSL moves from “which class representation matches best?” to “for which candidate do role-specific visible claims add predictive information beyond that candidate's own name-only hypothesis?”

why_not_module: masks, projections and attention are implementation tools. If a same-interface monotone transform of absolute `E_role,c`, a target-free ranker, or IDEA-171-style absolute-energy training matches RCEG, then the nested gain is unnecessary and IDEA-189 is rejected as an HCVC/reranking variant.

non_equivalence_test:
1. `Absolute-role control`: same S/V architecture, same parameter budget, same base-logit/tanh/std interface and same hard negative, but replace `G_c` with imagewise standardized `A_c=-log(E_role,c+1e-6)` during training and inference. If it matches Full, nested gain is unnecessary.
2. `Reference-difficulty control`: keep the same final interface but use `D_c=+log(E_name,c+1e-6)` so candidates are rewarded only because their name-reference prediction is bad. If it matches Full, `G` is a denominator artifact.
3. `Target-free ranker`: use the exact S/V inputs and `W_g/W_e/W_s/W_o` parameter shapes, but never open/read `t_m`. Let `o_c,m=W_o mean_k(h_c,k,m)` before normalization and `R_c=-mean_m ||o_c,m||²`; standardize `R` across the same candidate axis and use `base+base_std*tanh(Z_R)`. Train from the same initialization/batches/optimizer/update budget with `CE + softplus(0.1-(R_y-R_n))`. If it matches Full, hidden-target learning is unnecessary.
4. `Name identity`: replace the eight role conditions/queries by the name-only condition/query; this must make `E_role,c==E_name,c` and `G_c==0` within `atol=rtol=1e-6`.
5. `Same-class target shuffle`: training context/text stay fixed but target comes from a different same-class image with the same mask. Retaining more than 20% of Full's positive gain means class-average memory rather than instance verification.
6. `Role-text block shuffle`: within the 100 train and 50 eval class blocks separately, move all eight sentences by a fixed no-fixed-point cycle while keeping images/rivals/masks unchanged; it must destroy the gain.

minimal_viability: On the CUB 100/50 development split, dev-unseen images/text are excluded from all train losses and checkpoint choice. On frozen dev-unseen evaluation: both macro class rates `G_true>G_hardwrong` and `E_role,true<E_role,hardwrong` are at least 60% with class-bootstrap 95% lower bounds >50%; Full macro Top-1 exceeds Parent, S-off, V-off and I-off by at least `+1.0pp`, each class-bootstrap lower bound >0, and net corrections are positive; Full exceeds Absolute-role, Reference-difficulty and Target-free by at least `+0.5pp` with lower bounds >0; shuffles fail. This is proof-of-path only.

minimal_falsification: Run only one seed and the fixed four-mask 100/50 Gate. First test gain direction, absolute role-energy direction and shuffles; only if they pass compute task improvements. Any direction failure, zero prediction changes, module-off gap <1.0pp, control match, or shuffle retention immediately drops IDEA-189; no post-hoc gamma, scale, mask, target layer or prompt changes.

current_advantage: none; Full相对name-only Parent为`-10.601819pp`，而Target-free相对Parent为`+2.016324pp`。
performance_status: rejected_below_parent_and_target_free_control_dominates.

failure_boundary: Direct `conv1` patch targets may be too low-level for role text; LLM role facts may be wrong or unobservable; four masks may hide decisive context; rich-role and name-only predictors may differ only by text scale; the absolute-energy control may match; four masked forwards may be too costly. Any such outcome blocks Innovation admission.

paper_level_claim: Only after the proof Gate, formal Chen-style GZSL, multi-seed evidence and all three formal same-checkpoint `H_full-H_module_off>=1.0pp` gates pass: “Role-specific descriptions are evaluated by their incremental masked-evidence prediction over a name-only class hypothesis in GZSL.” No “first” claim.

## 3. Exactly three deployment modules

### S — Rolewise Contrast Questions

Name anchor input:
- Source: `att_splits.mat/allclasses_names`; no attribute matrix is opened.
- Exact string: existing `clean_class_name`, then one prompt `a photo of a {clean_class_name}.`.
- Tensor: frozen CLIP FP32 `[C,768]`, L2-normalized per class.

Full role input:
- Source: `/data/lby/projects/cv_project/GZSL_Warehouse/assets/texts/CUB/text-v2-bd935b8a4ed42d59/role_texts.json`.
- Verified SHA256: `bd935b8a4ed42d59c3a39c3f30bb99552c717ef18dadbf3349422b1cef728985`.
- Generator disclosure: Codex sub-agent, `clip_anchored_class_specific_eight_role_descriptions_v2`; `llm_world_knowledge_used=true`, `expert_attributes_used=false`; display-name spellings include owner audit corrections but no CUB attribute answers.
- Role order: `[beak, head_features, body_plumage, wings, tail, legs, overall_appearance, unique_discriminative_features]`.
- Tensor: frozen CLIP FP32 `[C,8,768]`, each sentence L2-normalized.

Verified actual candidate/rival example (verbatim):
- Laysan Albatross: `A photo of a Laysan Albatross, showing a long pink hooked bill with a dark tip.`; `A photo of a Laysan Albatross, showing a white head with a dusky eye patch.`; `A photo of a Laysan Albatross, showing a white body beneath a dark mantle.`; `A photo of a Laysan Albatross, showing very long narrow wings with dark upper surfaces.`; `A photo of a Laysan Albatross, showing a short wedge-shaped tail.`; `A photo of a Laysan Albatross, showing pale pink webbed feet.`; `A photo of a Laysan Albatross, showing a huge white-and-dark albatross with gliding wings.`; `A photo of a Laysan Albatross, showing a white head and body, dusky eye patch, and long pink bill.`
- Sooty Albatross: `A photo of a Sooty Albatross, showing a dark hooked bill edged by a yellow groove.`; `A photo of a Sooty Albatross, showing a smoky-brown head with a pale facial crescent.`; `A photo of a Sooty Albatross, showing uniform soot-brown body plumage.`; `A photo of a Sooty Albatross, showing extremely long narrow brown-black wings.`; `A photo of a Sooty Albatross, showing a dark wedge-shaped tail.`; `A photo of a Sooty Albatross, showing pale-gray webbed feet.`; `A photo of a Sooty Albatross, showing a slender uniformly dark albatross with sweeping wings.`; `A photo of a Sooty Albatross, showing smoky-brown plumage, pale facial crescents, and a yellow-edged dark bill.`

Rival algorithm is unique. Stable-sort the active candidate axis by `(-parent_logit, global_class_id)`. For each candidate `c`, `r(c)` is the first sorted class whose global ID differs from `c`: parent Top-2 if `c` is Top-1, otherwise parent Top-1. Train axis is exactly 100 dev-seen classes; frozen Gate evaluation axis is all 150 active development classes; future formal axis is all 200 classes. No Top-K pruning is used.

Action: for each role `k`, output `q_role,c,k=normalize(role_c,k-role_r(c),k)` and the ordered pair `(role_c,k,role_r,k)`. The name reference outputs `q_name,c=normalize(name_c-name_r(c))` and repeats the ordered name pair across eight role slots. No learnable per-class parameter or online text generation exists.

S-off: feed the repeated name-reference pair/query into both the enriched and reference paths, preserving shapes and checkpoint; by contract `G=0`.

### V — Masked Role Evidence Field

Input: raw normalized 336×336 image, S queries, frozen OpenAI CLIP ViT-L/14@336.

Masks: the 24×24 patch grid is split by `(row mod 2, column mod 2)` into four fixed interleaved groups of 144 target patches. Each view replaces exactly one group's normalized pixels by zero before `visual.conv1`, leaving 432 visible patches. Masks never inspect image content or text.

Target identity: run only frozen `visual.conv1` on the original unmasked normalized image. For patch `i`, `z_i=normalize(conv1_patch_i)` in width 1024; no positional embedding, transformer, `ln_post`, `visual.proj`, full-image teacher or learned target projection is used. For mask `m`, target `t_m=normalize(mean_{i in m} z_i)` is FP32 `[1024]` and stop-gradient. “Hidden” means hidden from the predictor context, not from the evaluation system.

Visible context: one frozen masked CLIP forward per mask. Read masked-view CLS `g_m[768]` and only the 432 non-target final projected tokens `P_m[432,768]`; target-position student tokens are inaccessible.

Role evidence has a fixed, parameter-free formula with temperature `0.07`:
`a+ = softmax(P_m q / 0.07)`, `a- = softmax(-P_m q / 0.07)`, `v+=sum a+P_m`, `v-=sum a-P_m`, `e=v+-v-`.
Compute this for eight role queries in the enriched path and for the repeated name query in the reference path.

Output per image/mask/candidate: `g_m[768]`, `e_role[8,768]`, `e_name[8,768]`, `t_m[1024]`, mask ID.

V-off: set both `e_role` and `e_name` to zero while preserving the same masked CLS, target, shapes, S conditions, checkpoint and scorer. This is explicitly the “role-conditioned local token field off” control, not an equal-information claim.

### I — Name-Anchored Role Gain Test

One role-shared predictor is used for both paths. Fixed hidden width is 64. `W_g:768→64`, `W_e:768→64`, `W_s:2304→64`, and `W_o:64→1024` are bias-free; mask embedding is `[4,64]`. For each role slot:
`u=W_g g_m`, `v=W_e e_c,k,m`, `s=W_s[semantic_c,k;semantic_r,k;q_c,k]`, `h=GELU(u+v+s+v*s+mask_embed_m)`. Mean the eight `h` values and L2-normalize `W_o mean(h)` to predict `t_m`. Reference and enriched paths share every predictor weight and differ only in S/V declared inputs.

Errors and gain:
`E_name,c,m=||t_m-pred_name,c,m||²`, `E_role,c,m=||t_m-pred_role,c,m||²`, `G_c=mean_m log((E_name,c,m+1e-6)/(E_role,c,m+1e-6))`.

Final interface is fixed. Over the active candidate axis, `Z_G=(G-mean(G))/sqrt(var(G)+1e-6)`, `base_std=sqrt(var(base_logits)+1e-6)`, and `final_logits=base_logits+base_std*tanh(Z_G)`. There is no learned/tuned alpha (`alpha=1`), no gate-selected gamma and no post-hoc scale grid; all conditions reuse the parent's frozen calibration.

Training uses only 100 dev-seen images/text candidates. For each image label `y`, hard wrong `n` is the highest parent-logit wrong class under the same stable 100-class order. Loss is `CE(final_logits,y) + softplus(0.1-(G_y-G_n)) + 0.5*(softplus(-G_y)+softplus(G_n))`. Margin is fixed `0.1`; all three displayed coefficients are fixed. The 50 dev-unseen role/name texts are absent from the training manifest and first loaded only by the frozen evaluation process.

I-off: keep S/V/predictor and compute `A_c=-log(E_role,c+1e-6)`, `Z_A=(A-mean(A))/sqrt(var(A)+1e-6)`, then `off_logits=base_logits+base_std*tanh(Z_A)`. Thus base logits, std, tanh, candidate axis and checkpoint are identical; only nested name-vs-role gain is replaced by absolute role energy.

Output: all-candidate `E_name`, `E_role`, `G`, final logits and diagnostics.

## 4. Data, proof Gate and formal boundary

- Gate training manifest: CUB xlsa17 100 dev-seen classes/4,702 images, 100 name rows and 100×8 role rows only.
- Frozen Gate eval manifest: 150 active name/role rows plus 50 dev-unseen classes/2,355 images; loaded only after training and checkpoint freeze. Official test-unseen images/text are absent.
- CLIP checkpoint/preprocessing/masks are identical across Parent/Full/off/controls. No teacher network, distillation, CUB attributes, human part/box labels, PCLR online inference or PCLR relation assets.
- Parent/Full/S-off/V-off/I-off use the same checkpoint and parent calibration. Separately trained Absolute-role, Reference-difficulty and Target-free controls clone the same initialization, batches, optimizer and update budget. Absolute-role still computes the name path through the shared weights for diagnostics but its loss/score only use `E_role`; Reference-difficulty uses only `E_name`; Target-free is defined in non-equivalence test 3 and is physically unable to open target tensors.
- Every Parent/Full/off/control macro Top-1 difference uses the same 50-class vector and one pre-generated 10,000×50 class-bootstrap matrix; required `+1.0pp/+0.5pp` differences must also have 95% lower bound >0. Report correction, damage and net correction on the same images. Receipts also record per-axis distributions of `E_name/E_role`, `G`, axis mean/variance and `base_std` for train-100, Gate-150 and any formal-200 run.
- Gate passing only establishes proof-of-path. Scheme 2 is not “达标” until later formal Chen-style evaluation shows Full above the declared name-only formal computational parent and same-checkpoint `H_full-H_Soff`, `H_full-H_Voff`, `H_full-H_Ioff` are each at least `+1.0pp`; `H=80` remains an overall target, not a pass line. The promoted V5 `H=81.0687766` is reported as a separate formal benchmark because the owner prohibited PCLR online inference in the new path. Owner confirmed this distinction on 2026-09-01.
- Git parent candidate is formal V5 `52b511d...`; CEC failure commits are never code parents. No branch, queue, implementation or Innovation status exists before review and owner confirmation.

## 5. Cost contract

CLIP-side cost per image is four masked forwards plus one `conv1` target extraction; it is independent of candidate count and may be frozen in an external cache. Candidate computation is the shared 64-D predictor vectorized over 100 train, 150 Gate-eval or 200 formal classes; no candidate repeats CLIP. Report cache wall time/bytes, training throughput, candidate-vectorized latency and peak memory.

## 6. Closest original work boundary (rechecked 2026-09-01)

- Diffusion Classifier, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.html — conditional generative error for zero-shot classification exists.
- Intriguing Properties of Generative Classifiers, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/file/3ba4d47a83e498c2b1a0868cba20f6de-Paper-Conference.pdf — reconstruct-under-class generative classification exists.
- RONIN, WACV 2026: https://openaccess.thecvf.com/content/WACV2026/html/Nguyen_Detecting_Out-of-Distribution_Objects_through_Class-Conditioned_Inpainting_WACV_2026_paper.html — class-conditioned inpainting consistency verifies detector hypotheses.
- RILS, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Yang_RILS_Masked_Visual_Reconstruction_in_Language_Semantic_Space_CVPR_2023_paper.html — language-semantic masked reconstruction exists.

closest_paradigm_work: Diffusion/Generative Classifiers, RONIN and local IDEA-171. The only narrow candidate distinction is the candidate-specific nested comparison “role-rich hypothesis versus the same class/rival name-only hypothesis” on directly observed, predictor-hidden current-image patch embeddings. This distinction is unproven and is rejected if the Absolute-role control matches.

## 7. 2026-09-01 Gate 0真实结果

- 运行commit：`7aea59e4076b2984f79090a015ba04a2114f26ae`；eval config SHA256：`eafc7a752de2ff8e3460b56beef49f3c0b1bf31bc94031ca3b3de2a346692351`。
- 资产bundle：`98f06c47e3d9fda4f698aca5de5d4a33292e507de10e523a36303cea93beb54f`；训练4,702张100类dev-seen，冻结评估2,355张50类dev-unseen；official test未加载。
- dev-unseen 150类联合竞争macro Top-1：Parent=`66.695482%`，Full=`56.093657%`，S-off=`66.695482%`，V-off=`50.304598%`，I-off=`56.248069%`。
- 强控制：Absolute-role=`55.280948%`，Reference-difficulty=`54.422265%`，Target-free=`68.711805%`。Target-free比Parent高`2.016324pp`，并比Full高`12.618150pp`；隐藏target不是收益必要来源，核心非等价反例成立。
- 方向门：`G_true>G_hardwrong` macro=`58.806753%`、95% CI=`[54.370809,63.197996]`，未达到60%；`E_role,true<E_role,hardwrong` macro=`56.627830%`、95% CI=`[52.395999,60.793346]`，同样未达到60%。
- Full相对Parent纠正169、损坏422、净纠正`-253`；虽然799张预测发生变化，但方向整体有害。
- 只有V-off差值门与两个shuffle破坏门通过；Parent、S-off、I-off、方向、净纠正及三个强控制门均失败，`gate_passed=false`。
- 失败收据：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/rceg/V5-TRY-003-GATE0/EVAL/failure.json@sha256:a7d96e3ffb729f0e6839727c25b1561928d0e74bd062ded869e5f82196f97c16`。
- 最终决策：按预注册`failure_condition`立即drop。禁止调gamma、scale、mask、target层或prompt救活RCEG；Target-free只能作为下一条底层路径的诊断证据，不能重包装成RCEG成功或范式创新。

## 8. 范式Idea双Agent对抗定稿记录

review_date: 2026-09-01
review_subject_sha256: `ea9e21244e1161b311b51c50923d7eb6819236677edabc3f7936c6553ada68bc`
review_agents: [`/root/idea189_a`, `/root/idea189_b`]
review_status: passed_for_proof_of_path_idea_only
owner_confirmation: 2026-09-01 owner确认name-only为方案2直接计算基线、FRAMEWORK-V5@52b511d及H=81.0687766只作额外formal benchmark、RCEG禁用PCLR在线推理。

- 第一轮准确草稿SHA256为`25dd8129c57900f96f810c5e999ab465d4944d9d8c60b1aeacc1c495319b431d`。两名Agent先独立审查，再交换完整清单并逐项回应；共同结论为`P0=0 / REVISE`。集中问题包括空null对候选是常数、可能退化为HCVC+Rank、target层/维度不清、100/50文本轴、off接口、rival算法和统计门。
- 主Agent只做集中修订：将空null改为候选自己的name-only reference；Full S固定为内容寻址的6部位＋overall＋unique八角色原文；target固定为无teacher的原图`visual.conv1` 1024维patch；补齐固定视觉证据公式、同接口off、Absolute-role、Reference-difficulty、Target-free、shuffle、绝对方向门、成本和统计合同。
- 中间修订版由同两名Agent完整预审，继续关闭Target-free实现定义、坏reference denominator、正式/计算父身份及真实八句证据。服务器原文SHA经只读复核为`bd935b8a4ed42d59c3a39c3f30bb99552c717ef18dadbf3349422b1cef728985`。
- 最终准确审核对象为上述`review_subject_sha256`。两名Agent分别完整读取后均给出`P0=0 / P1=0 / P2=0 / PASS`，随后交换完整终审清单并各自直接回应；双方均确认无补充、无异议、无遗漏。
- 最终共同结论：**范式Idea双Agent对抗审核通过**。
- 该通过只证明RCEG在当前证据下具备可证伪的proof-of-path方法资格，不代表Gate成立、Innovation晋级、论文claim成立，也不授权创建分支、实现代码、进入实验队列或启动RUN。


