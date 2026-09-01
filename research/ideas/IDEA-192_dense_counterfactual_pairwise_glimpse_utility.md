# IDEA-192：Dense Counterfactual Pairwise Glimpse Utility（D-CPGU，密集反事实二元高清观察效用）

status: revised_before_run_invalidated_by_owner_input_contract
idea_id: IDEA-192
source_type: experiment_result + diagnostic_result + first_principles + owner_hypothesis + nearest_work_boundary
method_name: Dense Counterfactual Pairwise Glimpse Utility
method_acronym: D-CPGU
current_run: none; pre-run code commit aa9eb4518a940a63341ae8fc24e2d2a1ac91b820 was never trained
problem: 当前2355-row诊断证明Top1/Top2高清crop具有+16.189pp oracle上限，但IDEA-172文本行动和CUAV expected-loss policy均失败；需要检验25维dense反事实效用能否在观察前预测有用动作并安全abstain。
hypothesis: 只用100 dev-seen生成每个样本25个crop动作的二元正确性向量，若模型能从全图name歧义预测该效用场，并在B≤1下安全abstain或获取一个高清crop完成固定keep/swap，则可在class-disjoint类别上超过Parent、普通selector/rerank及三项off。
core_change: 将学习对象从单动作标签或expected crop loss改为完整25维counterfactual correctness vector，并为pair-unreachable样本提供全零utility与固定0.5 abstain。
success_condition: 先通过本文100/50 whole-set Gate；后续正式Chen-style Full高于name-only计算父基线，且S/V/I同checkpoint module-off分别至少降低1.0 H。
failure_condition: Parent/Triggered controls/Sparse-action/Expected-loss/PairMLP/三off/content cycles/occupancy/B1任一硬门失败即drop；禁止B>1、多crop、阈值/几何/policy架构或PCLR救援。
evidence_refs:
  - research/ideas/IDEA-172_text_difference_active_evidence_acquisition.md
  - research/ideas/IDEA-191_counterfactual_utility_active_view.md
problem_category: visual_grounding
mechanism_tags: [dense_action_utility, pairwise_active_view, abstaining_glimpse, top2_falsification]
base_framework: FRAMEWORK-V5
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
gate_computational_parent: frozen OpenAI CLIP ViT-L/14@336 CLS × one canonical class-name prompt
formal_reference: FRAMEWORK-V5 H=81.06877662507551 is reported only; D-CPGU prohibits PCLR online inference
owner_decision: 2026-09-01 owner initially accepted D-CPGU, then clarified that the actual semantic input must be six part descriptions plus overall plus distinctive and that a real train-time attention mechanism was required. The name-only CLS+MLP implementation was therefore invalidated before server training. Its code/review evidence remains on `exp/v5/innovation/v5-try-006-dcpgu`; it is not a code parent. The changed input, representation and falsifiable hypothesis continue as IDEA-193 from the formal V5 parent.
reuse_refs: [IDEA-172, IDEA-191]

## Current evidence

On the current 2,355-row/150-class development Gate, name-only Parent macro Top-1=`66.695482%`. Under the exact IDEA-172 25-window geometry, a diagnostic Top1/Top2 PairCropOracle25 reaches`82.884902%` (`+16.189420pp`), with reachable coverage`83.566879%`, 384 parent-wrong true-challenger samples, 378 oracle corrections and zero damage. Center and train-StaticBest action12 both reach only`53.850198%`. This is a real instance-specific pairwise observation ceiling, not current method advantage.

CUAV is direct negative evidence: its all-class utility policy chose action4 on100% of eval images, Full=`66.119060%` below Parent, and Full=S-off=Image-only=StaticBest. D-CPGU therefore changes the learning signal and decision contract; it is not a CUAV rescue.

## Mandatory admission fields

old_solution_path: `one 336 global image → name-only all-class argmax`; IDEA-172 uses a static text heatmap for B=1 Top2 swap; CUAV learns expected all-class crop loss and collapses to StaticBest.

new_solution_path: `name-parent Top1/Top2 ambiguity → predict all 25 unseen-before-action counterfactual pair utilities → abstain or request exactly one original-resolution crop → fixed keep/swap decision → all non-pair classes untouched`.

principle_difference: The learning object is a dense 25-dimensional action-utility field, not one selected action label, expected all-class CE, text heatmap, or crop score. Every seen image supplies counterfactual supervision for every action under one frozen pair decision rule. Deployment predicts the utility field before observing any crop and may abstain; if it acts, it acquires one new raw observation.

old_signal_or_primitive: class point similarity, expected crop loss policy, opaque action score, or all-class crop fusion.

new_signal_or_primitive: `u*(x)∈{0,1}^25`, the frozen counterfactual correctness of each original-resolution crop action for the current name-parent leader/challenger pair, plus an all-zero unreachable state.

paradigm_shift: GZSL becomes abstaining active pair verification. The model forecasts which possible new observation can resolve the current pair before acquiring it, rather than passively scoring existing evidence.

why_not_module: The MLP architecture and crop executor are tools. If Sparse-action CE, CUAV Expected-loss, StaticBest, Image-only utility, no-crop PairMLP, or same-crop binary reranker matches Full, dense counterfactual utility is unnecessary and D-CPGU is only a crop selector/Top2 reranker.

closest_work_boundary: AdaptVision, CropVLM, recurrent visual attention and active visual categorization already cover active acquisition, zoom, glimpse policies and action-value learning. D-CPGU cannot claim any of those. Its only possible narrow distinction is dense seen-only crop-action correctness supervision for abstaining Top1/Top2 verification in class-disjoint GZSL under a strict at-most-B=1 budget.

GapSight is the closest boundary because it uses a target model's loss/performance gap to train when and where to request an additional crop: https://arxiv.org/abs/2608.21762. D-CPGU therefore cannot claim loss-gap or utility-guided cropping. The claim is further restricted to a fixed25-action, binary correctness utility vector for a frozen name-only Top1/Top2 pair, explicit outside-pair all-zero abstention, class-disjoint GZSL and B≤1.

## Exactly three deployment modules

### S — Pair Ambiguity State

Inputs: frozen class-name embeddings`T[C,768]`, normalized full-image CLS`z_full[768]`, and parent logits`L_parent=z_full T^T/0.07` over the active100/150/200 axis. Stable-sort`(-logit,global_id)` to get leader`l` and challenger`c`.

Output:
`q=normalize(T_l-T_c)`, `m0=L_l-L_c`, `entropy(L_parent)`, `mean/std(L_parent)`, leader/challenger IDs and parent logits.

S-off: V utility network receives only`z_full`; no top2/query/stat tensor is constructed. Class names remain available to the frozen Parent and final pair decision.

### V — Dense Utility Glimpse Policy and B1 Executor

Policy input is `[z_full,q,m0,entropy,mean,std]`. Shared network:

`h=GELU(W_z z_full+W_q q+W_s[m0,entropy,mean,std])`, `u_hat=sigmoid(W_u h)`.

`W_z/W_q:768→64`, `W_s:4→64`, `W_u:64→25`, bias-free; seed7 defaults for upstream weights and zero`W_u` initialization. Step1 requires nonzero`W_u` gradient and permits zero upstream gradients; after one optimizer step, step2 requires finite nonzero gradients for all four weights.

Crop actions inherit exact IDEA-172 geometry: 6×6 patches on the24×24 grid, starts`[0,4,9,14,18]`, row-major25, action-list SHA`4e64cb1fa0a24b3fd734d53dc60dadf94057bfadf36ff65fb0e0a063bfdb74cb`.

Training may precompute all25 original-resolution crop CLS only for100 dev-seen images. For each image/action, use the frozen I rule below to create targets:
- true label=leader: `u*_a=1` iff crop rule keeps leader;
- true label=challenger: `u*_a=1` iff crop rule swaps to challenger;
- true label outside pair: `u*_a=0` for all25.

Loss is elementwise BCE over all25 actions and all train images; no hard oracle action label is constructed. Save per-class/action positive density and all-zero rate.

Frozen eval predicts`u_hat` before raw-image open. If`max(u_hat)≤0.5`, abstain and keep Parent without crop. Otherwise choose smallest action ID among maxima, open the raw image once, execute exactly that crop, and run one frozen CLIP forward. Eval using labels to decide reachable/outside, encoding all25 before action, B>1 or threshold tuning is P0.

V-off: same predicted utility/action/abstention, but selected crop is taken from the already preprocessed336 tensor and encoded once; original-resolution raw path is not opened.

### I — Frozen Pair Keep/Swap Solver

For an executed crop:
`m_crop=(z_crop·T_l-z_crop·T_c)/0.07`.

Fixed decision: keep leader iff`m_crop≥0`; otherwise swap leader/challenger. Other class logits/order are untouched. If V abstains, keep Parent.

I-off: same Full utility field, abstention, selected action, crop and cost, but physically bypass crop pair decision and always keep Parent. This isolates whether the acquired crop is actually used.

Physical off receipts: S-off does not construct ambiguity; V-off does not open original raw path; I-off does not compute crop pair margin. Record opened keys and module call counts; calculate-then-discard is failure.

## Training and evaluation boundaries

Gate train:100 dev-seen classes/4,702 images. Frozen eval:50 dev-unseen classes/2,355 images under150 candidate axis. Official test absent; dev-unseen images/text never enter gradient, threshold, checkpoint or action selection.

Training labels may define leader/challenger/outside and `u*`; eval labels are used only after Full predictions freeze for metrics and offline diagnostic controls.

The `0.5` abstention threshold, crop margin threshold`0`, action tie-break and geometry are fixed before results and cannot be rescued.

## Hard controls

All deployment controls use at most one extra crop encoding:
1. Parent: keep leader, no crop.
2. Center / StaticBest / Random / TextHeatmap: fixed or heuristic B=1 actions with the same I rule; TextHeatmap reruns IDEA-172 algorithm on the same2,355 rows/150 axis.
3. Image-only utility: separately trained S-off network.
4. Low-res crop: Full utility/action but V-off observation.
5. I-off: Full action/crop, always keep Parent.
6. Sparse-action: use the same seen counterfactual utilities but collapse each vector to the smallest positive action ID and train a fixed26-way CE (`actions0..24`, `abstain index25`); all-zero samples target index25. Eval argmax25 abstains, otherwise executes the predicted action.
7. Expected-loss: exact CUAV policy expectation objective with the same pairwise I rule.
8. No-crop PairMLP: input`z_full/q/m0/stats`, directly predict keep/swap/abstain without crop.
9. Same-crop PairMLP: Full selected crop plus pair state into a shared binary MLP; if it matches, fixed I solver is unnecessary.
10. PairCropOracle25: labels/all25 only after Full freeze, ceiling only.

To isolate “whether to act” from “where to look”, frozen Full checkpoint also runs eval-only same-trigger controls. They reuse Full's exact `max(u_hat)>0.5` trigger/abstain decision and replace only the selected action:
- Triggered-Center uses action12;
- Triggered-StaticBest uses the train100 frozen StaticBest action;
- Triggered-Random uses `int(SHA256("seed7:" || relative_path || ":" || leader_global_id || ":" || challenger_global_id),16) mod25` and records the full mapping SHA;
- Triggered-TextHeatmap reruns IDEA-172 action on the same rows.

Full must exceed each triggered control by1pp with paired CI lower>0. Training these controls or changing the trigger is forbidden.

Content controls are unique:
- `eval action derangement`: frozen Full `u_hat/trigger`; only executor action becomes`(a+1)%25`.
- `utility-target cycle`: train-time retrained control. Within each true class, train images are repository-relative-path sorted and the complete25-vector`u*` is forward-cycled with no fixed point; image, pair state, crop features and labels remain. Mapping/SHA freeze before training. Outside all-zero vectors are reported separately because cycling them is inert.
- `crop-feature cycle`: frozen Full eval diagnostic. Take all triggered eval rows globally, require at least2, stable-sort by repository-relative path and forward-cycle with no fixed point. For row`j` with selected action`a_j`, replace its selected crop feature by the next row's feature for that same action`a_j`, read from the diagnostic all25 table. Full utility/action/pair/label remain; no label block is used and every triggered row is covered. This control is built only after Full checkpoint/actions freeze.
- `pair-name cycle`: frozen Full eval diagnostic. In separate sorted100-seen/50-unseen class maps, leader/challenger name embeddings are replaced by next-class embeddings only in V utility input; numeric parent stats and actual I pair remain. Mapping/SHA freeze before eval.

Every control writes the exact mapping array, ordering keys, no-fixed-point assertion and SHA. Train may adapt only to the explicitly train-time utility-target control.

## Gate

One seed7 proof Gate passes only if all hold:
- PairCropOracle25 on current rows is at least Parent+1pp;
- Full whole-set macro Top-1 ≥Parent+1pp;
- Full ≥Center/StaticBest/Random/TextHeatmap/Image-only/Low-res/I-off/Sparse-action/Expected-loss/No-crop PairMLP/Same-crop PairMLP by≥1pp;
- Full ≥Triggered-Center/Triggered-StaticBest/Triggered-Random/Triggered-TextHeatmap by≥1pp;
- S/V/I-off each≥1pp; every difference has paired50-class bootstrap95% lower bound>0;
- corrections-damages>0; report leader-correct/challenger-correct/outside group counts, trigger/abstain, correction/damage and utility density;
- highest action occupancy≤70%, at least10 actions selected among triggered rows;
- Full eval records`all25_eval_encoding_count=0`, selected crop forwards≤N_eval and action before raw open;
- derangement/cycles retain at most20% of positive gain.

Gate passing is proof only. Formal success later requires Chen-style Full above name-only computational parent and same-checkpoint`H_full-H_Soff/Voff/Ioff≥1pp`; H=80 is target, not pass line.

minimal_falsification: First use existing current-row oracle receipt, then train only Full and Image-only utility. Evaluate Parent, StaticBest, Center, Low-res, S/V/I-off, Sparse-action and No-crop PairMLP. If Full does not beat every displayed control by1pp with CI lower>0, net correction≤0, action collapses, or outside damage is uncontrolled, immediately drop before remaining controls. No B2, multi-crop, geometry/threshold change, entropy regularizer, PCLR or policy architecture search.

current_advantage: none. PairCropOracle25 is only a diagnostic ceiling.
performance_status: proof_of_path_not_run.

oracle_evidence_ref: `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/diagnostics/IDEA-192-pair-oracle/receipt.json@sha256:8fe3b8e20bbf49e1188b40fb9fc07d1e73d48548a435fc151f8895d0b8b30c8f`; binds current2,355 rows, 150 active axis, CUAV bundle/manifest SHAs, fixed geometry and source commit.

receipt_contract: Report leader-correct/challenger-correct/outside counts; per-group all-zero rate, positive-action density, trigger/abstain, correction/damage; max-utility tie rate with smallest-action-ID tie-break; action histogram/occupancy; B1 raw-open/forward counts; all control mapping SHAs. No-crop PairMLP tests whether crop acquisition is necessary; Same-crop PairMLP tests whether the fixed I solver is necessary; Sparse-action tests whether the dense25-vector supervision is necessary.

I_off_interpretation: I-off only proves that the acquired crop is used by the fixed keep/swap solver. Crop acquisition necessity is established jointly by No-crop PairMLP, Low-res and same-trigger controls; I-off alone cannot support that claim.

failure_boundary: dense utilities may be unlearnable before observation; outside all-zero targets may dominate abstention; policy may collapse to StaticBest; no-crop PairMLP may predict swap equally well; fixed crop margin may damage parent-correct samples. Any hard-control match or module-off below1pp immediately rejects D-CPGU.

paper_level_claim: Only after proof Gate, formal H, module-off and multi-seed evidence: “Dense seen-only counterfactual crop utilities enable abstaining B=1 Top1/Top2 verification in class-disjoint GZSL.” No first-active-vision, first-Q-learning, first-crop-selector or first-binary-rerank claim.

## 范式Idea双Agent对抗定稿记录

review_date: 2026-09-01
review_agents: [`/root/idea189_a`, `/root/idea189_b`]
review_subject_sha256: `3fa7122055b9b35e7eba1f67a68bae837c27cf138ff03a8b89a3f3d66d88fd45`
review_status: passed_for_proof_of_path_idea_only

- 前置论证先否定“直接把IDEA-172旧500-row oracle包装为方案5”。主Agent随后在当前2,355 rows/150轴重算PairCropOracle25，得到Parent=`66.695482%`、Oracle=`82.884902%`、gain=`+16.189420pp`、378 corrections/0 damage，并冻结诊断receipt SHA。
- 第一准确草稿SHA256=`17fe70de72b2259503c697750f6d4c01574b5f6804e370fe7495f7536264df12`。双方独立审查与直接交叉判REVISE；问题为outside安全、I-off公平、Triggered controls、cycles与GapSight边界。
- 第一集中修订SHA256=`54790dbef55ffa99b89790e7e66b1574b254caeede359d9f6573e17671073a4b`。双方继续发现crop-feature cycle singleton、Sparse-action维度和Random/I-off解释问题。
- 最终准确审核对象为上述`review_subject_sha256`。两名Agent分别完整读取后均给出`P0=0 / P1=0 / P2=0 / PASS`，随后交换完整终审清单并确认无补充、无异议、无遗漏。
- 最终共同结论：**范式Idea双Agent对抗审核通过**。
- 该通过只证明D-CPGU具备可证伪proof-of-path资格，不代表Gate成立、Innovation晋级或论文claim成立。任一普通Q/selector/reranker控制追平、outside安全失败、B1违规或三off不足1均必须drop。
