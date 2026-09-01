# IDEA-190：Observable Role Entailment Field（OREF，可观察角色蕴含场）

status: proposed_owner_confirmed_proof_of_path_candidate
idea_id: IDEA-190
source_type: experiment_result + code_analysis + first_principles + owner_hypothesis + nearest_work_boundary
method_name: Observable Role Entailment Field
method_acronym: OREF
current_run: none
problem: RCEG证明隐藏补全显著有害，而可见token与角色文本的判别交互仅作为普通控制产生正信号；需要检验动态角色命题能否通过显式支持/反证账本和非补偿式求解形成独立路径。
hypothesis: 候选相对当前最强竞争类的八个可见角色命题，若由当前图像patch witness形成有符号蕴含状态，并由反证优先规则求解，应在class-disjoint类别上超过name-only父基线、普通token scorer、MLP/CBM/ECOC控制及三项module-off。
core_change: 将类别判别从点相似度或opaque reranker改写为动态role-vs-rival claims、signed visible ledger和fixed falsification-first solver。
success_condition: 先通过本文100/50 proof Gate；后续正式Chen-style Full高于name-only计算父基线，且S/V/I同checkpoint module-off分别至少降低1.0 H。
failure_condition: Parent/Target-free/方向/module-off/普通scorer/CBM/random-code/content-control任一硬门失败即drop；禁止事后调整温度、反证系数、adapter rank、角色集、query或融合救活。
evidence_refs:
  - research/ideas/IDEA-104_rgwps.md
  - research/ideas/IDEA-105_crgwps.md
  - research/ideas/IDEA-110_pdrs.md
  - research/ideas/IDEA-158_gave.md
  - research/ideas/IDEA-159_rgt.md
  - research/ideas/IDEA-164_observable_signed_evidence.md
  - research/ideas/IDEA-189_role_contrast_evidence_gain.md
problem_category: visual_grounding
mechanism_tags: [dynamic_role_claim, visible_witness_field, signed_support_refutation, falsification_solver]
base_framework: FRAMEWORK-V5
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
gate_computational_parent: frozen OpenAI CLIP ViT-L/14@336 CLS × one canonical class-name prompt
formal_reference: FRAMEWORK-V5 H=81.06877662507551 is reported only; OREF prohibits PCLR online inference
owner_decision: 2026-09-01 owner明确接纳IDEA-190/OREF作为方案3 proof-of-path候选，并授权继续按尝试→失败→合规补救→经验总结闭环执行；该授权不改变预注册失败门，不允许核心反例成立后调参救活。
reuse_refs: [IDEA-104, IDEA-105, IDEA-110, IDEA-158, IDEA-159, IDEA-164, IDEA-189]

## Evidence status

RCEG Gate 0 is negative evidence, not OREF current advantage. On the exact 100/50 development split, Parent=`66.695482%`, RCEG Full=`56.093657%`, and RCEG Target-free control=`68.711805%`. This proves hidden completion was harmful and a supervised visible-token/text ranker can work on this Gate; it does not prove any new paradigm. OREF starts with `current_advantage:none` and must beat Parent, the frozen RCEG Target-free result, and same-feature ordinary scorers.

## Mandatory admission fields

old_solution_path: `name-only CLS↔class prototype similarity → calibrated all-class argmax`; RCEG adds candidate completion/gain; RCEG Target-free adds an opaque learned scalar ranker.

new_solution_path: `name-parent ambiguity → eight candidate-vs-rival visible claims → all 576 current-image patch witnesses → signed per-role support/refutation ledger → fixed falsification-first non-compensatory solver → all-class logits`.

principle_difference: A class is not represented as one point or fixed concept code. For every current image and candidate, OREF creates eight dynamic hypotheses relative to that image's strongest name-parent rival. Every hypothesis is tested against all visible patch witnesses. The persistent intermediate state is an explicit `[candidate,role]` signed ledger; final reasoning penalizes one confidently refuted role more than another role's large support can compensate.

old_signal_or_primitive: global similarity, token maximum similarity, fixed concept activation/code, PCLR edge score, reconstruction energy, or opaque output norm.

new_signal_or_primitive: dynamic role entailment state `e_c,k=o_c,k*tanh(m_c,k/0.2)`, where `m` is positive-vs-negative patch witness evidence and `o` is evidence concentration. The solver consumes eight signed states rather than a class embedding.

paradigm_shift: GZSL becomes falsification-first hypothesis testing: candidate roles are supported, refuted, or effectively unobserved by current-image witnesses; class selection operates on this evidence state instead of direct compatibility.

why_not_module: Patch/text adapters and logit fusion are implementation tools. OREF is invalid if the same ledger passed to a scalar MLP, FILIP-style max similarity, a learned CBM scorer, random semantic codes, or the frozen RCEG Target-free control matches it. Then the useful part is ordinary discriminative reranking, not falsification-first solving.

closest_paradigm_work:
- FILIP (ICLR 2022) already uses token-wise maximum image-text similarity for fine-grained late interaction: https://openreview.net/pdf?id=cpDhcsEDC2
- LaBo (CVPR 2023) already uses LLM-generated category sentences and CLIP as a language concept bottleneck: https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Language_in_a_Bottle_Language_Model_Guided_Concept_Bottlenecks_for_CVPR_2023_paper.html
- Label-free CBM (ICLR 2023) already builds CLIP concept bottlenecks without concept labels: https://openreview.net/pdf?id=FlCg47MNvBA
- Semantic Output Codes (NeurIPS 2009) and ZSECOC already formulate zero-shot recognition through semantic/error-correcting codes: https://papers.nips.cc/paper_files/paper/2009/hash/1543843a4723ed2ab08e18053ae6dc5b-Abstract.html

closest_work_conclusion: OREF cannot claim token alignment, automatic language concepts, semantic codes, or visual entailment itself. Its only possible narrow distinction is the combination of image-conditioned candidate-vs-rival role claims, an explicit signed support/refutation witness ledger, and a fixed non-compensatory falsification solver in class-disjoint GZSL.

## Exactly three deployment modules

### S — Dynamic Role Claim Composer

Inputs are explicit:
1. name anchor: `att_splits.mat/allclasses_names → clean_class_name → "a photo of a {class name}." → frozen CLIP [C,768]`;
2. existing text-v2 eight role sentences `[C,8,768]` in fixed order `[beak,head_features,body_plumage,wings,tail,legs,overall_appearance,unique_discriminative_features]`, SHA `bd935b8a4ed42d59c3a39c3f30bb99552c717ef18dadbf3349422b1cef728985`, with `llm_world_knowledge_used=true / expert_attributes_used=false`;
3. frozen name-parent logits for the image.

Stable-sort each active class axis by `(-parent_logit,global_class_id)`. For candidate `c`, rival `r(c)` is parent Top-2 if `c` is Top-1, otherwise Parent Top-1. Train/eval/formal axes are exactly 100/150/200; no Top-K pruning.

For role `k`, output `q_c,k=normalize(role_c,k-role_r(c),k)`. No per-class or learnable S parameter.

S-off: replace all eight `q_c,k` with `q_name,c=normalize(name_c-name_r(c))`, repeated eight times. Same checkpoint/axis/final interface.

### V — Visible Witness Field

Input: one unmasked frozen CLIP image forward. Frozen L2-normalized final projected patch tokens are `P_i∈R^768, i=1..576`; CLS is retained only for V-off. No mask, teacher, distillation, hidden target, crop, box, part label or PCLR asset.

Every condition that reads patch tokens first applies the same shared trainable residual:

`P'_i=normalize(P_i+W_o GELU(W_i P_i))`, with bias-free `W_i:768→64`, `W_o:64→768`, zero initialization of `W_o`.

Full, S-off, I-off, Ledger-MLP, FILIP, Signed-Ledger, Learned-CBM and Random-code all compute their visual statistics from `P'`. Text-only does not instantiate/read the adapter. V-off physically bypasses both the adapter and 576-token asset and reads only frozen CLS. Because `W_o=0` makes the identity initialization exact, the gradient receipt is two-step: first backward requires finite nonzero `W_o` gradient and permits exactly zero `W_i` gradient; after the first optimizer step, the second backward requires finite nonzero gradients for both `W_i/W_o`. Failure of either required gate aborts the condition.

For every `c,k`, let `s_i=P'_i·q_c,k`, fixed patch temperature `tau_p=0.07`, `N=576`:

`support=tau_p*logsumexp(s_i/tau_p)-tau_p*log(N)`

`refutation=tau_p*logsumexp(-s_i/tau_p)-tau_p*log(N)`

`m=support-refutation`

`p+=softmax(s/tau_p)`, `p-=softmax(-s/tau_p)`

`o=max(1-H(p+)/log(N), 1-H(p-)/log(N))`

`e=o*tanh(m/0.2)`

Output ledger per candidate/role: `{support,refutation,m,o,e,argmax_support_patch,argmax_refute_patch}`. Witness IDs are diagnostic only and never trainable.

V-off: replace the 576 patch witnesses by one normalized global witness `g=normalize(CLS)`. Define `support=g·q`, `refutation=-g·q`, `o=1`, `e=tanh((support-refutation)/0.2)`. This is explicitly local-witness-field off, not no-visual.

### I — Falsification-First Solver

Input: base logits and eight `e_c,k` values. I has no learned MLP, Gate or class-specific parameter.

`positive_c=mean_k relu(e_c,k)`

`negative_c=tau_r*logsumexp(relu(-e_c,k)/tau_r)-tau_r*log(8)`, fixed `tau_r=0.1`

`F_c=positive_c-2*negative_c`

`Z_F=(F-mean_axis(F))/sqrt(var_axis(F)+1e-6)`

`final_logits=base_logits+base_std*tanh(Z_F)`, `base_std=sqrt(var_axis(base_logits)+1e-6)`. No alpha or gamma grid.

Training updates only the V residual defined above; all class text, CLS and CLIP backbone are frozen. Loss on 100 dev-seen candidates is `CE(final,y)+softplus(0.1-(F_y-F_hardwrong))`; hard wrong is the highest frozen parent-logit wrong class. Fixed 1,000 updates, seed7, AdamW lr1e-3/weight_decay1e-4, batch8, last checkpoint only; 50 dev-unseen images/text are absent from training.

I-off: same S/V ledger and checkpoint, physically bypass the positive/negative falsification solver and set `F_off=mean_k(e_c,k)`; keep identical axis standardization, `base_std`, `tanh` and parent logits. Tests must monkeypatch the falsification solver to raise if I-off calls it.

Physical off contract:
- S-off loader does not open/read role embeddings and never constructs role queries; it repeats only the name query.
- V-off loader does not open/read the 576-token file and never instantiates/calls `W_i/W_o`; it uses CLS global witness only.
- I-off never calls the non-compensatory solver; it consumes the already built ledger through the compensatory mean.
- Each off condition writes opened asset keys and module call counts into its receipt; calculation-then-discard is a failure.

## Hard non-equivalence controls

All separately trained controls clone the same V-adapter initialization, batches, optimizer, budget and final base/std/tanh interface.

1. `Ledger-MLP`: flatten the same eight `{support,refutation,m,o,e}` entries to 40 values; a shared trainable `40→64→1` scorer is added on top of the same V adapter. This is deliberately a stronger-capacity反证, not a parameter-matched control; no dummy parameters are used. If it matches Full, fixed solver is unnecessary.
2. `FILIP`: same `P'` and dynamic role queries; candidate score is `mean_k max_i(P'_i·q_c,k)`, then identical standardization/fusion/loss.
3. `Signed-Ledger`: use the exact same `{support,refutation,m,o,e}` ledger but set score to the compensatory `mean_k(e_c,k)` and train its own V adapter from the shared initialization. This is the separately trained statistical-ledger/FILIP-strength control; if it matches Full, falsification solving is unnecessary.
4. `Learned-CBM`: use eight absolute candidate role queries `role_c,k` (not rival differences), activations `a_c,k=max_i(P'_i·role_c,k)`, and a shared trainable `8→64→1` scorer. It uses the same training budget/final interface and is a stronger CBM control than fixed mean activation.
5. `Random-code`: a CPU generator `manual_seed(7)` creates one global raw Gaussian tensor `[200,8,768]` exactly once in global class-ID order; every row is L2-normalized and the complete tensor SHA is frozen. Train/eval/formal use `index_select(global_class_ids)` from this same codebook and record selected-axis/output SHAs. For the same stable rival algorithm, replace `role_c,k/role_r,k` by `random_c,k/random_r,k` and define `q_random=normalize(random_c,k-random_r,k)`; V/I are otherwise identical. Generating separate 100/150/200 codebooks is forbidden.
6. `Frozen RCEG Target-free`: exact comparator is bound to asset bundle `98f06c47e3d9fda4f698aca5de5d4a33292e507de10e523a36303cea93beb54f`, 2,355 identical eval rows, the same 150-class global axis and macro Top-1 definition, result `68.71180534362793%`, and receipt `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/rceg/V5-TRY-003-GATE0/EVAL/failure.json@sha256:a7d96e3ffb729f0e6839727c25b1561928d0e74bd062ded869e5f82196f97c16`. If any row/axis identity differs, rerun Target-free on OREF rows instead of reusing this value.
7. `Text-only`: remove image tokens and score only candidate/rival role geometry with a shared `24→64→1` model over per-role cosine/distance/norm features. It does not instantiate the V adapter and reports `visual_adapter_gradient=not_applicable`.

Content controls: role-text block cycle, cross-image token cycle within each 100/50 block, and sign flip `q→-q`; each must destroy the positive gain. For every cycle, blocks are defined by the frozen train/eval global class IDs; within class, images are stably sorted by repository-relative path; a single block-local forward rotation maps row `j→(j+1) mod n`, requires `n≥2`, and has no fixed point. Role-text cycles similarly use global class-ID sorted rows within the 100/50 blocks. Mapping arrays and SHAs are frozen before training.

## Gate and statistics

Gate train uses CUB xlsa17 100 dev-seen classes/4,702 images. Frozen eval uses 50 dev-unseen classes/2,355 images under the 150 active class axis. Official test is absent; unseen images/text never enter gradient or checkpoint selection.

One seed7 Gate passes only if all hold:
- Full macro Top-1 ≥ Parent +1.0pp and ≥ frozen RCEG Target-free `68.711805%` +0.5pp;
- Full relative to S-off, V-off and I-off each ≥+1.0pp;
- Full relative to Ledger-MLP, FILIP, Signed-Ledger, Learned-CBM, Random-code and Text-only each ≥+0.5pp;
- every displayed difference has a paired 50-class bootstrap 95% lower bound >0 using one seed7 10,000×50 matrix;
- `F_true>F_hardwrong` macro ≥60% with 95% lower bound >50%; corrections-damages >0;
- role-text cycle, cross-image token cycle and sign-flip retain at most20% of Full's positive gain.

Gate passing is only proof-of-path. Formal success later requires Chen-style Full above the declared name-only computational parent and same-checkpoint `H_full-H_Soff/Voff/Ioff≥1.0pp` each. `H=80` is a target, not a pass line.

minimal_falsification: First execute only Parent, Full, three off paths, Ledger-MLP, FILIP and Signed-Ledger on the fixed Gate. If Full does not beat Parent/Target-free, direction fails, or any ordinary scorer matches, immediately reject OREF before the remaining controls. Do not tune temperatures, coefficient2, adapter rank, role set, query definition or fusion after results. The fixed `tau_p=0.07`, margin scale`0.2`, `tau_r=0.1` and refutation coefficient`2` are acknowledged untested inductive biases; failure may not be rescued by changing them.

current_advantage: none. RCEG Target-free is only a diagnostic comparator.
performance_status: proof_of_path_not_run.

failure_boundary: Patch tokens may encode texture but not the role claim; LLM role facts may be wrong; rival-dependent queries may simply reproduce pairwise reranking; the fixed worst-evidence penalty may over-refute occluded roles; ordinary FILIP/MLP/CBM may explain all gains. Any hard-control match immediately rejects OREF and forbids module-level rescue.

control_scope_disclosure: Learned-CBM covers the absolute language-concept bottleneck boundary only. Dynamic pairwise-query and same-ledger scorer alternatives are covered by Ledger-MLP and Signed-Ledger; no single control is claimed to cover every prior family.

implementation_cost_contract: train batch=8, frozen eval batch=4, candidate chunk=5, role chunk=8, token count=576; tokens are stored FP16 and cast FP32 per chunk. Full formal 200-class execution must report peak memory and may reduce batch only as an engineering rerun without changing candidate/role/token axes.

paper_level_claim: Only after Gate, formal H, three module-off gates and multi-seed evidence: “Dynamic candidate-vs-rival role claims are resolved through an explicit visible support/refutation ledger and non-compensatory falsification in GZSL.” No first-token-alignment, first-concept-bottleneck, first-entailment or first-code claim.

## 范式Idea双Agent对抗定稿记录

review_date: 2026-09-01
review_agents: [`/root/idea189_a`, `/root/idea189_b`]
review_subject_sha256: `7140685cf7a80f567225346b86eacd8672a260514c1de5d28c2ba9bea8c70f52`
review_status: passed_for_proof_of_path_idea_only

- 前置独立论证：Agent A从第一性原理提出OREF；Agent B否定“把RCEG Target-free +2.0163pp倒包装为范式成功”，双方直接交叉后共同把该结果降为负证据线索，并要求OREF独立胜过普通reranker。
- 第一准确草稿SHA256=`d86b8326776600a2674588926fce7827e84cd91aa3ea6cf5b89a53fd43b34c91`。双方独立审查和直接交叉共同判`REVISE`；集中问题为V adapter未明确进入ledger、FILIP/CBM/ECOC控制偏弱、三off缺少物理关闭。
- 第一集中修订SHA256=`ae2d177e26d7e010ebedb1bb61e25b375956e0938467a89d64b53f8c6855895a`。双方再次独立审查与交叉，关闭Signed-Ledger、MLP强反例、Learned-CBM、global random-code、off、Target-free rows/SHA和chunk合同；继续发现identity初始化两步梯度与global codebook轴合同。
- 最终准确审核对象为上述`review_subject_sha256`。两名Agent分别完整读取后均给出`P0=0 / P1=0 / P2=0 / PASS`，随后交换完整终审清单并直接回应；双方均确认无补充、无异议、无遗漏。
- 最终共同结论：**范式Idea双Agent对抗审核通过**。
- 该通过只证明OREF具备可证伪的proof-of-path方法资格，不代表Gate成立、Innovation晋级或论文claim成立。任何强控制追平、cycle保留收益或三模块贡献不足都必须按合同drop。
