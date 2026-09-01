# IDEA-205 draft v0 / Role-Grounded Relation Alignment (RGRA)

idea_id: IDEA-205
status: proposed_proof_of_path_not_innovation
source_type: owner_requirement + V5_result + CRC_failure_diagnosis + first_principles + nearest_work_boundary
problem_category: class_competition
mechanism_tags: [end_to_end_joint_training, role_patch_attention, relation_field, multiplicative_grounding, graph_free_inference]
base_framework: FRAMEWORK-V5
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
candidate_framework: FRAMEWORK-V6-DEVELOPMENT
method_name: Role-Grounded Relation Alignment
method_acronym: RGRA

problem: V5 reaches H=81.068777, but its TG/GTD, relation Reader, dynamic PCLR potential and R3/R4 role ensemble are not one deployed score graph trained by the final classification error. CRC tried to compile the relation graph after loading frozen V5 representations, so it was graph-free at inference but still only trained an external head. The owner requires the semantic, visual and interaction deployment modules to participate in the same forward, backward and optimizer step.

hypothesis: For fine-grained CUB competition, a relation direction should affect a class only when the corresponding role-level visual evidence is present. Jointly training role semantics, role-conditioned patch attention and a graph-integrated relation field through the final seen-class CE will preserve V5 transfer while turning local evidence and pairwise relations into a non-additive, graph-free classifier. If the hypothesis is correct, the same selected checkpoint will beat the accepted V5 H=81.068777 and each S/V/I module-off will reduce H by at least1.0 point.

old_solution_path: `8 role texts -> TG/GTD prototypes`; independently `image CLS -> PCLR Reader -> Top-K edge mask -> per-image Laplacian solve -> potential`; after training, R3 tunes graph inference and R4 adds role0/role6 logits and calibration. Deployment is a post-training composition and PCLR performs graph logic online.

new_solution_path: `8 role texts -> shared semantic composer -> class prototypes/role queries`; `same-asset CLS+36 coarse patches -> shared visual adapter and role-conditioned soft attention -> class-local support`; `438 directional relation texts -> one fixed regularized incidence integration -> class relation field`; one non-additive matcher multiplies relation compatibility by visual role support; one final logit tensor is optimized by one total loss and exported without an online graph.

principle_difference: The old path independently estimates absolute class similarity and a dynamic graph correction. The new learning object is a jointly optimized evidence tuple `(semantic prototype, visible role support, class relation direction)` and the relation term is conditionally grounded by observed patch support. No per-image Top-K, mask, Laplacian solve, post-hoc role fusion or post-training compiler remains.

old_signal_or_primitive: Absolute text prototype similarity plus separately trained pairwise edge scores and post-hoc graph potentials.

new_signal_or_primitive: A class relation field `G[c]` deterministically integrated from legal directional relation texts, coupled multiplicatively to learned role-specific patch support in the final classification graph.

paradigm_shift: The classifier changes from separately scored absolute prototypes plus online graph correction to a single conditional relation matcher whose basic unit is “this visible role evidence supports this class-relative direction.”

why_not_module: The semantic composer and patch attention have close prior art and are not individually claimed as novel. The candidate novelty is the graph-free class relation field conditioned by learned role visibility and trained jointly with the final GZSL classifier. If the multiplicative term is no better than an additive relation head or if fixed `G` is only a reparameterized class prototype, the paradigm claim fails and the result can only be retained as an engineering framework.

## Exact inputs and three deployment modules

Frozen backbone boundary: OpenAI CLIP ViT-L/14@336 remains fixed to preserve the exact V5 visual setting. “End-to-end” here means end-to-end from the frozen CLIP feature tensors through all three trainable deployment modules to final logits; it does not claim raw-pixel CLIP fine-tuning.

Text input: `role_sentence_embeds[200,8,768]` from the V5 asset. The eight sentences are exactly six visual roles (`beak`, `head_features`, `body_plumage`, `wings`, `tail`, `legs`) plus `overall_appearance` and `unique_discriminative_features`. This is not three generic class-name prompts and does not use expert attributes.

Visual input: `CLS[B,768]` plus `coarse_patches[B,36,768]`, loaded from the exact accepted V5 manifest `/data/lby/projects/cv_project/GZSL_Warehouse/assets/v3/CUB_openai_vitl14_336_dynamic_v3_v1/asset_manifest.json@sha256:3a6b261a63e2aa241d7a9cd2b3c9b0051a0ba01133ef61dc35e0d043fc119fa6`. The bound files are `train_coarse_patch_features.npy@ee5e937a4b78cb6d8f8babaa22cfed43ab2807080fda0d1dd4bba4206485c1f0`, `test_seen_coarse_patch_features.npy@ce7b902e43bd0cb89c4244b9d13e19d3af0ef7f91997d718e3e4a5afa389d405`, and `test_unseen_coarse_patch_features.npy@ac63d8badf1d1daf9c6328c57016964a24511f815eabd502ca9d58bba17ff78b`. Shapes are `[7057,36,768]`, `[1764,36,768]`, `[2967,36,768]`, dtype float16. The manifest binds OpenAI ViT-L/14@336 checkpoint SHA `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`, preprocessing, raw-image order, parent manifest, and `24x24 -> non-overlapping 4x4 mean -> 6x6` formula. Historical generator is `tools/prepare_v3_dynamic_assets.py` at commit `63c4887d5457571ab31237f5b01f28174e1e4969` (blob `65eb15f5d68029a074d70a9ab0967346b813e19a`), but the accepted manifest does not record generator commit/file SHA, so `feature_provenance_complete=false` remains an inherited paper-claim limitation. No new patch asset is generated for RGRA.

Relation input: the fixed 438 undirected class pairs and two normalized directional relation-text embeddings per pair from `CUB_pclr_relations_453d684b5080f477`. These texts contain class-relative appearance differences and no unseen images or expert attributes.

S / Role Semantic Composer (RSC): from fixed role tensor `T[200,8,768]`, define exact raw group queries `q_raw=normalize(stack(mean(T[:,0:6],dim=1),T[:,6],T[:,7]))[200,3,768]` and `p_mean8=normalize(mean(T,dim=1))[200,768]`. A shared bottleneck residual `A_s` and learned `softmax(w_s)[3]` produce `q=normalize(q_raw+A_s(q_raw))[200,3,768]`. The accepted V5-R2 prototype table `p_v5[200,768]` is loaded only from its bound checkpoint and is a non-trainable initialization anchor; `p=normalize((1-rho_s)*p_v5+rho_s*sum_g softmax(w_s)[g]q[:,g])`, with bounded trainable `rho_s` initialized0.10. Output is `l_s=scale*z*p^T [B,200]`. S-off uses exactly `p=p_mean8` and `q=q_raw`; it disables `A_s,w_s,rho_s,p_v5` while V and I continue with the raw queries and unchanged relation field. Thus S-off changes only semantic composition, though its downstream consumers legitimately receive the bypass queries.

V / Role Visual Aligner (RVA): a residual adapter maps CLS to `z=normalize(x+A_v(x))[B,768]`; shared query/key projections compute `a[b,c,g,n]=softmax_n(W_q q[c,g] dot W_k patch[b,n] / tau_v)[B,200,3,36]`. Weighted cosine role evidence and the same three group weights give `l_v[B,200]`. Output is `(z,l_v,a)`. V-off uses exactly `z=normalize(x)`, `l_v=zeros[B,200]`, and support gate `g_v=0.5*ones[B,200]`; RFM still computes its own reader and relation score from raw `z`, so V-off is not a hidden V+I combined removal. If this honest V-off gap is below+1.0 H, V fails the module contract.

I / Relation Field Matcher (RFM): build incidence `B[438,200]`, directional difference `D[438,768]`, `M=(B^T B+0.3I)^-1 B^T`, and fixed `G=normalize(MD)[200,768]` once at model construction. A shared residual relation reader gives `z_r=normalize(z+A_r(z))[B,768]` and `r=z_r G^T/tau_r [B,200]`. For Full, define `standardize(l_v)=clamp((l_v-mean_c(l_v))/(std_c(l_v,unbiased=False)+1e-6),-5,5)` and `g_v=sigmoid(standardize(l_v))[B,200]`; the exact deployed term is `l_i=alpha*g_v*r [B,200]`, with bounded `alpha`. I-off sets only `l_i=zeros[B,200]`; S and V remain unchanged. For V-off, `g_v` is the fixed sigmoid-neutral0.5 tensor while RFM remains active on raw CLS. Deployment stores `G` and learned weights only and never opens edges, relation text, incidence, Top-K or Laplacian assets.

Final logits: `l_full = l_s + beta_v*l_v + l_i - gamma*1_seen`, with inherited calibration `gamma` fixed and present in Full and every S/V/I-off. `beta_v` and `alpha` are bounded trainable scalars. Training CE uses the150 seen-class slice so true-unseen images never enter gradients; evaluation uses200-class GZSL and50-class ZS competition.

## End-to-end optimization contract

Initialization binds only the accepted V5 R2 checkpoint `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/pclr/V4-TRY-023-R2/model_best.pth@sha256:16b5071f21a3217e58a72315029c28b8cfd97b68f812641bd0145d3f5e0702ab`, source config SHA `0861877ae3e4725e29aff547d45e0b6d56a186179309acb5493c5906b803fd49`, and source code commit `b0a756dd624e883eb50d19a2455ba06bdc73f118`. It is a warm-start initialization, not a teacher, and no sequential stage or freezing follows. At update1, trainable shared RSC composer/weights/scalar, RVA adapter/query/key attention parameters, and RFM reader/scalar are all in one AdamW optimizer and one `zero_grad -> forward -> total.backward -> step`. The CUDA micro-batch separately backpropagates `L_cls` alone and requires finite nonzero group gradient norm for each of those three deployed module groups; it then records total-loss gradient norms separately. True-unseen class rows receive no supervised label gradient and must generalize only through the shared text composer, shared visual aligner and fixed relation field; no claim of per-unseen-row CE gradient is made.

`L_total = L_cls(full_seen_logits,y) + 0.3*L_topology + 0.1*L_direction`. Attention receives its task signal directly from `L_cls`; no attention-balance loss is added in the first implementation. Auxiliary terms may regularize but may not replace the direct `L_cls` gradient to any deployed module group. There is no teacher, distillation, expert attribute, unseen-image gradient, PCLR online inference, sequential freezing, per-stage checkpoint selection or post-hoc compiler.

Checkpoint selection: one global best checkpoint selected only by Full H under the disclosed Chen-style official-test-selected protocol; no separate S/V/I checkpoint or stage selection. Module contributions are measured by same-checkpoint S/V/I-off with all other conditions fixed.

## Falsification and performance contract

minimal_viability: Before any official evaluation, one real CUDA batch50 must prove all bound asset shapes/SHAs, finite Full logits/loss, nonuniform finite patch attention, executable S/V/I-off paths, exact `alpha=0 == I-off`, and finite nonzero `L_cls`-only gradient norms into the shared trainable RSC/RVA/RFM groups. Export parity is required only after a checkpoint exists. A short fixed-trace screen may establish numerical viability, but it cannot satisfy the performance contract.

minimal_falsification: The formal 28,228-update, batch50, seed7 Chen-style run is rejected as a successful RGRA framework if Full does not exceed accepted V5 `H=81.068777` or any same-checkpoint gap `H_full-H_Soff`, `H_full-H_Voff`, `H_full-H_Ioff` is below+1.0. Because the parent is accepted V5, there is no lower parent-performance gate: `H=80` is a target only and does not pass this candidate. Report update0, complete official history, best Full U/S/H/ZS, independent best-ZS observation, all off/control metrics, net corrections and every identity SHA. A short screen may only stop a catastrophic/numerically invalid implementation, not reject a stable learning hypothesis for being undertrained.

non_equivalence_test: (1) deployment audit confirms no edge/relation/incidence/graph/Top-K asset access; (2) `alpha=0` returns the I-off S+V logits exactly, bitwise on CPU and within1e-6 on CUDA; (3) additive control uses the same checkpoint and `l_i_add=alpha*0.5*r`, preserving neutral expected scale; Full conditional must exceed additive by at least+0.5 H and deliver at least20 more corrections than damages relative to additive; (4) shuffled-support control applies a fixed seed7 per-image class permutation to `g_v` while preserving its values and `r`; Full conditional must exceed shuffled by at least+0.5 H and deliver at least20 more corrections than damages relative to shuffled; (5) export logits must match model logits within1e-5. Items3-4 are hard gates for the conditional relation-field innovation claim. If framework performance and S/V/I contracts pass but either non-equivalence gate fails, RGRA may be retained only as an engineering framework and cannot be registered as innovation/paper_core_innovation.

current_advantage: none for RGRA. Accepted V5 remains H=81.068777. Prior CRC compilation proved algebraic/export feasibility but not end-to-end performance; prior V2 RGVE showed patch attention is feasible but only a weak gain. Therefore this card remains a proposed `proof_of_path` and must not enter the active Innovation/paper-core list until a real accuracy or measured speed/cost/generality advantage exists.

performance_status: proof_of_path

failure_boundary: Exact V5 coarse patches contain only36 pooled regions and may erase small bird parts; their accepted manifest lacks generator commit/file SHA, so patch provenance remains incomplete for a final paper claim even though run identity is content-bound. Seen-only CE can overfit RVA to seen appearance; a fixed relation field may collapse into another class prototype; multiplicative grounding may suppress useful negative relation evidence; V5 warm start can be damaged by joint fine-tuning; official test selection is non-blind; initial CUB evidence does not establish cross-dataset generality. Attention maps are diagnostics, not part localization or causal evidence. If conditional interaction fails additive/shuffle gates, it cannot be claimed as the innovative point even if Full H rises.

problem_family: Fine-grained GZSL where classes share global appearance and differ in local role evidence and relative descriptions.

shared_bottleneck: Absolute similarity does not express which visible local cue supports one member of a confusing class pair, while ungrounded relation scores can react to irrelevant evidence.

reusable_capability: If supported, a graph-free relation field can transfer pairwise text differences into a deployable classifier and role support can condition when those differences matter.

coverage_and_transfer: Initially CUB/seed7 only. AWA2/SUN are not claimed and require their own legal role/relation assets and same-protocol confirmation.

frontier_shift: Potentially replaces per-image graph solving and post-hoc tuning with one exportable joint classifier; speed/cost advantage must be measured, not assumed.

downstream_effects: Expected but unproven outputs are role attention maps, relation support traces and error attribution for S/V/I. They are diagnostics, not localization or causal explanations.

paper_level_claim: Only if performance, non-equivalence and closest-work controls all pass: “a graph-free class-relation field, conditionally grounded by role-level visual support and optimized jointly with role semantics, improves fine-grained GZSL under a frozen CLIP backbone.” No “first” claim.

## Closest paradigm work checked from primary pages

- Huynh and Elhamifar, CVPR2020, *Fine-Grained Generalized Zero-Shot Learning via Dense Attribute-Based Attention*: semantic-conditioned region attention and end-to-end GZSL already exist, but use expert attribute vectors/attribute semantics rather than 6+1+1 free-text roles and a class relation field. https://openaccess.thecvf.com/content_CVPR_2020/html/Huynh_Fine-Grained_Generalized_Zero-Shot_Learning_via_Dense_Attribute-Based_Attention_CVPR_2020_paper.html
- Liu et al., CVPR2023, *Progressive Semantic-Visual Mutual Adaption for Generalized Zero-Shot Learning*: dual semantic-visual transformers and instance-centric attribute prototypes already model nontrivial visual-semantic interaction. RGRA cannot claim first cross-modal attention; its narrower candidate point is conditional pairwise relation-field grounding without attributes. https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Progressive_Semantic-Visual_Mutual_Adaption_for_Generalized_Zero-Shot_Learning_CVPR_2023_paper.html
- Saha et al., CVPR2024, *Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions*: structured descriptions for VLM adaptation already exist, so 6+1+1 descriptions alone are not novel. https://openaccess.thecvf.com/content/CVPR2024/html/Saha_Improved_Zero-Shot_Classification_by_Adapting_VLMs_with_Text_Descriptions_CVPR_2024_paper.html
- Jiang et al., CVPR2025, *Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning*: visual/semantic collaboration and prompt tuning already exist. RGRA freezes CLIP and works on exact cached CLS/patch features; this is a protocol difference, not by itself novelty. https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_Visual_and_Semantic_Prompt_Collaboration_for_Generalized_Zero-Shot_Learning_CVPR_2025_paper.html
- Kampffmeyer et al., CVPR2019, *Rethinking Knowledge Graph Propagation for Zero-Shot Learning*: graph propagation from semantic class nodes to classifiers already exists. RGRA must show its image-conditioned role-support multiplication is not just a DGP/GCN-like class prototype or additive graph head. https://openaccess.thecvf.com/content_CVPR_2019/html/Kampffmeyer_Rethinking_Knowledge_Graph_Propagation_for_Zero-Shot_Learning_CVPR_2019_paper.html

owner_requirement: 2026-09-02 owner explicitly rejected the CRC two-stage/frozen-parent interpretation, required an end-to-end method, and had already authorized continued in-scope attempts without repeated approval. After the required dual-Agent adversarial finalization below, this authorizes proof-of-path implementation only; it does not authorize promotion as Innovation.

## 2026-09-02 范式 Idea 双 Agent 对抗定稿

- final_draft_sha256: `1dd8b7c82aac240f41eaf2f755c558e36603238bbfe6cf35661a3d6997f5d5a3`
- Agent A 独立初审: `9abe556099ddc298749ea1a72cffe7d961f75e360ada5f51d2e8aa0e3b12917a`，`P0=0/P1=6/P2=4`，`revise`。
- Agent B 独立初审: `c6e0e6dd0a1b267054b4eb707c993ac46d7684bef247ed3d0b4b2d908c103986`，`P0=0/P1=5/P2=5`，`revise`。
- Round1 直接交叉: A=`2687d3e42b8344c4a4f7fe4a28cfd1053fb882fb839f121445d394c8575ca556`；B=`0ceed6ac538b39640a26a92ef7ae689251d3c60bede344a57e5bcec99eb51d00`。集中关闭 patch 身份、`L_cls` 单独梯度、S/V/I-off、非等价控制和 V5 成绩门。
- Agent A Round2 独立复核: `bce6f172a22abe8204377e30be5ccadf54910c3f68d07283733a8176d83f0168`，`P0=0/P1=0/P2=0/pass_for_proof_of_path`。
- Agent B Round2 独立复核: `be0bba865265021dbb6158295a9573fe5ea69b983133c870cdc508e72e710c74`，`P0=0/P1=0/P2=0/pass`。
- Round2 最终直接交叉: A=`ee3532d22d632a66d50693e32d726b8f0766bfc3b306a128385a41e4e91b12c4`；B=`37843631f976840cee8deac3feb34207def1ca9b39ef14fd91828ad333f14bee`。
- 共同结论: `P0=0/P1=0/P2=0`，**范式Idea双Agent对抗审核通过**。此结论只允许把 RGRA 作为 `proof_of_path` 候选实现；真实优势和 hard non-equivalence gates 成立前不得称为 accepted Innovation 或 paper core。
- owner_confirmation_basis: owner 已明确要求“我要端到端的”，并已给出本任务范围内“以后不用再问，失败继续补救”的持续执行授权；据此批准从准确父提交 `52b511d77b4ad048f35b40dc3cbd9afd092167e9` 独立实现 proof-of-path，不批准提前晋级。

## 2026-09-02 实现代码双Agent交叉审查

- reviewed_commit: `b69c4f3548c9188849f33dc58f85f01c4f5ef291`
- reviewed_tree: `f8cf8899b1956ab866bae6a92090b40d8f61b0aa`
- config_sha256: `95b0bc5791e0e7ccabbf1014f09d086cf1a81260e24c5291b0f3def98ff15c6f`
- review_receipt: `experiments/v6/innovation/V6-TRY-008_RGRA_REVIEW.md`
- final_A: `fa0aea250db6e2e89f2be115131924ac8054ea8709ea508451f99ee84d72feef`
- final_B: `6c054c4d275376db6f3893cd8d0f28cca1d7d1c2de8bbdb28ed9ecff1b642a36`
- result: `P0=0/P1=0`，**双Agent交叉审查通过**；只授权固定配置RUN。

## 2026-09-02 CUDA micro与正式RUN失败结果

- execution_commit: `c830096b50e9f4721b72478f687b412b308bb832`
- config_sha256: `95b0bc5791e0e7ccabbf1014f09d086cf1a81260e24c5291b0f3def98ff15c6f`
- CUDA micro通过：batch50；`L_cls`到RSC/RVA/RFM梯度范数=`0.463408/2.033284/0.058595`；attention std=`0.032037`；`alpha0==I-off`误差0；未加载official test。
- 正式RUN在update22,983 checkpoint后被主动停止。直接原因是R2源模型在提取`P_v5`时未调用`eval()`，TG dropout使两次锚点重放max-abs=`0.145699`、SHA不同，违反固定V5 anchor合同。
- 停止前全局best仍为update423：`U/S/H/ZS=75.371444/80.628538/77.911411/84.670228`，低于V5 `H=81.068777`。
- 同checkpoint gaps：S=`+9.548995`、V=`+2.250195`、I=`+0.064736`；Full-additive=`-0.021937`、Full-shuffled=`+0.035142`。即使忽略工程身份错误，I模块与conditional non-equivalence也失败。
- failure_receipt: `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/rgra/V6-TRY-008/failure.json@sha256:7637294196a5fe00da91ff2550a2bd11e8ed9068f7c2541d319ebe5e09e9d5d7`
- decision: `drop_IDEA205_engineering_invalid_and_empirically_below_parent`。不在失败代码上继续堆；新的锚点保持/关系尺度公式另建IDEA-206并从正式父commit独立分叉。
