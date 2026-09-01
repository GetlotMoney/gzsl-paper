# IDEA-198：Safe Explicit Action Verification（SEAV）

idea_id: IDEA-198
status: rejected_pre_implementation
base_framework: FRAMEWORK-V6-DEVELOPMENT
source_code_parent: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
predecessor_evidence: IDEA-196 EAAC and IDEA-197 RoleTriPool, both rejected and not code parents
problem_category: reliability_robustness
mechanism_tags: [explicit_abstention_action_policy, learned_crop_safety_verifier, zero_semantic_off, sequential_seen_training, one_shot_verification]
implementation_branch: exp/v6/innovation/v6-try-002-seav
current_run: V6-TRY-002 / pre-implementation minimal falsification

problem: EAAC learned a conservative action policy (11.89% trigger) but its fixed rule swapped whenever selected-crop class-name margin was negative, producing53 corrections and59 leader damages. RoleTriPool and direct low-resolution pair classification failed, so the remaining verified bottleneck is whether the acquired crop should be trusted, not another action representation.

hypothesis: A seen-trained safety verifier conditioned on Parent state, chosen action, action confidence and selected-crop margin can filter harmful swaps while preserving class-disjoint corrections, making the same explicit-abstention action policy useful under B<=1.

exact_three_modules:

1. S Eight-Role Pair Questions: identical 6-part+overall+distinctive pair questions used by EAAC. S-off sets the entire learned question tensor to zero while retaining the frozen name-only Parent and name embeddings required by the final pair interface. This tests the complete semantic-question module, not rich text versus name-only prompting.
2. V Explicit-Abstention Action Policy: identical reviewed EAAC policy `[fixed zero abstain,25 action logits]`, 26-way CE target, 4:4 sampler, fixed1000 updates. V-off uses global CLS broadcast instead of 576 local patches; S remains Full.
3. I Crop Safety Verifier: replaces only the fixed negative-margin swap. For a frozen selected action/crop, input is `[Parent leader margin, Parent entropy, Parent logit mean, Parent logit std, crop leader-minus-challenger margin, abs(crop margin), selected policy confidence, normalized action box8]` =15 values. Shared `Linear(15,32)->GELU->Linear(32,1)`, zero output head, sigmoid>0.5 means swap; otherwise keep. I-off preserves Full action/trigger/crop/logical cost and always returns Parent.

training_contract:

- Stage1 independently reproduces EAAC from the formal source identity: seen-only target generation, exact natural target histogram, fixed4:4 sampler, seed7/batch8/1000 updates/AdamW1e-3 wd1e-4, final checkpoint; no eval data.
- Freeze all S/V parameters and decisions.
- Stage2 on dev_train only: compute the frozen policy's selected action for every row; only triggered rows enter verifier training. Read that one selected feature from the train all25 table. Target swap=1 iff truth is Parent challenger; target0 for leader/outside. Pre-registered current identity has574 triggered train rows,300 positive and274 negative. Train1000 updates with fixed16 positive+16 negative, seed7, BCE, no class weight/threshold search.
- Save one combined checkpoint containing frozen policy and verifier, both config/asset/oracle identities and both gradient receipts.

deployment_contract: Freeze action/trigger from only336 CLS+patches; if abstain keep Parent. If triggered, open and encode exactly one raw crop, build15-D verifier feature, swap only if verifier probability>0.5. No eval labels/all25, no B>1, no crop fusion.

Gate0: fixed50 dev-unseen class vector/150-axis macro Top1. Owner's current hard contract is observed Full-Parent>=1.0pp and same-checkpoint Full-Soff/Voff/Ioff each>=1.0pp; paired bootstrap CIs are always reported beside every gap but are not preliminary hard gates for Parent/module points, and any CI crossing zero is explicitly marked as formal/multi-seed risk. S-off, V-off and I-off all reuse the single Full-trained verifier checkpoint without retraining. Full must beat three same-action/trigger/crop verifier controls by>=0.5pp with paired CI lower>0: (1) Fixed-I; (2) No-crop verifier, trained/evaluated with the two crop-margin inputs fixed to zero and every other 15D input/capacity/batch identical; (3) Margin-only verifier, trained/evaluated with only crop-margin/abs active and the other13 inputs zero. Full must also achieve positive net, corrections>damages, challenger trigger>leader, both trigger/abstain, action occupancy and B1 gates. Center/Static/Random/TextHeatmap are reported with the Full verifier and trigger but are diagnostic, not additional hard gates for this verifier-focused Gate0.

Gate1_after_Gate0:

- Name-only semantic off is reported separately from zero S-off to quantify the contribution of 6+1+1 beyond class names; zero S-off remains the deployment-module contract.
- Natural-sampler action policy is attribution only; matching removes 4:4 from final method.

direct_diagnostic_evidence: Frozen EAAC actions plus the exact verifier above, trained only on574 seen triggered rows, gave class-disjoint eval Parent66.692923, Full68.014519 (+1.321596), corrections60, damages29, net+31. Same verifier with zero-S actions gave66.657313 (Full gap+1.357206), V-off66.730476 (gap+1.284043), I-off=Parent (gap+1.321596). Fixed-I Full was66.400172, so learned-I gain+1.614348 with paired CI[+0.721217,+2.749943]. Full/offs Parent differences had CI lower slightly below zero and are disclosed; the preliminary owner contract is point gain, with formal/multi-seed work required later.

non_equivalence_test: Fixed-I must lose to learned-I under identical actions/crops; no-crop and margin-only controls must not match Full. Otherwise the safety verifier is either unnecessary or only a no-glimpse group classifier.

minimal_viability: exact stage1/2 counts, all gradient gates, nonconstant verifier scores, both keep/swap among triggers, selected-crop feature identity and B1 physical counts.
minimal_falsification: Reproduce exactly seven Gate0 conditions: Full, S-off, V-off, I-off, Fixed-I, No-crop verifier and Margin-only verifier. Any observed1pp Parent/module gate, net, corrections-versus-damages, trigger/action-occupancy or B1 failure drops. Fixed-I, No-crop or Margin-only failing to lose to Full by>=0.5pp with paired CI lower>0 also drops the verifier mechanism; it may be retained only as accurately labelled engineering evidence and cannot support a crop-safety claim. No threshold, verifier width, feature set, action policy, prompt, geometry or B2 rescue is allowed in this TRY.

novelty_boundary: learned verification, selective prediction and crop routing are established; no standalone novelty claim. Under owner policy this is a performance-oriented V6 framework component. Closest boundaries include GapSight crop re-reading and selective/reject-option classifiers.
current_advantage: accuracy diagnostic +1.321596pp over Parent and all three observed module gaps>1pp; not yet a frozen Gate result.
performance_status: proof_of_path_diagnostic_above_parent
failure_boundary: diagnostic selection used the disclosed dev protocol; verifier may learn Parent-group priors rather than crop evidence; no-crop/margin controls may match; point gains may not be statistically stable; S-zero off is a broader semantic ablation than name-only off.
paper_level_claim: none until frozen Gate0, Gate1, formal and multi-seed evidence.

identity_and_receipts: Freeze and record the Stage1 checkpoint SHA, policy code commit, train-config SHA, natural-target-histogram SHA, sampled-batch-trace SHA, train/eval selected-action SHA and train/eval trigger SHA. Stage2 Full/no-crop/margin verifiers share the same frozen Stage1 decisions, target rows, initialization seed,1000 updates and balanced batch trace. S/V/I-off all use the single Full-trained verifier checkpoint without retraining. Report train/eval verifier probability histograms, positive/negative score separation and per-group keep/swap rates; these are diagnostics, not tuning inputs, and cannot be used to tune the fixed0.5 threshold. The stated68.014519 result remains a disclosed single dev-protocol diagnostic, not a frozen Gate result or generalization claim.

## 2026-09-02 范式 Idea 双 Agent 对抗定稿

- 审核草稿 SHA256：`d28dd37bb868e3d658e841c16cb4229218cb47df894a57034be459b4ef1e3237`
- Agent A 最终清单 SHA256：`2c99407b1ce061af34a0ce6c662a23525cb763ce1195aa2de5f85cadc688e30e`
- Agent B 最终清单 SHA256：`159eb425d2d5855369a8694b088bd23dbf24a10e5e1e6e6af390a15574268dd2`
- A/B 分别独立审查后直接交换完整清单并逐项回应；Gate0 七条件、同 checkpoint off、Stage1 身份、固定阈值和诊断边界均已关闭。
- 双方最终结论均为 `P0=0 / P1=0 / P2=0 / pass`。
- 共同结论：`范式Idea双Agent对抗审核通过`。这只允许建立可证伪候选，不代表指标、统计稳定性或论文新颖性已经成立。

## 2026-09-02 最小证伪结果

failure_receipt: `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/seav/V6-TRY-002-PRECHECK/failure.json@sha256:3d10db639cb2a55e4e676c53d63c722302913ae43eff978a019e071105d3090c`

- 在正式实现前，用冻结的 EAAC 动作和同一份已披露 dev 诊断协议训练三种 verifier：Full、No-crop、Margin-only；三者使用相同 15→32→1 容量、初始化、574 个 triggered seen rows 和同一 1000-step batch trace。
- Parent=`66.692923`，Full=`68.014519`，No-crop=`68.548231`，Margin-only=`66.161810`，Fixed-I=`66.400172`。
- `Full-No-crop=-0.533712pp`，paired CI95=`[-1.462184,+0.291184]`。No-crop 不仅匹配，而且点估计高于 Full，违反预注册的 `Full-No-crop>=0.5pp 且 CI lower>0` 硬门。
- Full 发生 60 次纠正、29 次破坏；No-crop 发生 84 次纠正、42 次破坏。增益主要来自 Parent 状态、动作置信度和位置先验，不需要 selected-crop margin；加入 crop margin 反而降低结果。

root_cause: The verifier learned a transferable pre-crop group/action prior rather than a crop-evidence safety decision. Therefore the proposed interaction module is not causally necessary.

decision: Drop IDEA-198 before formal code freeze, code review or training. Interrupted partial unreviewed implementation was deleted and is not a code baseline. The next candidate must not relabel a no-crop prior classifier as crop verification.
