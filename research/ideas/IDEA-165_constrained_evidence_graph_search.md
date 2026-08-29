# IDEA-165：Constrained Evidence Graph Search

idea_id: IDEA-165
source_type: experiment_result + first_principles + owner_hypothesis + nearest_work_boundary
status: rejected
problem_category: visual_grounding
mechanism_tags: [evidence_graph, exact_assignment, bitmask_dp, patch_capacity, algorithmic_inference]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs:
  - IDEA-162
  - IDEA-163
  - IDEA-164
problem: IDEA-162证明27个跨类别共享概念可以从冻结CLIP patch中学习读取，但现有Attention/最大池化让每个角色独立选择证据，同一背景或身体patch可同时证明多个角色。IDEA-163/164进一步证明在边权尚不可靠时强行定义三态或可观察性会失败；尚未检验“固定已成立边权，只改变全局证据组合求解”是否能够减少重复证据并改善class-disjoint分类。
hypothesis: 把图像patch与文本概念构造成二分证据图，并对每个类别的最多六个角色节点执行带Unknown选项和patch容量约束的精确最大权匹配，应能惩罚依赖同一patch重复证明多个角色的错误类别，同时保留真正来自不同区域的组合证据。若同一Reader边权下，容量约束求解相对独立最大池化产生class-disjoint净纠正和因果删除优势，则GZSL可被进一步重写为约束证据搜索而非独立相似度聚合。
core_change: 复现并冻结IDEA-162的共享概念Reader，只使用其固定27个可迁移概念，不训练新的视觉网络。每张图一次性生成`concept×576 patch`边权；候选类别通过文本簇成员关系得到最多六个角色概念节点。独立基线让每个角色各取最高正边；新求解器允许角色Unknown=0并要求patch容量。角色数最多6，每个角色只保留Top-6 patch仍不损失最优解。生产求解使用C实现的Hungarian精确可选匹配以避免Python DP约1000倍开销；状态压缩DP只作为32例全576逐值oracle，求解器名称不构成claim。
old_signal_or_primitive: 角色证据独立最大池化或attention加权，patch可被无限重复使用，分类仍是若干互不约束局部分数的加和。
new_signal_or_primitive: 图像表示为patch—概念加权二分图，类别表示为带Unknown的角色节点集合，分类原语变为满足容量约束的全局证据匹配解及其显式assignment。
paradigm_shift: GZSL从“网络直接输出类别分数”改写为“学习共享证据边，再在证据图上精确求解类别解释”；学习与求解解耦，匹配assignment本身就是解释。
why_not_module: DP、Top-6定理和bitmask只是求解工具；claim必须来自证据图原语、容量约束和可证伪的全局一致解释。若DP只降低重复率而不改善class-disjoint分类、因果删除或通用性，则只能记为proof_of_path，不能作为Innovation。
closest_paradigm_work:
  - Zero-Shot Recognition via Optimal Transport（WACV 2021，https://openaccess.thecvf.com/content/WACV2021/html/Wang_Zero-Shot_Recognition_via_Optimal_Transport_WACV_2021_paper.html）已把GZSL与OT结合，但依赖辅助attributes并进行分布输运，不是文本唯一角色—patch容量匹配。
  - PatchCT（ICCV 2023，https://openaccess.thecvf.com/content/ICCV2023/papers/Li_PatchCT_Aligning_Patch_Set_and_Label_Set_with_Conditional_Transport_ICCV_2023_paper.pdf）把多标签分类重写为patch集合与label集合的条件输运；IDEA-165面向GZSL class-disjoint语义迁移，使用每类别角色子图、可跳过Unknown和精确离散容量约束。
  - FedOTP（CVPR 2024，https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Global_and_Local_Prompts_Cooperation_via_Optimal_Transport_for_Federated_CVPR_2024_paper.pdf）用OT对齐局部视觉与prompt以解决联邦异质性；不是零样本类别谓词的解释搜索。
  - ProtoProp（NeurIPS 2021，https://proceedings.neurips.cc/paper/2021/hash/584b98aac2dddf59ee2cf19ca4ccb75e-Abstract.html）在组合图中传播独立attribute/object prototypes；不在单图patch—谓词图上做容量约束assignment。
  - IDEA-163/164已经否定类特定离散三态与候选无关可观察性公式，IDEA-165不得复用其Reader、阈值、损失或把失败热图改名。
optimization_round_1: 删除200类逐patch重复Reader计算、类别Top-K、A*、倒排索引、反义词图和新训练loss；边权只计算一次，求解只比较independent与capacity-1精确assignment。
optimization_round_2: 用Top-R精确保留定理把576 patch缩为每角色Top-6，并将概念边权在图像级共享；生产路径改为小矩阵Hungarian，bitmask DP仅保留oracle。第一版不声称refute，只使用正边support与skip=Unknown，避免重建IDEA-163失败三态。
slimmed_execution: 复用一次完整测试、每个预注册solver path一次GPU micro-batch、一次共享边权Reader训练；real/shuffled共享Reader和图像分数，不为三个求解条件重复训练。先运行capacity-1；若失败自动运行rescue-1 capacity-2；仍失败运行rescue-2 2×2相邻区域节点。三个条件只改变冻结边权上的求解配置。
minimal_falsification: 固定CUB formal-seen、seed 7、100类训练共享概念Reader、50类图像完全不进Reader梯度；全部200类文本允许。Reader必须以原soft-attention evidence口径复现固定eligible_indices的27概念pseudo-unseen median AUC≥0.65且打乱标签≤0.55；edge-max AUC另报，不得冒充Reader复现。固定抽取500张唯一隔离类图像，真类均至少含2个概念，所有50类竞争。主条件要求：(1)独立路径重复patch率≥10%；(2)capacity-1精确匹配相对同表示independent准确率+1.0个百分点、净纠正≥20且损坏少于纠正；(3)仅在前两门通过后，对250张图删除cached分类实际assignment，raw重编码assignment一致率≥95%、patch同源≥0.99且删除下降优于不重叠随机区域≥70%；(4)Top-6生产匹配与全576 bitmask DP在32例逐值误差≤1e-6，Hungarian平均求解≤1ms/类。任一失败则本条件drop并按顺序进入两次补救。
rescue_plan:
  - rescue_1: 仅把每patch容量从1改为2，解决喙/头等真实重叠；其余边权、Reader、样本、门槛不变。仍要求相对independent准确率+1.0、净纠正≥20和删除≥70%。
  - rescue_2: 回到容量1，但把2×2相邻patch组成区域节点，边权取区域内均值，解决单token过细与邻接部位；其余不变。
  - 三个条件全部失败后将IDEA-165置为rejected，禁止继续调容量、区域尺度、概念数、Top-R或融合权重，自动转向新的问题类别。
paper_level_claim: Gate和后续正式U/S/H均成立后，可窄化声称“在文本唯一概念证据图上以可跳过、容量受限的精确assignment替代独立局部聚合，使GZSL产生全局一致且可删除验证的类别解释”；DP本身、图论或OT均不得声称首次。正式目标为同checkpoint Full/Off双门槛并推动V4 H从78.119641达到至少80.0。
evidence_refs:
  - research/ideas/IDEA-162_learnable_concept_readout_probe.md
  - research/ideas/IDEA-163_tri_state_evidence_predicate_set.md
  - research/ideas/IDEA-164_observable_signed_evidence.md
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-162-learnable-concept-readout-seed7/result.json@sha256:4f73cbbd0308b9e96af1342df2f45bb2f89ed0ffb8ec1bf6001e835101b574af
success_condition: capacity-1或两次预注册补救之一同时通过Reader复现、准确率、净纠正、删除与复杂度门槛，才允许进入正式Innovation模型集成；单纯重复率归零或解释图更漂亮不算成立。
failure_condition: Reader不复现IDEA-162、独立重复率不足、三个solver条件均无≥1点class-disjoint增益、净纠正不足、删除不优于随机或复杂度超过门槛，均按预注册自动drop并换方向。
owner_decision: 2026-08-29 owner授权无需逐次批准，IDEA失败后最多补救2次，仍失败自动换方向；优先尝试证据图＋状态压缩DP，并以正式H≥80为最终目标。

## 2026-08-29 三条件结果

- 分支：`exp/v4/innovation/innovation-005-constrained-evidence-graph`
- 运行commit：`273fd7a6ad36302bd4ef7798dfabbcfd32ce164a`
- config SHA：`8c7cb224904e6dd24e64760e79d6b99d4c3616c0b46cf8f750c97a1f98ef7f41`
- 输出：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-165-constrained-evidence-graph-seed7/result.json`
- result SHA：`bffb20c37106c8de000ca5d0503ffcdd8017d3f0c57afc363147b9510b0e492d`
- Reader复现：真实soft-attention median AUC=`0.785477`，打乱标签=`0.450058`，Reader门通过；edge-max median AUC=`0.782301`。因此求解失败不能归因于IDEA-162边权未复现。

| 条件 | independent acc | constrained acc | Δacc | corrected/damaged/net | duplicate rate | 决策 |
|---|---:|---:|---:|---|---:|---|
| capacity-1主条件 | 8.4% | 8.2% | -0.2pp | 1/2/-1 | 35.0% | fail |
| capacity-2补救 | 8.4% | 8.2% | -0.2pp | 0/1/-1 | 35.0% | fail |
| 2×2区域补救 | 0.2% | 0.2% | 0.0pp | 0/0/0 | 0.2% | fail |

- 容量1确实面对真实重复证据：独立路径35%的真类解释复用同一patch；但禁止复用没有提高分类，反而净损坏1张。说明角色证据真实允许共享区域，或Reader重复响应并不是错误类别的主要成因。
- 容量2没有恢复正收益，证明失败不只是“一patch一角色过严”。
- 2×2区域几乎消除重复率，但整个表示的准确率降到0.2%，区域均值抹掉了Reader可用的局部信号。
- 三种生产Hungarian与full-576 bitmask oracle在真实32例逐值误差均为0，平均求解约0.038–0.052ms/类；失败不是求解器错误或复杂度造成。
- 三个条件都未过准确率/净纠正预门，因此按预注册不执行250图删除，避免为无效求解器重复CLIP重编码。
- 最终决策：`reader_gate=true / gate_fail / rejected`。容量1与两次补救预算全部用尽；不增加容量、区域尺度、OT、A*、Top-K或融合权重，自动停止局部视觉证据图方向并切换问题类别。

## 审核记录

- 预冻结`6f08f7a` Round 1发现复杂度假门、Reader AUC换义、删除assignment错配、概念/样本轴未锁及准入字段缺失，集中修复到`697b520`。
- Round 2发现真实32例未直接绑定生产Hungarian，修复后最终commit `273fd7a`专项`6 passed`、完整`543 passed, 2 warnings, 3 subtests passed`。
- 两名最终Agent均报告`P0=0 / P1=0`并明确“双Agent交叉审查通过（第2轮通过）”；P2仅为TF32披露、非空assignment条件化删除、threshold tie、字段命名及故障注入测试固化，不影响本次失败结论。
old_solution_path: `patch→各角色独立max/attention→角色分数平均→类别排序`，同一patch可无限复用。
new_solution_path: `共享概念Reader→patch—概念证据图→Top-6精确保留→容量约束可选匹配→assignment类别分数与解释`。
principle_difference: 旧路径假设局部证据条件独立；新路径把证据所有权作为离散资源约束并全局求解。
non_equivalence_test: 在同一冻结边权上，只有存在重复patch竞争时约束匹配才与独立聚合不同；必须观察到≥10%重复率且产生真实净纠正，不能靠重训Reader或融合权重解释差异。
minimal_viability: 固定27概念Reader复现、500张唯一class-disjoint图、50类全竞争、Top-6与全图精确等价、至少一个solver满足准确率/净纠正/删除/复杂度全门。
current_advantage: IDEA-162已提供可迁移共享概念信号；IDEA-165只检验全局求解，不再次发明边权、可见性或三态Head。
performance_status: pre-run，尚无IDEA-165数值；正式V4父H=78.119641，最终目标H≥80仅在Gate成立后验证。
failure_boundary: 三种预注册solver均失败后，证据图容量约束方向用尽，不增加新容量、区域尺度、OT、A*、Top-K或融合残差。
