# IDEA-165：Constrained Evidence Graph Search

idea_id: IDEA-165
source_type: experiment_result + first_principles + owner_hypothesis + nearest_work_boundary
status: proposed_owner_approved_for_three_attempts
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
core_change: 复现并冻结IDEA-162的共享概念Reader，只使用其已经验证的27个可迁移概念，不训练新的视觉网络。每张图一次性生成`concept×576 patch`边权；候选类别通过文本簇成员关系得到最多六个角色概念节点。独立基线让每个角色各取最高正边；新求解器允许角色Unknown=0，并要求patch容量。角色数最多6，使用状态压缩DP求精确最大权匹配。容量1时每个角色只需保留Top-6 patch仍不损失最优解：若最优边排名超过6，至少一个更高边未被其他最多5个角色占用，可替换提高目标；因此每类DP最多扫描36个patch并保持全50/200类竞争，不使用类别Top-K。
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
optimization_round_2: 用Top-R精确保留定理把576 patch缩为每角色Top-6，并将概念边权在图像级共享；DP状态仅64个mask。第一版不声称refute，只使用正边support与skip=Unknown，避免重建IDEA-163失败三态。
slimmed_execution: 复用一次完整测试、每个预注册solver path一次GPU micro-batch、一次共享边权Reader训练；real/shuffled共享Reader和图像分数，不为三个求解条件重复训练。先运行capacity-1；若失败自动运行rescue-1 capacity-2；仍失败运行rescue-2 2×2相邻区域节点。三个条件只改变冻结边权上的求解配置。
minimal_falsification: 固定CUB formal-seen、seed 7、100类训练共享概念Reader、50类图像完全不进Reader梯度；全部200类文本允许。Reader与IDEA-162相同：27个满足至少2个训练正类与1个评估正类的共享概念，1000 updates、rank64、自然类别无关prompt及class-level弱监督，必须复现pseudo-unseen median concept AUC≥0.65且打乱标签≤0.55，否则求解Gate无效并立即停止。对同一冻结边权比较independent与capacity-1 DP，所有50类竞争且只评价真类至少含2个可迁移角色概念的图像。主条件同时要求：(1)独立路径的重复patch率≥10%，证明约束有真实对象；(2)DP相对independent的50类准确率提高≥1.0个百分点、净纠正≥20且损坏少于纠正；(3)在250张隔离类图像上，对真类DP assignment的mean-fill删除使类别证据下降大于相同数量随机区域的比例≥70%；(4)精确Top-6 DP与全576 DP在固定小样本逐值一致，平均求解开销不超过independent路径的5倍。任一研究门槛失败则本条件drop。
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
