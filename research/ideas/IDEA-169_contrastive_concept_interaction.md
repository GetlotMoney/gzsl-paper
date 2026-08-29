# IDEA-169：Contrastive Concept Interaction

idea_id: IDEA-169
source_type: IDEA-168_result + owner_rescue + first_principles
status: proposed_owner_approved_for_rescue1
problem_category: visual_grounding
mechanism_tags: [input_intervention, contrastive_concept_margin, nuisance_cancellation, non_additive_interaction]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs: [IDEA-162, IDEA-168]
problem: IDEA-168发现跨扰动交互符号稳定，但目标区域对的交互绝对值不优于困难或随机区域对。原始目标概念logit同时携带通用鸟体、角色部位和具体概念三类变化，通用图像损伤可能淹没真正的概念特异交互。
hypothesis: 在固定原图目标Attention的同一空间读出下，用“目标概念相似度减去同角色相似竞争概念均值”的对比margin替代裸目标logit，可以消除通用鸟体与角色级损伤；若跨区域依赖属于具体文本概念，目标区域对的`abs(eta_margin)`应稳定超过困难与随机对照。
core_change: 对每个目标概念，预先从同角色、当前类别不属于其成员、概念频率在2倍以内的候选中，按文本查询余弦选择最多3个最近竞争概念。所有概念共用目标概念的原图固定Attention，定义`margin=sum_p w_target(p)*(sim_target(p)-mean_k sim_competitor_k(p))`。A、B、A并B每次完整重跑CLIP，但不重新计算主读出Attention；以margin计算非加性交互。动态Attention与裸目标logit只旁报。
old_signal_or_primitive: IDEA-168直接在目标概念裸logit上测区域交互，无法扣除对所有同角色概念共同发生的通用视觉损伤。
new_signal_or_primitive: 同一空间权重下的目标概念—同角色竞争概念差分交互，用差分中的差分隔离具体概念信号。
paradigm_shift: 继承IDEA-168的输入干预新信号，但把被干预的学习对象从绝对概念响应收窄为概念间相对证据；本卡是已拒绝Gate的测量补救，不把它单独包装成已成立Innovation。
why_not_module: 不增加分类Head、Gate、残差或融合，也不进入TG+GTD；只检验概念特异性是否被绝对logit中的公共扰动项遮蔽。
closest_paradigm_work:
  - One Explanation is Not Enough: Structured Attention Graphs for Image Classification（NeurIPS 2021）已研究区域组合对分类置信度的影响；本补救不主张首次组合解释。
  - Information-Theoretic Visual Explanation for Black-Box Classifiers（arXiv 2009.11150）已研究类别特异PMI；本补救只检验class-disjoint共享文本概念的相对交互margin。
minimal_falsification: 完全复用IDEA-168的seed7、100/50类别隔离、固定500图、每图最多3概念、Top-2不重叠4×4-patch窗口、mean-fill/local-blur、困难与随机两类对照、两层bootstrap、最少100对/25类、patch余弦0.99和双卡身份合同。唯一语义变化是主分数改为固定目标Attention下的对比概念margin；每个目标至少需要2个、最多3个同角色文本近邻。四个`magnitude_excess`均要求95% CI下界>0且标准化效应≥0.2，跨扰动符号一致性的95% CI下界>0。任一失败则补救1失败，不调邻居数、文本距离、窗口、阈值或统计门。
paper_level_claim: 只有本Gate与后续任务优势都成立后，才能声称“相对概念证据而非绝对概念响应揭示了class-disjoint细粒度概念的跨区域依赖”；不得声称首次对比解释或因果定位。
old_solution_path: `目标概念裸logit→区域单删/联合删→eta`。
new_solution_path: `固定目标Attention→目标概念减同角色近邻→对比margin→区域单删/联合删→eta_margin`。
principle_difference: 公共视觉损伤在同空间、同角色概念差分中被抵消；只有改变目标与竞争概念相对关系的区域作用被保留。
non_equivalence_test: 若只给所有同角色概念加入相同扰动项，对比margin及其eta必须保持不变；真实500图上目标区域的`abs(eta_margin)`还必须超过困难和随机区域对。
minimal_viability: Reader复现、至少100个合格概念区域对覆盖25类、四项概念特异性效应门与跨扰动稳定门全部通过；只记proof_of_path，不直接进入正式H训练。
current_advantage: none；IDEA-168只证明符号稳定，四项概念特异性门全部失败。
performance_status: rescue1_pending；正式V4 H=78.119641，本Gate不报告H/U/S/ZS。
problem_family: 公共图像损伤掩盖细粒度概念相对证据的局部或跨区域依赖。
shared_bottleneck: 绝对概念响应混合通用前景、角色部位和具体概念变化。
reusable_capability: 若成立，可提供对通用视觉损伤不敏感的概念相对干预读出。
coverage_and_transfer: 当前只验证CUB seed7的pseudo class-disjoint划分；跨seed与跨数据集未知。
frontier_shift: unknown。
downstream_effects: Gate通过后才允许设计交互感知分类；Gate失败不进入主模型。
failure_boundary: 竞争概念只来自同角色共享文本簇，不使用人工属性、部位、框或unseen图像梯度。补救1失败后不在本卡调参；owner已授权的补救2必须另建Idea并只改变干预真实性，使用内容感知补全替代mean-fill/blur。
owner_decision: 2026-08-29 owner在IDEA-168失败后明确要求“开始补救”，批准先执行概念对比差分补救；若失败再自动进入一次内容感知补全补救。
