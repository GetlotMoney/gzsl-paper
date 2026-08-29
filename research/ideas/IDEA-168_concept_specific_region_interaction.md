# IDEA-168：Concept-Specific Region Interaction

idea_id: IDEA-168
source_type: experiment_result + first_principles + owner_hypothesis + nearest_work_boundary
status: proposed_owner_approved_for_gate0
problem_category: visual_grounding
mechanism_tags: [input_intervention, concept_specificity, non_additive_interaction, class_disjoint_transfer]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs: [IDEA-162, IDEA-165, IDEA-167]
problem: IDEA-162证明共享文本概念可从冻结CLIP patch中学习读取，但IDEA-165证明按patch复用次数施加容量约束不会改善分类。当前真正未知的是：同一概念是否需要两个不同图像区域共同提供证据，以及这种联合影响是否超过两个区域独立影响之和，而不是通用鸟体信息受损或Attention自证。
hypothesis: 在50个完全不进入Reader梯度的class-disjoint类别图像上，同一共享文本概念的两个不重叠高响应区域会产生跨两种输入扰动方向一致、相对Attention强度匹配无关概念区域更强的非加性交互。
core_change: 不训练新模型、不实现超图或分类融合。复现IDEA-162共享Reader后，只在500张分层抽取的pseudo-unseen图像上，用576个原生patch响应提出候选位置；每图最多3个超过训练类校准阈值的概念，每概念只取Top-2不重叠的固定4×4-patch窗口。分别扰动A、B、A并B并重新运行完整CLIP，以未饱和概念logit计算单区下降和`eta=drop(A并B)-drop(A)-drop(B)`，再减去Attention强度、区域位置和概念频率匹配的无关概念对照。
old_signal_or_primitive: 冻结图像的静态patch分数、独立概念响应或patch复用次数。
new_signal_or_primitive: 输入级干预产生的概念特异跨区域交互量；区域对不再被看成两个可独立相加的静态分数。
paradigm_shift: 从“读取并相加局部响应”转为“通过真实输入扰动测量两个区域对同一概念的条件联合效应”。
why_not_module: Gate 0没有新增Head、Gate、重排器、融合权重或分类残差，也不以Attention窗口本身作为部位定位；它只验证父框架不存在的干预交互信号是否真实存在。只有后续证明该信号改善任务，才可能登记为Innovation。
closest_paradigm_work:
  - One Explanation is Not Enough: Structured Attention Graphs for Image Classification（NeurIPS 2021，https://proceedings.neurips.cc/paper/2021/hash/5e751896e527c862bf67251a474b3819-Abstract.html）已用beam search寻找多种区域解释，并表示区域组合对分类置信度的影响；因此不能声称首次多区域组合解释。
  - Explanations for Occluded Images（ICCV 2021，https://openaccess.thecvf.com/content/ICCV2021/html/Chockler_Explanations_for_Occluded_Images_ICCV_2021_paper.html）已用因果责任分析黑盒分类器的遮挡解释；本Idea不声称输入扰动等同真实因果删除。
  - Information-Theoretic Visual Explanation for Black-Box Classifiers（arXiv 2009.11150，https://arxiv.org/abs/2009.11150）已比较移除输入特征前后的信息增益与类别特异PMI；本项目潜在边界仅是共享文本概念、class-disjoint GZSL迁移和跨区域非加性交互能否带来任务优势。
minimal_falsification: 固定seed7和IDEA-162的100/50类别隔离；100类seen图像用于训练共享Reader及校准候选阈值，50类图像完全不进入梯度。分层抽500张唯一pseudo-unseen图像；每图最多3个过阈值概念，每概念只取两个不重叠4×4-patch窗口。均值填充与局部模糊均需重新运行完整CLIP。主统计使用未饱和概念logit，报告`eta=drop(A并B)-drop(A)-drop(B)`及减去困难对照后的`eta_specific`。困难对照匹配窗口面积、边缘/内部位置、Attention强度和概念频率。按“类别→图像”两层bootstrap；只有两种扰动下`eta_specific`同号、95%区间均排除0且标准化效应绝对值不小于0.2，Gate 0才通过。否则停止，不实现超图、DP、冗余或最小充分集合。
paper_level_claim: Gate 0与后续任务优势均成立后，只能窄化声称“在无人工属性的class-disjoint GZSL中，学习到的共享文本概念呈现可迁移的概念特异跨区域非加性交互，并可作为交互感知识别信号”；不得声称首次区域组合解释、因果视觉解释或超图推理。
old_solution_path: `冻结图像→独立patch/概念分数→直接聚合或类别相似度`。
new_solution_path: `候选区域提出→输入级单删/联合扰动→概念特异交互量→后续交互感知识别（仅在Gate 0后）`。
principle_difference: 旧路径假设区域贡献可静态独立读取；新路径把核心对象定义为必须通过联合干预测出的条件交互，单区分数不能重参数化得到该量。
non_equivalence_test: 在控制区域面积、位置、Attention强度和概念频率后，目标概念区域对的`eta`仍应显著区别于无关概念区域对，并在两种扰动下保持方向一致；若差异消失，则新路径只是静态Attention依赖或通用鸟体损伤。
minimal_viability: 500张class-disjoint图像、困难对照、两种扰动和两层bootstrap下，`eta_specific`达到预注册统计门；这只证明`proof_of_path`，不证明分类优势。
current_advantage: none；目前只有IDEA-162的共享概念可读性和IDEA-165容量约束失败作为前置证据，尚未证明accuracy、speed_or_cost或generality优势。
performance_status: proof_of_path_pending；正式V4 H=78.119641，Gate 0不训练TG+GTD也不报告新H。
problem_family: 多区域共同表达一个细粒度概念；是否覆盖其他数据集或任务尚未验证。
shared_bottleneck: 静态局部打分无法区分独立证据、通用前景损伤与真正的跨区域条件交互。
reusable_capability: 若成立，可提供概念特异的区域交互测量；分类复用价值待Gate 1验证。
coverage_and_transfer: 当前只预注册CUB seed7的100/50类别隔离；跨seed、SUN、AWA2均未验证。
frontier_shift: unknown；Gate 0只检验新信号是否存在。
downstream_effects: 只有Gate 1证明任务优势后，才考虑交互感知分类或证据搜索；Gate 0不预建这些模块。
failure_boundary: Attention峰值只表示Reader依赖候选，不等于真实部位定位；均值填充和局部模糊只支持“扰动稳健性”，不等于真实因果删除。Gate 0失败后不调窗口尺度、概念数、阈值、扰动类型或组合枚举，不实现超图、DP、保留充分性和分类融合。
owner_decision: 2026-08-29 owner在确认最小问题与方案后回复“开始吧”，批准IDEA-168通过范式候选的Gate 0准入并从FRAMEWORK-V4准确父commit独立执行；Gate 0通过前仅记录为proof_of_path，不登记为已成立Innovation。
