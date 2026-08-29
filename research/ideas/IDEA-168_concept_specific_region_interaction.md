# IDEA-168：Shared-Concept Cross-Region Non-Additive Interaction

idea_id: IDEA-168
source_type: experiment_result + first_principles + owner_hypothesis + nearest_work_boundary
status: rejected
problem_category: visual_grounding
mechanism_tags: [input_intervention, concept_specificity, non_additive_interaction, class_disjoint_transfer]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs: [IDEA-162, IDEA-165, IDEA-167]
problem: IDEA-162证明共享文本概念可从冻结CLIP patch中学习读取，但IDEA-165证明按patch复用次数施加容量约束不会改善分类。当前真正未知的是：同一概念是否需要两个不同图像区域共同提供证据，以及这种联合影响是否超过两个区域独立影响之和，而不是通用鸟体信息受损或Attention自证。
hypothesis: 在50个完全不进入Reader梯度的class-disjoint类别图像上，同一共享文本概念的两个不重叠高响应区域会产生跨两种输入扰动稳定、且绝对强度相对Attention匹配无关概念区域对与随机匹配区域对更大的非加性交互。
core_change: 不训练新模型、不实现超图或分类融合。复现IDEA-162共享Reader后，只在500张分层抽取的pseudo-unseen图像上，用576个原生patch响应提出候选位置；每图最多3个超过训练类校准阈值的概念，每概念只取Top-2不重叠的固定4×4-patch窗口。分别扰动A、B、A并B并重新运行完整CLIP。主统计固定原图的目标概念Attention权重，只读取扰动后patch相似度，避免softmax重新归一化凭空制造交互；动态Attention logit只旁报。以固定读出的未饱和logit计算`eta=drop(A并B)-drop(A)-drop(B)`，主比较量为`magnitude_excess=abs(eta_target)-abs(eta_control)`；不以`eta>0`作为通过条件。
old_signal_or_primitive: 冻结图像的静态patch分数、独立概念响应或patch复用次数。
new_signal_or_primitive: 输入级干预产生的概念特异跨区域交互量；区域对不再被看成两个可独立相加的静态分数。
paradigm_shift: 从“读取并相加局部响应”转为“通过真实输入扰动测量两个区域对同一概念的条件联合效应”。
why_not_module: Gate 0没有新增Head、Gate、重排器、融合权重或分类残差，也不以Attention窗口本身作为部位定位；它只验证父框架不存在的干预交互信号是否真实存在。只有后续证明该信号改善任务，才可能登记为Innovation。
closest_paradigm_work:
  - One Explanation is Not Enough: Structured Attention Graphs for Image Classification（NeurIPS 2021，https://proceedings.neurips.cc/paper/2021/hash/5e751896e527c862bf67251a474b3819-Abstract.html）已用beam search寻找多种区域解释，并表示区域组合对分类置信度的影响；因此不能声称首次多区域组合解释。
  - Explanations for Occluded Images（ICCV 2021，https://openaccess.thecvf.com/content/ICCV2021/html/Chockler_Explanations_for_Occluded_Images_ICCV_2021_paper.html）已用因果责任分析黑盒分类器的遮挡解释；本Idea不声称输入扰动等同真实因果删除。
  - Information-Theoretic Visual Explanation for Black-Box Classifiers（arXiv 2009.11150，https://arxiv.org/abs/2009.11150）已比较移除输入特征前后的信息增益与类别特异PMI；本项目潜在边界仅是共享文本概念、class-disjoint GZSL迁移和跨区域非加性交互能否带来任务优势。
minimal_falsification: 固定seed7和IDEA-162的100/50类别隔离；100类seen图像用于训练共享Reader及校准候选阈值，50类图像完全不进入梯度；全部200类文本允许构造共享概念并明确披露。分层抽500张唯一pseudo-unseen图像；每图最多3个过阈值概念，每概念只取两个不重叠4×4-patch窗口。均值填充与局部模糊均需重新运行完整CLIP。主统计固定原图目标概念Attention，以未饱和线性读出计算`eta=drop(A并B)-drop(A)-drop(B)`；`eta<0`只记互补/缺一不可候选，`eta>0`只记冗余/替代候选，二者都不作因果定性。两种对照分别为同角色、目标概念Attention强度相近且概念支持频率在2倍以内的无关概念区域对，以及随机同面积、同边缘/内部位置类型区域对。主门比较`abs(eta_target)`是否同时超过两种对照的`abs(eta_control)`，按“类别→图像”两层bootstrap；只有两种扰动下两个`magnitude_excess`的95%区间下界均大于0、标准化效应均不小于0.2，并且逐区域对的两种扰动交互符号一致性显著高于随机50%（层级bootstrap的平均符号乘积95%下界大于0），Gate 0才通过。允许不同样本分别呈现正负交互，不要求总体均值同号。否则停止，不实现超图、DP、冗余或最小充分集合。
paper_level_claim: Gate 0与后续任务优势均成立后，只能窄化声称“在无人工属性的class-disjoint GZSL中，学习到的共享文本概念呈现可迁移的概念特异跨区域非加性交互，并可作为交互感知识别信号”；不得声称首次区域组合解释、因果视觉解释或超图推理。
old_solution_path: `冻结图像→独立patch/概念分数→直接聚合或类别相似度`。
new_solution_path: `候选区域提出→输入级单删/联合扰动→概念特异交互量→后续交互感知识别（仅在Gate 0后）`。
principle_difference: 旧路径假设区域贡献可静态独立读取；新路径把核心对象定义为必须通过联合干预测出的条件交互，单区分数不能重参数化得到该量。
non_equivalence_test: 在控制区域面积、位置、Attention强度和概念频率后，目标概念区域对的`abs(eta)`仍应显著大于Attention匹配无关概念对与随机匹配对，并在两种扰动下稳定；若差异消失，则新路径只是静态Attention依赖、通用鸟体损伤或扰动伪影。
minimal_viability: 500张class-disjoint图像、两类对照、两种扰动和两层bootstrap下，`magnitude_excess`达到预注册统计门；这只证明`proof_of_path`，不证明分类优势，也不把正负号直接命名为协同或冗余事实。
current_advantage: none；目前只有IDEA-162的共享概念可读性和IDEA-165容量约束失败作为前置证据，尚未证明accuracy、speed_or_cost或generality优势。
performance_status: rejected_at_gate0；正式V4 H=78.119641，Gate 0未训练TG+GTD且未报告新H。
problem_family: 多区域共同表达一个细粒度概念；是否覆盖其他数据集或任务尚未验证。
shared_bottleneck: 静态局部打分无法区分独立证据、通用前景损伤与真正的跨区域条件交互。
reusable_capability: 若成立，可提供概念特异的区域交互测量；分类复用价值待Gate 1验证。
coverage_and_transfer: 当前只预注册CUB seed7的100/50类别隔离；跨seed、SUN、AWA2均未验证。
frontier_shift: unknown；Gate 0只检验新信号是否存在。
downstream_effects: 只有Gate 1证明任务优势后，才考虑交互感知分类或证据搜索；Gate 0不预建这些模块。
failure_boundary: Attention峰值只表示Reader依赖候选，不等于真实部位定位；均值填充和局部模糊只支持“扰动稳健性”，不等于真实因果删除。Gate 0失败后不调窗口尺度、概念数、阈值、扰动类型或组合枚举，不实现超图、DP、保留充分性和分类融合。
owner_decision: 2026-08-29 owner在确认最小问题与方案后回复“开始吧”，批准IDEA-168通过范式候选的Gate 0准入并从FRAMEWORK-V4准确父commit独立执行；Gate 0通过前仅记录为proof_of_path，不登记为已成立Innovation。

## 2026-08-29 Gate 0真实结果

- 冻结RUN commit：`92ec80a9a6220c797d64ff5457f9c3038d5c68ce`；父commit：`52088f69d7ac4e574e7b63c28b21ac0da7789933`；config SHA：`8006277083a6710d3b6b4510b82065de2eb45672b3a699144e5f20ddd5c6a643`。
- 数据边界：Reader只用100个pseudo-seen类图像训练；固定50个pseudo-unseen类各10张、共500张图只评估；正式unseen图像未使用；全部200类文本参与共享概念构造。
- Reader复现通过：pseudo-unseen中位AUC=`0.785477`，27个概念中24个AUC≥0.60。
- 形成136个合格区域对，覆盖120张图、26个类别；raw/cache patch余弦均值=`0.999915`、最小=`0.999380`。
- `mean_fill`相对hard/random的`magnitude_excess`分别为`-0.000555`（95% CI `[-0.222270,0.222464]`）与`-0.052237`（`[-0.869979,0.685480]`）。
- `local_blur`相对hard/random分别为`0.202818`（95% CI `[-0.009883,0.438766]`）与`0.305282`（`[-0.173981,0.792563]`）。四个区间均未满足下界>0，四项概念特异性硬门全部失败。
- 跨扰动符号稳定性通过：点估计=`0.869658`，95% CI `[0.729038,0.967521]`；mean-fill/local-blur的负交互比例分别为`99.26%/92.65%`。这只能说明扰动响应方向稳定，不能证明目标概念区域对比对照更特殊。
- 决策：`rejected_at_gate0 / gate_fail_stop_direction`。不调窗口、阈值、概念数、扰动或对照，不实现超图、DP、分类融合，也不报告H/U/S/ZS。
- 输出：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-168-concept-region-interaction-seed7/result.json@sha256:4aad826407f6d898ca2fe1edbce8ef26dc22ccc210038a318d90d6dd8b813387`。
- 审查：初始冻结commit `9804cbd4`发现并集中修复动态Attention假交互、资产SHA、困难对照频率和双卡环境四项P1；`984208b2`真实micro发现cuBLAS确定性入口P1；最终`92ec80a9`专项`10 passed`、整仓`547 passed, 3 subtests passed`，最终独立审查`P0=0 / P1=0，第2轮通过`。
