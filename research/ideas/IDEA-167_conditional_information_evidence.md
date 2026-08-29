# IDEA-167：Conditional Information Evidence

idea_id: IDEA-167
source_type: experiment_result + first_principles + owner_hypothesis + nearest_work_boundary
status: revised
problem_category: visual_grounding
mechanism_tags: [conditional_information_gain, shared_evidence, synergy, redundancy, minimal_sufficient_subset]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs: [IDEA-162, IDEA-165]
problem: IDEA-165发现35%的真类解释重复使用同一patch，但容量1/2约束均使准确率下降0.2pp，证明patch复用次数不是错误证据的充分判据。同一区域可能合法支持多个谓词，多个区域也可能协同支持一个谓词；真正未知的是共享、协同和冗余能否由真实干预下的条件信息增益稳定识别。
hypothesis: 在固定IDEA-162共享概念Reader和class-disjoint图像上，区域对角色概念与类别margin的干预增益可稳定分解为共享、协同和冗余；若由条件增益选出的少量区域同时满足删除必要性和保留充分性，并跨两种干预一致，则有资格把GZSL重写为最小充分证据超图搜索。
core_change: 不训练新网络、不做超图搜索。固定500张pseudo-unseen图像与36个6×6区域；每个角色按冻结Reader选择Top-2区域。对A、B、A+B及随机同面积区域分别执行局部模糊和均值填充并重新编码。以真实类别相对最强竞争类的margin和角色概念分数计算`g(R|S,c)`及交互`eta(A,B)=g(B|A,c)-g(B|empty,c)`。同时测删除选中集合的必要性与只保留选中集合的充分性。
old_signal_or_primitive: patch独立分数、attention或复用次数决定证据重要性。
new_signal_or_primitive: 真实输入干预下、依赖已有区域集合的条件margin增益，以及连接多区域—多谓词的共享/协同/冗余超边。
paradigm_shift: 从局部边权聚合改为条件信息交互；区域价值不再固定，而由它相对已选择证据提供的新信息决定。
why_not_module: 超图、集合覆盖和搜索算法尚不进入模型；Gate 0只验证新原语是否存在。若现象不稳定或只产生漂亮解释而无删除/保留合同，不得登记Innovation。
closest_paradigm_work:
  - Structured Attention Graphs / Minimal Sufficient Explanations（NeurIPS 2021）已搜索多个最小充分图像解释；本Idea的潜在区别仅在文本角色谓词、class-disjoint GZSL和多谓词条件增益。
  - Explanations for Occluded Images（ICCV 2021）已研究组合最小解释；不能声称首次组合区域解释。
  - Information-Theoretic Visual Explanation（arXiv 2009.11150）已定义信息增益与PMI归因；本Idea必须证明GZSL谓词超边与分类迁移的额外价值。
minimal_falsification: seed7固定100类仅用于复现IDEA-162 Reader，50类图像完全不进梯度；从真类至少含2个共享概念的50类中分层抽500张唯一图像。36个非重叠区域；每角色Top-2。两种干预均计算单区、相邻Top区组合、随机同面积、删除全部选中区和只保留选中区。主门同时要求：共享证据样本≥20%；跨两种干预稳定的协同或冗余样本≥15%；选中集合删除优于随机≥70%；只保留选中集合仍保留原真类margin至少90%的样本≥60%；两种干预对共享/协同/冗余类型一致率≥70%。Reader soft-attention median AUC必须≥0.65、打乱标签≤0.55。任一失败则Gate 0失败，不实现超图搜索。
paper_level_claim: Gate 0、后续超图搜索与正式H均成立后，只能窄化声称“用文本谓词条件信息增益构造class-disjoint最小充分视觉证据超图”；不得声称首次最小充分解释、信息增益归因或视觉超图。
old_solution_path: `固定patch分数/复用规则→局部聚合`。
new_solution_path: `真实区域干预→条件margin增益→共享/协同/冗余超边→最小充分证据集合`。
principle_difference: 旧路径把区域价值视为静态；新路径把价值定义为相对已有证据的条件变化。
non_equivalence_test: 同一块区域可同时对多个谓词有独立正增益；重复背景在已有区域后条件增益接近0；相邻区域联合增益可显著超过单区增益和。
minimal_viability: 五项现象/因果/充分性硬门、双干预一致、500唯一class-disjoint图和Reader复现全部成立。
current_advantage: IDEA-162提供可迁移共享概念Reader；IDEA-165的35%重复率与容量惩罚失败为“复用不等于冗余”提供直接本地证据。
performance_status: pre-run；尚无IDEA-167结果，正式V4 H=78.119641，H≥80只在超图Gate成立后验证。
failure_boundary: Gate 0失败后不实现集合覆盖、子模优化、分支限界或超图网络；不得靠搜索阈值、区域数、解释大小或融合权重补救。
owner_decision: 2026-08-29 owner明确“开始尝试”，并授权失败后自动处理、无需逐次批准。

## 2026-08-29 运行前修订

- 本Idea没有运行、没有结果，也没有进入实现。
- 原Gate同时混入共享、协同、冗余和最小充分集合，无法在失败时定位是哪条假设不成立；固定36区与保留90%图像也会引入过强的人为表示和分布外干预。
- owner将当前问题收窄为“class-disjoint图像中是否存在概念特异的跨区域非加性交互”。该修改改变了核心可证伪假设，因此不覆盖本卡，另立`IDEA-168`。
- 本卡永久保留为`revised`历史；禁止按这里的五项联合硬门、固定36区或最小充分集合继续运行。
