# IDEA-163：Tri-State Evidence Predicate Set

idea_id: IDEA-163
source_type: experiment_result + first_principles + owner_hypothesis + nearest_work_boundary
status: proposed_owner_approved_for_minimal_falsification
problem_category: visual_grounding
mechanism_tags: [text_only_predicates, support_refute_unknown, class_bag_learning, class_disjoint_transfer, evidence_reasoning]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs:
  - IDEA-160
  - IDEA-161
  - IDEA-162
problem: V4把类别压缩为单点原型，相似度低时无法区分“图像明确反驳该描述”和“相关部位根本没有被拍到”；IDEA-162证明冻结CLIP patch含有可由seen类概念弱监督学习并迁移到class-disjoint类别的信号，但现有二元探针不能处理遮挡、反证、背景捷径、角色缺失和可解释分类贡献。
hypothesis: 将类别表示为六个文本视觉谓词，并让共享读取器为每个谓词输出`support / refute / unobserved`三态证据，可在不使用人工属性、部位、框或true-unseen图像梯度的前提下，把seen类图像包中学到的细粒度证据规则迁移到pseudo-unseen类别；三态证据应能解释并纠正全局Mean8/TG+GTD的细粒度混淆，而不是成为另一条不可解释分数残差。
core_change: 前六角色文本不再只被平均成一个类别点，而是形成六个可验证视觉谓词。共享query-conditioned读取器从576个冻结CLIP patch中提取证据；同类多图组成bag以允许谓词在部分图像中不可见，同角色文本近邻定义困难反证，角色可观察性由该图像对该角色全部候选的共同响应决定、不得由单个候选自报。推理以角色级有符号证据账本聚合：support为正、refute为负、unobserved严格为零，并按可观察角色数归一化。
old_signal_or_primitive: seen类别标签监督下的单点文本原型与图像全局相似度；低相似度同时混合反证、遮挡和未知。
new_signal_or_primitive: 类级文本视觉谓词、同类图像包、同角色困难反证与跨类别共享读取共同定义三态证据；类别原语由单点向量变为带状态、区域、置信度和有符号贡献的谓词集合。
paradigm_shift: GZSL从“图像与类别原型有多像”改写为“图像对类别提出的每条视觉命题是支持、反驳还是未观察”，最终预测由全局语义先验与可观察证据似然共同决定。
why_not_module: Cross-attention、bag聚合和同角色对比只是识别三态隐变量的实现；候选的基本学习对象、监督粒度、类别表示和推理合同都发生变化。若最终仅表现为给V4增加一个patch残差、无法输出可信三态账本或关闭后不能返回父模型，则不得作为该Idea成立。
closest_paradigm_work:
  - PAPER-003（CVPR 2020 Dense Attribute-Based Attention）使用人工类级attributes进行逐属性视觉对齐、属性评分与自校准；IDEA-163禁止人工attributes，并显式区分反证与未观察。
  - Attribute Prototype Network（NeurIPS 2020，https://proceedings.neurips.cc/paper/2020/file/fa2431bf9d65058fe34e9713e32d60e6-Paper.pdf）用类级人工attributes回归、去相关并定位局部属性；没有文本唯一谓词、图像包missingness或三态证据账本。
  - TransZero（AAAI 2022，https://ojs.aaai.org/index.php/AAAI/article/view/19909）以人工attribute语义引导视觉局部化和embedding；仍把局部化结果用于兼容度分类，不建模`support/refute/unobserved`。
  - CREST（arXiv 2024，https://arxiv.org/abs/2404.09640）使用attribute定位、Evidential Deep Learning和不确定性融合处理hard negatives；最接近“证据/不确定性”边界，但仍依赖人工attributes，未把每个文本谓词的未观察与反证分开，也没有类图像包三态推理。
  - Counterfactual ZSL（CVPR 2021，https://openaccess.thecvf.com/content/CVPR2021/html/Yue_Counterfactual_Zero-Shot_and_Open_Set_Visual_Recognition_CVPR_2021_paper.html）生成样本级反事实并以一致性判断seen/unseen；不学习patch级文本谓词证据账本。
  - PAPER-008 / VADS（CVPR 2024）以视觉信息更新生成式动态语义prototype；仍以动态prototype生成特征，不把类别改写为三态谓词集合。
  - PAPER-010（CVPR 2024）使用LLM结构化文本描述适配VLM；说明“生成描述+视觉适配”本身已有先例，但没有三态证据变量和未观察边界。
minimal_falsification: 固定CUB formal-seen 150类、seed 7，将100类用于训练、50类完全隔离评估；只用前六角色无类别名文本、正式最终层576-patch及seen图像。训练共享rank-64 query读取器，bag size=4；每个正谓词与同角色文本最近的4个其他类谓词对比。阈值和温度只由100类训练分布确定。评估同时要求：(1)50类中真实谓词对同角色困难反例的pairwise accuracy≥65%；(2)在Mean8的Top-1错误且真类位于Top-5的样本中，三态证据使真类证据高于错误Top-1的比例≥60%，且对Mean8正确样本的反转破坏率<10%；(3)在固定seen图像子集上重新编码原图，删除模型声称的证据区域后对应谓词下降幅度大于随机同面积区域的样本比例≥70%。任一条件失败即drop当前三态公式，不通过增加层、prompt、Top-K或容量补救。
paper_level_claim: 若三项最小证伪及后续正式GZSL门槛均成立，可窄化声称“使用文本唯一角色描述和seen类图像包学习可迁移的三态视觉谓词集合，使GZSL从点原型兼容度转向可观察证据推理”；在完成更系统近期检索前禁止写“首次”。
evidence_refs:
  - research/ideas/IDEA-160_full_resolution_concept_grounding.md
  - research/ideas/IDEA-161_intermediate_patch_concept_signal.md
  - research/ideas/IDEA-162_learnable_concept_readout_probe.md
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-162-learnable-concept-readout-seed7/result.json@sha256:4f73cbbd0308b9e96af1342df2f45bb2f89ed0ffb8ec1bf6001e835101b574af
success_condition: 三项最小证伪必须同时通过，且打乱谓词标签对照不得通过任一主门槛；通过只允许进入正式Innovation实现，不等于论文创新或H成立。
failure_condition: 三项任一失败、证据只在训练类有效、删除测试不优于随机、unknown由候选分数自行逃避、或证据纠错伴随≥10%正确样本破坏，则状态置为rejected并停止该公式。
owner_decision: 2026-08-29 owner回复“开始”，批准IDEA-162作为supported证据、批准建立IDEA-163、确认父commit为52088f69d7ac4e574e7b63c28b21ac0da7789933，并批准创建独立实验分支执行最小验证。
