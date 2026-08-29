# IDEA-163：Tri-State Evidence Predicate Set

idea_id: IDEA-163
source_type: experiment_result + first_principles + owner_hypothesis + nearest_work_boundary
status: rejected
problem_category: visual_grounding
mechanism_tags: [text_only_predicates, support_refute_unknown, class_bag_learning, class_disjoint_transfer, evidence_reasoning]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs:
  - IDEA-160
  - IDEA-161
  - IDEA-162
problem: V4把类别压缩为单点原型，相似度低时无法区分“图像明确反驳该描述”和“相关部位根本没有被拍到”；IDEA-162证明冻结CLIP patch含有可由seen类概念弱监督学习并迁移到class-disjoint类别的信号，但现有二元探针不能处理遮挡、反证、背景捷径、角色缺失和可解释分类贡献。
hypothesis: 将类别表示为六个文本视觉谓词，并让共享读取器为每个谓词输出`support / refute / unobserved`三态证据，可在不使用人工属性、部位、框或true-unseen图像梯度的前提下，把seen类图像包中学到的细粒度证据规则迁移到pseudo-unseen类别；本轮最小证伪只检验能否解释并纠正全局Mean8的细粒度混淆，通过后才允许在正式Experiment中验证TG+GTD。
core_change: 前六角色文本不再只被平均成一个类别点，而是形成六个可验证视觉谓词。共享query-conditioned读取器从576个冻结CLIP patch中提取证据；同类多图组成bag以允许谓词在部分图像中不可见。每个候选谓词固定与同角色4个文本近邻比较，训练类正/负分数确定角色级绝对阈值：候选超过阈值且压过近邻为support，近邻超过阈值且压过候选为refute，两边都不够强为unobserved。推理以角色级有符号证据账本聚合，并按已观察角色数归一化。
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
minimal_falsification: 固定CUB formal-seen 150类、seed 7，将100类用于训练、50类完全隔离评估；只用前六角色无类别名文本、正式最终层576-patch及seen图像。训练共享rank-64 query读取器，bag size=4；每个正谓词固定与同角色文本最近的4个其他类谓词对比，三态阈值只由100类训练正/负分数确定，因此不受100/50候选数量影响。评估同时要求：(1)50类中真实谓词对同角色困难反例的pairwise accuracy≥65%；(2)在Mean8的Top-1错误且真类位于Top-5的样本中，三态证据使真类证据高于错误Top-1的比例≥60%，且只有错误类证据严格高于真类才计为反转，破坏率<10%；(3)对200个唯一seen图像重新编码原图，只选择账本中实际support的谓词，删除其attention区域后完整`候选+4反证`有符号贡献下降应大于随机同面积区域，比例≥70%。任一条件失败即drop当前三态公式，不通过增加层、prompt、Top-K或容量补救。
paper_level_claim: 若三项最小证伪及后续正式GZSL门槛均成立，可窄化声称“使用文本唯一角色描述和seen类图像包学习可迁移的三态视觉谓词集合，使GZSL从点原型兼容度转向可观察证据推理”；在完成更系统近期检索前禁止写“首次”。
evidence_refs:
  - research/ideas/IDEA-160_full_resolution_concept_grounding.md
  - research/ideas/IDEA-161_intermediate_patch_concept_signal.md
  - research/ideas/IDEA-162_learnable_concept_readout_probe.md
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-162-learnable-concept-readout-seed7/result.json@sha256:4f73cbbd0308b9e96af1342df2f45bb2f89ed0ffb8ec1bf6001e835101b574af
success_condition: 三项最小证伪必须同时通过，且打乱谓词标签对照必须分别低于pairwise门槛、低于纠错门槛并不满足低破坏门槛；通过只允许进入正式Innovation实现，不等于论文创新或H成立。
failure_condition: 三项任一失败、证据只在训练类有效、删除测试不优于随机、unknown由候选分数自行逃避、或证据纠错伴随≥10%正确样本破坏，则状态置为rejected并停止该公式。
owner_decision: 2026-08-29 owner回复“开始”，批准IDEA-162作为supported证据、批准建立IDEA-163、确认父commit为52088f69d7ac4e574e7b63c28b21ac0da7789933，并批准创建独立实验分支执行最小验证。

## 2026-08-29 最小证伪结果

- 分支：`exp/v4/innovation/innovation-003-tri-state-evidence-set`
- 最终运行commit：`befbe22b7b86313f212cb779d8342fb8a4500501`
- config SHA：`19639e4fc69ba5404deee4f14a2552b4c718fe38640cb3a47b70e46fccd81f8e`
- 输出：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-163-tristate-predicate-seed7/result.json`
- result SHA：`3373601ac0736936d193136895c29178d58619ad585a7590d38f03c9db4bfc91`
- 数据边界：训练与阈值只用100个formal-seen类；50个pseudo-unseen类只评估；true-unseen图像未读取、未参与梯度或筛选。

| 门槛 | 要求 | 真实结果 | 判定 |
|---|---:|---:|---|
| 同角色困难反例pairwise accuracy | ≥65% | 37.19% | fail |
| Mean8错误对中真类证据优于错误Top-1 | ≥60% | 29.39% | fail |
| Mean8正确样本被证据反转 | <10% | 92.28% | fail |
| 删除support区域优于随机区域 | ≥70% | 48.00% | fail |
| 原图重编码与正式patch最小余弦 | ≥0.99 | 0.82318 | fail |

- 打乱谓词对照按预注册要求分别失败：pairwise=`37.27%`、错误纠正=`59.14%`、正确样本破坏=`97.72%`，没有产生对照假通过。
- IDEA-162的`0.774`中位AUC证明共享的31个概念簇可以被二元读取，但不能推出“每个类别的六条独特角色文本”能够形成可迁移的同角色反证和50类证据推理。当前bag训练把更简单的概念成员关系任务扩张成类特定谓词排序后，迁移性消失。
- attention删除结果低于随机且大量support贡献在删除后反而上升，说明注意力权重不是可信因果证据区域；不能把热图解释包装为三态证据。
- 正确样本92.28%的反转说明当前有符号账本不是父模型的安全补充，而是几乎独立且错误的分类器。
- 原图重编码与缓存patch的最小余弦未过身份门槛，因此删除解释链还存在局部token同源风险；鉴于其余四个核心门槛也大幅失败，不为该解释链单独补救或重跑。
- 最终决策：`minimal_falsification_fail / rejected`。不登记V4 TRY、不启动正式U/S/H训练、不通过增加层、prompt、Top-K、rank、bag size或额外loss继续补救该公式。

## 审核记录

- 预冻结commit `81164e78a4ca0b4e3a1286a9969997b0fd2fd034` Round 1：两名独立Agent合计发现阻断P1，集中修复候选数污染、删除账本、打乱门槛、all-unknown、三态语义及资产/merge身份。
- post-fix commit `befbe22b7b86313f212cb779d8342fb8a4500501`：专项`6 passed`；完整`543 passed, 2 warnings, 3 subtests passed`。
- Round 2：两名新的独立Agent均报告`P0=0 / P1=0 / P2=2`并明确“第2轮通过”。P2为反例身份随固定50类集合变化、merge未反读制品内部元数据；不影响本次固定协议失败结论。
