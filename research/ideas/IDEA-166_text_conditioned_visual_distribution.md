# IDEA-166：Text-Conditioned Visual Distribution

idea_id: IDEA-166
source_type: experiment_result + first_principles + owner_hypothesis + nearest_work_boundary
status: rejected
problem_category: class_competition
mechanism_tags: [distributional_prototype, empirical_bayes, low_rank_covariance, text_neighbor_transfer, robust_likelihood]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs: [IDEA-012, IDEA-160, IDEA-165]
problem: V4与Mean8/GTD把每个类别表示为一个点，但同一类别图像存在姿态、背景、视角和遮挡分布；seen类有视觉样本可估计分散度，unseen类只有文本点，联合竞争因此使用不对称置信度。此前共享视觉轨道只会让所有类别一起膨胀，未检验由类别文本邻域迁移的类条件协方差。
hypothesis: 在冻结CLIP全局空间中，100个开发seen类的类内视觉形状可由文本邻域预测并迁移到50个validation-unseen类；若文本条件对角形状同时超过点原型、共享度量、同体积类标量控制和错配方差控制，则类别原语从点变分布的路径得到proof-of-path支持。
core_change: 所有图像只使用现有全局768维CLIP特征。PCA只读取100类图像相对各自视觉类均值的类内残差，得到固定rank-16正交基；每类在该基上估计对角方差并向全局方差收缩，剩余752维使用共享正方差`v_perp`。50个隔离类的log-variance由V4 text-v2 Mean8文本对100个源类文本的固定Top-5邻域迁移；主评分与class-scalar/shared/shuffled控制使用相同Mean8中心和50类竞争集合。
old_signal_or_primitive: 每类单点文本原型与统一余弦温度。
new_signal_or_primitive: 每类由文本均值、文本邻域迁移的低秩视觉协方差和显式概率似然共同定义的条件分布。
paradigm_shift: GZSL从点原型最近邻改写为“哪个文本条件视觉分布最可能生成当前图像”，同时建模类内变化和类别特异不确定性。
why_not_module: PCA、经验贝叶斯收缩和Student-t只是估计/求解工具；核心是类别原语从点变分布。若结果只表现为统一temperature或seen/unseen常数校准，则不得作为Innovation。
closest_paradigm_work:
  - Class-Conditioned Deep Generative Models（arXiv 1711.05820）已用attribute条件生成类分布；IDEA-166不生成伪样本、不训练VAE，只迁移冻结CLIP残差协方差。
  - Aligned VAE（CVPRW 2019，https://openaccess.thecvf.com/content_CVPRW_2019/html/Uncertainty_and_Robustness_in_Deep_Visual_Learning/Schonfeld_Generalized_Zero-Shot_Learning_via_Aligned_Variational_Autoencoders_CVPRW_2019_paper.html）学习跨模态潜空间并生成特征；本Idea保持原CLIP空间和解析似然。
  - Over-Complete Distribution（CVPR 2020，https://openaccess.thecvf.com/content_CVPR_2020/html/Keshari_Generalized_Zero-Shot_Learning_via_Over-Complete_Distribution_CVPR_2020_paper.html）使用CVAE生成seen/unseen分布；本Idea是非生成式文本邻域经验贝叶斯。
  - Semantics-Free Inter-Class Feature Generation（CVPR 2025，https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Generalized_Zero-Shot_Classification_via_Semantics-Free_Inter-Class_Feature_Generation_CVPR_2025_paper.html）以联合条件Gaussian和相邻类混合生成特征；IDEA-166不生成特征，邻域来自冻结角色文本并直接预测协方差。
optimization_round_1: 删除flow/diffusion、全协方差、文本MLP、伪样本生成和新分类器；只保留rank-16共享基、类条件对角方差与解析likelihood。
optimization_round_2: 训练/评估只做一次残差投影；主条件shrinkage=0.5 Gaussian，补救1 shrinkage=0.8，补救2在同一方差上换Student-t(df=5)。三个条件不重复拟合PCA或读取图像。
slimmed_execution: 一个脚本、一个配置、一次100/50 split、一次PCA/方差统计、三个解析评分条件；无新资产、无神经网络训练、无checkpoint、无正式test选择。
minimal_falsification: 使用xlsa17标准开发划分100个dev-seen和50个dev-unseen类。Gate A严格逐类LOO，用其余99类重新拟合basis、全局方差、`v_perp`、floor和tau，Conditional NLL相对Global的类别bootstrap 95% CI上界必须<0。Gate B固定50类macro Top-1，要求Conditional相对Point≥1.0pp、Shared≥0.5pp、同体积ClassScalar≥0.5pp，后两项类别bootstrap下界>0，净纠正≥20并超过32个方差错配的95%分位。三条件全失败即reject，不调rank、K、tau、df或融合。
paper_level_claim: Gate与正式GZSL均成立后，才可声称“以文本邻域经验贝叶斯视觉分布替代冻结CLIP点原型，在不生成unseen样本的前提下建模类特异不确定性”；正式目标仍为相同checkpoint Full/Off双门并推动H≥80，不能声称首次分布式ZSL。
old_solution_path: `image×point prototype→统一余弦排序`。
new_solution_path: `seen残差→共享低秩基/类方差→文本邻域迁移unseen方差→解析分布似然`。
principle_difference: 旧方法只比较中心位置；新方法比较中心与类内形状共同定义的生成可能性。
non_equivalence_test: Conditional必须显著超过共享方差和保持相同rank-16 log-volume但删除方向形状的ClassScalar控制，并超过错配方差；否则只能解释为统一Mahalanobis、类特定温度或体积校准。
minimal_viability: 100类严格LOO证明文本能预测视觉类内形状；50类Point/Shared/ClassScalar/Conditional/Shuffled同源比较、宏准确率、类别bootstrap、净纠正与正定合同全部通过。
current_advantage: none；严格LOO支持文本可预测部分视觉方差，但未形成accuracy、speed_or_cost或generality父条件优势。
performance_status: below_parent / rejected_at_gate；Gate未训练TG+GTD且未产生H/U/S/ZS。
failure_boundary: 单位CLIP特征上的外在Gaussian/Student-t近似；Mean8中心Gate不证明V4 TG+GTD增益。lambda0.5、lambda0.8、Student-t(df5)全部失败后，不继续调rank、K、tau、df、协方差结构或融合权重。
owner_decision: 2026-08-29 owner授权失败Idea最多2次补救，仍失败自动换方向，无需逐次批准，并以H≥80为目标。

## 2026-08-29 100/50解析Gate结果

- RUN commit：`fd978d635debd025bf8f6e8b589a02b594a0df02`；父commit：`52088f69d7ac4e574e7b63c28b21ac0da7789933`；config SHA：`029d9d4a6863fdc08496d99f101018651d6498c8d5b0fb0a2cf5b3a89ebb8f62`。
- 数据：V4 text-v2资产`69c9c6d82a755fe8`；100个dev-seen类共4,702张图拟合，50个validation-unseen类共2,355张图只评估；未加载official test或真正unseen图像。
- 机制Gate A三次均通过：Conditional相对Global的严格LOO类平均NLL差分别为`-0.059939`、`-0.035220`、`-0.045969`，95% CI上界均<0。文本邻域能预测一部分源类视觉方差。
- Point Mean8宏准确率=`82.2852%`。Conditional三次分别=`79.6276% / 79.6317% / 79.6203%`，相对Point=`-2.6577 / -2.6535 / -2.6650pp`。
- Conditional相对Shared仅`+0.0943 / +0.0985 / +0.0870pp`，相对同体积ClassScalar为`-0.0042 / +0.0985 / +0.0879pp`；类别bootstrap均未通过，且三次都低于32个Shuffled的95%分位。
- 三次纠正/损坏分别为`87/144`、`86/143`、`86/143`，净纠正均为`-57`。方差正定、数值有限，但分类路径稳定损坏点原型。
- 决策：`gate_fail_reject / below_parent`。信号存在但不能转化为类别竞争优势；不进入TG+GTD正式联合训练，不调rank、K、tau、df、收缩、融合或协方差结构。
- 输出：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-166-text-conditioned-distribution-seed7/result.json@sha256:f4b455bc6e41d0f4c3ea7ade5c74d3fe6ed5669da8c47420b6958247eaa1bf6a`。
- 审查：专项`8 passed`；两名Agent完成一轮独立清单、一次交叉交换和真实RTX4090 micro，最终双方均为`P0=0/P1=0，双Agent交叉审查通过`。

adversarial_admission: 2026-08-29两名独立Agent完成方案提出/证伪及交叉交换，集中修正奇异协方差、视觉残差中心、ClassScalar、严格LOO、V4 text-v2资产、自由度、split SHA与CUDA路径；收敛方案P0=0/P1=0。
