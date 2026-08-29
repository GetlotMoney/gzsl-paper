# IDEA-166：Text-Conditioned Visual Distribution

idea_id: IDEA-166
source_type: experiment_result + first_principles + owner_hypothesis + nearest_work_boundary
status: proposed_owner_approved_for_three_attempts
problem_category: class_competition
mechanism_tags: [distributional_prototype, empirical_bayes, low_rank_covariance, text_neighbor_transfer, robust_likelihood]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs: [IDEA-012, IDEA-160, IDEA-165]
problem: V4与Mean8/GTD把每个类别表示为一个点，但同一类别图像存在姿态、背景、视角和遮挡分布；seen类有视觉样本可估计分散度，unseen类只有文本点，联合竞争因此使用不对称置信度。此前共享视觉轨道只会让所有类别一起膨胀，未检验由类别文本邻域迁移的类条件协方差。
hypothesis: 在冻结CLIP全局空间中，先用100个训练类学习共享低秩残差基，再为每个训练类估计收缩协方差，并按Mean8文本邻域把方差参数迁移到50个隔离类，可以把类别原语从点改为文本条件视觉分布。若类条件分布比共享协方差和点余弦更好地描述隔离类图像并提高50类准确率，则该表示有资格进入GZSL联合竞争。
core_change: 所有图像只使用现有全局768维CLIP特征。训练100类图像相对其Mean8文本原型形成残差，PCA得到固定rank-16共享基；每类在该基上估计对角方差并向全局方差收缩。50个隔离类的log-variance由其Mean8文本与100个训练类文本的固定Top-5 softmax邻域加权得到，不学习MLP、不使用隔离类图像。分类使用带log-determinant的Gaussian/Student-t负对数似然，而不是额外logit Head。
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
minimal_falsification: CUB formal-seen按seed7固定100类拟合、50类图像只评估；全部200类文本允许。基线为同50类Mean8余弦。主Gaussian及两次补救均需相对点基线准确率+1.0个百分点、净纠正≥20且损坏少于纠正；类条件Gaussian还必须优于共享全局方差≥0.5个百分点，证明不是统一度量。方差全部有限正值，文本邻域权重和为1，打乱类条件方差不得获得同等增益。三条件全部失败即reject并切换方向。
paper_level_claim: Gate与正式GZSL均成立后，才可声称“以文本邻域经验贝叶斯视觉分布替代冻结CLIP点原型，在不生成unseen样本的前提下建模类特异不确定性”；正式目标仍为相同checkpoint Full/Off双门并推动H≥80，不能声称首次分布式ZSL。
old_solution_path: `image×point prototype→统一余弦排序`。
new_solution_path: `seen残差→共享低秩基/类方差→文本邻域迁移unseen方差→解析分布似然`。
principle_difference: 旧方法只比较中心位置；新方法比较中心与类内形状共同定义的生成可能性。
non_equivalence_test: 类条件分布必须显著超过共享方差控制；否则退化为统一Mahalanobis或温度缩放。
minimal_viability: 100/50隔离、point/shared/conditional三方同源比较、+1pp/净纠正门、打乱方差反例和数值正定合同全部通过。
current_advantage: 不依赖已经失败的patch读取、三态、图搜索或类特定Reader；复用V4已有全局特征和Mean8文本。
performance_status: pre-run；V4父H=78.119641，最终目标H≥80仅在Gate通过后验证。
failure_boundary: shrinkage0.5、shrinkage0.8、Student-t(df5)全部失败后，不继续调rank、邻居K、df、协方差结构或融合权重。
owner_decision: 2026-08-29 owner授权失败Idea最多2次补救，仍失败自动换方向，无需逐次批准，并以H≥80为目标。
