# IDEA-160：Full-Resolution Concept Grounding

idea_id: IDEA-160
source_type: first_principles + local_observation + experiment_result
status: rejected
problem_category: visual_grounding
mechanism_tags: [text_only_concepts, full_resolution_patches, concept_grounding, prequeue_oracle]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
problem: 先前在36个粗patch上得到的概念检测中位AUC仅0.568807，不能区分“冻结CLIP最终层没有细粒度信号”和“24×24 patch被池化成6×6时丢失了信号”，因此不足以决定是否继续视觉证据范式。
hypothesis: 若冻结CLIP ViT-L/14@336最终层的576个原始patch保留了可复用角色概念，保持概念、文本、seen图像、最大patch相似度和AUC口径不变时，576-patch应使31个迁移概念达到中位AUC至少0.60，且至少60%的概念AUC达到0.60。
core_change: 不训练模型，只把已审计概念oracle的视觉输入从每图[36,768]粗池化patch替换为同源、已审计的每图[576,768]最终层patch；其他条件保持不变。
old_signal_or_primitive: V4使用全局CLS与点原型；先前概念oracle只读取6×6粗池化patch。
new_signal_or_primitive: 由文本唯一的跨类角色概念组成类别，并在冻结CLIP的24×24完整局部token上进行概念级视觉落地。
paradigm_shift: 若成立，类别从单点文本原型改为可复用角色概念的组合，图像按概念证据而非单一全局相似度解释。
why_not_module: 该oracle不增加Gate、Head、重排、校准或训练loss，只检验新表示原语赖以成立的视觉信息是否存在。
closest_paradigm_work: 本候选在pre-queue最小证伪阶段已失败，尚未形成新颖性claim，因此没有启动近期论文系统检索；不得据此声称首次文本唯一概念组合。
minimal_falsification: 在CUB 7,057张seen训练图像上，用text-v2前六角色去类别前缀后的短语构造同角色mutual-5NN概念簇；阈值0.85，要求每簇至少含3个seen类和1个unseen类；固定31簇，以簇内短语CLIP均值为概念向量，图像分数取576个patch的最大余弦，并以seen类别成员关系计算AUC。真正unseen图像不参与计算。
paper_level_claim: 当前无。若最小证伪通过，后续可检验“无人工属性标注的文本概念组合能否在冻结CLIP局部token上形成可迁移视觉原语”；本次结果不支持该claim。
evidence_refs:
  - experiments/v3/PATCH_ASSET_REBUILD_AUDIT.md
  - /data/lby/projects/cv_project/GZSL_Warehouse/assets/rgve/CUB_openai_vitl14_336_projected_patch_final_v1/asset_manifest.json@sha256:d096087c9bd37d90157688e21e79b8ba6a61f0ea9b1fa91f4f544f8bc1dd1ad0
  - /data/lby/projects/cv_project/GZSL_Warehouse/assets/texts/CUB/text-v2-bd935b8a4ed42d59/role_texts.json@sha256:bd935b8a4ed42d59c3a39c3f30bb99552c717ef18dadbf3349422b1cef728985
success_condition: 31个固定概念簇的中位AUC不低于0.60，且至少60%的概念AUC不低于0.60；中位AUC不低于0.70只作为强信号观察，不替代主门槛。
failure_condition: 中位AUC低于0.60，或AUC不低于0.60的概念比例低于60%，则当前“最终层576-patch直接概念落地”假设立即拒绝，不进入Innovation、TRY或训练。

## 2026-08-29 最小证伪结果

- 数据边界：只使用7,057张seen训练图像及全部200类文本；真正unseen图像未读取、未参与梯度或筛选。
- 概念构造复现：threshold=`0.80 / 0.85 / 0.90`时迁移簇数=`32 / 31 / 29`；正式诊断固定threshold=0.85、31簇，与36-patch诊断一致。
- 36-patch既有结果：median AUC=`0.568807`，mean AUC=`0.562299`，AUC≥0.60为`8/31=25.81%`。
- 576-patch结果：median AUC=`0.598779`，mean AUC=`0.604205`，AUC≥0.60为`15/31=48.39%`，AUC≥0.70为`5/31=16.13%`，范围=`0.387288–0.884302`。
- 固定seed 7标签打乱对照：median AUC=`0.502243`，mean AUC=`0.499558`，说明非随机结果不是由AUC实现或类别比例泄漏造成。
- 判定：两个主门槛均未通过，状态`rejected`。36→576确实恢复了一部分局部信息，但冻结CLIP最终层完整patch仍不足以稳定支撑文本概念组合或其上的遮挡/反事实训练。
- 边界：该结论只否定“最终层576-patch + 最大短语余弦”这一直接落地方式；现有多层资产只有每层全局向量，不是多层576局部token，因此本结果不能外推为“冻结CLIP所有中间层都没有细粒度信息”。

