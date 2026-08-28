# IDEA-159：Refutation-Gated Transport

idea_id: IDEA-159
source_type: experiment_result + first_principles + nearest_work_boundary
status: proposed
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
problem: GTD使用文本几何和seen teacher为每个true-unseen类别确定固定迁移角θ，但没有实例级视觉否决机制；直接使用patch重排logits的CLPR系列为负，GAVE方向验证仅带来+0.027340H与净纠正+1，说明局部视觉不适合取代全局CLS做正向分类。
hypothesis: patch负向证据虽然不足以直接分类，却可能识别“当前实例不支持某个候选的GTD迁移”；仅用反驳比例衰减该候选已有θ、且绝不生成新方向或直接修改非候选logit，可减少错误unseen迁移造成的竞争而保留GTD已验证的正向收益。
core_change: 对V4 Top-5候选，用前六local与unique定位patch，分别汇总patch对GTD切向的正支持和负反驳；只有负证据强于正证据时生成refutation ratio。实例角度θ(x)=θ×(1-α×ratio)，α∈[0,1]；seen类θ=0、overall角色、非Top-5类别和GTD方向均不可改变。α=0逐值复现V4。
nearest_work_boundary:
  - VADS与PSVMA使用视觉信息生成或更新实例语义原型；RGT不生成原型、不学习视觉到语义映射，只在原GTD方向上单调减小已审迁移角。
  - ProtoMM在测试流中动态组合多模态原型；RGT不使用测试流记忆、自训练或跨样本更新。
  - Counterfactual ZSL生成样本级反事实并判断seen/unseen一致性；RGT只使用候选局部反驳控制GTD角度，不生成图像或视觉特征。
evidence_refs:
  - experiments/v4/confirmation/INDEX.md
  - experiments/v4/EXPERIMENT_QUEUE.csv@exp/v4/innovation/innovation-001-gave
  - https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Visual-Augmented_Dynamic_Semantic_Prototype_for_Generative_Zero-Shot_Learning_CVPR_2024_paper.html
  - https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Progressive_Semantic-Visual_Mutual_Adaption_for_Generalized_Zero-Shot_Learning_CVPR_2023_paper.html
  - https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Dynamic_Multimodal_Prototype_Learning_in_Vision-Language_Models_ICCV_2025_paper.html
  - https://openaccess.thecvf.com/content/CVPR2021/html/Yue_Counterfactual_Zero-Shot_and_Open-Set_Visual_Recognition_CVPR_2021_paper.html
success_condition: 参数无训练CUB诊断的理论best相对V4父H至少+0.5、seen+unseen净纠正至少20、ZS不下降；三项全部满足才允许进入训练候选。
failure_condition: H潜力低于+0.5，或净纠正低于20，或ZS下降，则训练前drop并关闭当前反驳衰减公式。
