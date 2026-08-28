# IDEA-158：Geodesic-Aligned Visual Evidence

idea_id: IDEA-158
source_type: first_principles + code_analysis + nearest_work_boundary
status: revised_weak_signal_only_not_promoted
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
problem: V4的TG与GTD只依据全局CLS、角色文本和seen视觉中心学习类别原型；真正测试图像的局部patch没有验证某个候选类别的Mean8→Value语义移动方向是否在该实例中可见。直接patch-text Top-K、局部视图路由和Top-2角色差比较均不能回答“当前视觉证据是否支持V4的语义迁移方向”。
hypothesis: 对V4 Top-5候选，以前六个局部角色和unique只负责定位patch，再测量这些patch与候选Mean8→Value球面切向的一致性；真实类别应比困难竞争类获得更强的多角色方向覆盖，因此该证据可在不使用overall patch query、不移动原型且不引入类别专属参数的前提下纠正细粒度混淆。
core_change: 新增GAVE候选残差。角色文本选择局部位置，视觉分数来自patch对V4测地切向的方向支持；取七个local+unique角色中最强三个的均值，并在Top-5候选内中心化为零和残差。overall角色完全不参与patch查询，留给后续整体条件创新；强度为零时逐值复现V4。
nearest_work_boundary:
  - TransZero使用属性引导Transformer进行属性定位，但不验证已学习类别原型的测地迁移方向。
  - FILIP使用视觉/文本token最大相似度做细粒度late interaction，但不使用候选原型的球面切向。
  - PLOT与SOT-GLP使用OT分配视觉局部与多prompt；GAVE不做prompt-patch OT，避免与“防重叠分配”创新重合。
  - 本仓库PCPC比较Top-2候选的角色文本差；GAVE独立评估每个Top-5候选的Mean8→Value方向，并用角色只定位、不直接作为类别证据。
evidence_refs:
  - experiments/v4/framework_diagram.html
  - experiments/v3/PATCH_ASSET_REBUILD_AUDIT.md
  - research/ideas/IDEA-157_pcpc.md@codex/v3-dual-visual-candidates
  - https://arxiv.org/abs/2112.01683
  - https://arxiv.org/abs/2111.07783
  - https://arxiv.org/abs/2210.01253
  - https://arxiv.org/abs/2603.08347
success_condition: 首轮真实CUB错误诊断中，至少一个非零残差强度使H高于V4父结果，wrong-to-right多于right-to-wrong，且纠正样本的正证据至少覆盖三个local+unique角色；随后训练候选才允许进入双1H正式门槛。
failure_condition: 所有非零强度均不提高H，或最佳条件的right-to-wrong不少于wrong-to-right，或证据退化为单角色/全局背景响应，则训练前drop并停止该方向。
diagnostic_result: V4-TRY-001 @ 9d89b4789876fafdc08a659e5cffe4d083db8d0b；α=0.5，U/S/H/ZS=79.591095/76.754338/78.146981/85.794950，较父H仅+0.027340；seen纠正2破坏0、unseen纠正0破坏1，净纠正+1；result SHA=addeff949e4313999c98a816faedddc5443903e29d232580f157b8ec5024b171。
