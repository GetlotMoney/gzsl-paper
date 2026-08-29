# IDEA-170：Content-Aware Inpainted Interaction

idea_id: IDEA-170
source_type: IDEA-168_result + IDEA-169_result + owner_rescue + first_principles
status: proposed_owner_approved_for_rescue2
problem_category: visual_grounding
mechanism_tags: [input_intervention, deterministic_inpainting, perturbation_realism, non_additive_interaction]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs: [IDEA-162, IDEA-168, IDEA-169]
problem: IDEA-168的裸目标概念交互在mean-fill与local-blur下稳定但不具概念特异性；IDEA-169对比margin因合法同角色近邻覆盖不足且三项效应门失败。最后一个未排除的根因是固定方块均值/模糊扰动本身产生通用视觉损伤，掩盖目标区域的概念作用。
hypothesis: 保持IDEA-168的固定目标Attention、裸目标概念logit、500图、区域与对照完全不变，仅把人工方块扰动替换为从窗口周围像素推断内容的确定性补全后，目标区域对的`abs(eta)`若稳定超过困难和随机对照，则说明此前失败主要来自扰动分布外伪影。
core_change: 第一种干预使用固定64步Jacobi调和补全，让窗口内部逐步满足邻域平均并由四周真实像素提供边界；第二种使用最近边界反射补全，把窗口外相邻纹理沿最近边界镜像到内部。两者都只读取当前图片，不使用生成模型、人工部位、框或额外图像。主分数返回IDEA-168的固定原图Attention裸目标logit；动态Attention只旁报。
old_signal_or_primitive: mean-fill用全局均值色制造平块，local-blur保留被删区域的低频内容，二者都可能让CLIP产生通用异常响应。
new_signal_or_primitive: 由当前图像窗口边界内容决定的确定性补全干预，使被测量的区域作用更少依赖固定人工遮挡图案。
paradigm_shift: 继承输入干预范式，但这是最后一次干预真实性补救；不把补全算法本身包装为核心创新。
why_not_module: 不训练补全网络、不增加分类模块、不进入TG+GTD，只替换Gate 0的输入干预生成方式。
closest_paradigm_work:
  - Explanations for Occluded Images（ICCV 2021）说明遮挡解释对输入干预设计敏感；本补救不声称真实因果删除。
  - 生成式补全解释已有先例；本卡只使用无外部权重的确定性边界补全，不能主张首次inpainting解释。
minimal_falsification: 完全复用IDEA-168的seed7、100/50隔离、同500图、每图最多3概念、136对形成逻辑、Top-2窗口、hard/random对照、固定目标Attention裸logit、最少100对/25类、两层bootstrap与四项效应门。唯一变化是两种扰动改为`harmonic_inpaint_64`和`boundary_reflect_inpaint`。四个magnitude-excess均要求95% CI下界>0、标准化效应≥0.2，跨扰动符号一致性CI下界>0。任一失败则补救2失败，彻底停止跨区域干预方向。
paper_level_claim: 只有本Gate和后续任务优势成立后，才能窄化声称“内容条件干预揭示共享文本概念的跨区域非加性”；不得声称首次补全解释或真实因果定位。
old_solution_path: `人工均值/模糊方块→完整CLIP→固定Attention eta`。
new_solution_path: `当前图像边界内容补全窗口→完整CLIP→固定Attention eta`。
principle_difference: 被删除区域的替代内容由当前图像上下文决定，而不是固定均值或仍含原内容的模糊。
non_equivalence_test: 同一目标、困难和随机窗口必须使用相同补全算法；若目标区域的交互优势在两种内容补全下均不超过对照，则扰动真实性不能救回该方向。
minimal_viability: 同500图至少100对/25类、四效应门、跨补全稳定门和patch同源全部通过；只记proof_of_path。
current_advantage: none；IDEA-168四项效应门全失败，IDEA-169覆盖门与三项效应门失败。
performance_status: rescue2_pending；本Gate不报告H/U/S/ZS。
problem_family: 人工遮挡伪影掩盖细粒度局部或跨区域概念证据。
shared_bottleneck: 干预图像偏离冻结CLIP的自然输入分布。
reusable_capability: 若成立，可提供无需外部补全模型的上下文条件干预。
coverage_and_transfer: 仅CUB seed7 pseudo class-disjoint。
frontier_shift: unknown。
downstream_effects: 通过后才允许任务级分类验证；失败后转离视觉干预方向。
failure_boundary: 调和与反射补全仍不等于真实生成式反事实；不允许调迭代数、窗口、阈值或统计门。失败后不再进行第三次补救。
owner_decision: 2026-08-29 owner批准两次自动补救；IDEA-169失败后自动进入本次最终补救。
