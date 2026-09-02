# IDEA-213 / Role-Region Transport Verification (RTV)

- status: `rejected_at_prequeue_gate_after_rescue`
- problem_category: `visual_semantic_interaction`
- mechanism_tags: `exclusive_matching`, `role_region_assignment`, `abstention`, `generate_verify`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`

## 假设与方法边界

RTV试图把交互从可互换的标量margin改为显式角色—区域指派：八句语义原型产生候选，8个自然语言角色通过带null的排他Hungarian/Sinkhorn匹配36个patch，候选由结构化匹配能量验证。V另用文本无关MIL池化形成全类视觉logits。该方向不使用教师、蒸馏、专家属性、人工部位答案、未见图像梯度或PCLR在线推理。

- old_solution_path: Top2后由多个head对同一标签输出可互换标量并相加。
- new_solution_path: 语义生成假设，视觉形成bag表示，交互保留角色—patch指派计划后验证候选。
- principle_difference: 中间学习对象是带排他/弃权约束的匹配计划，而不是另一个attention pooled scalar。
- non_equivalence_test: 必须胜过同affinity的row-max、等信息小head和破坏role结构控制，否则只是结构化pooling。
- minimal_viability: 冻结Gate要求unseen条件AUC>=0.55、相对全部控制AUC+0.02、模拟H相对八句语义基线+1、seen/unseen均净纠正、相对控制H+0.5。
- current_advantage: none；`performance_status=proof_of_path`。
- failure_boundary: CLIP粗patch缺少角色可辨识性、全部角色被正余弦迫使匹配、局部文本含非视觉信息。

最接近的原始方法包括DeepEMD（CVPR 2020）的局部最优匹配、Semantic Correspondence as OT（CVPR 2020）的多对一/背景约束、Gumbel-Sinkhorn（ICLR 2018）的可微匹配、RegionCLIP（CVPR 2022）的region-text对齐、APN（NeurIPS 2020）的弱监督局部属性原型、Attention MIL（ICML 2018）和Slot Attention（NeurIPS 2020）。RTV不继承RegionCLIP的伪标签/蒸馏，也不使用APN的专家属性。

## 双Agent pre-queue Gate审查

- v0独立A/B：`6c6cf96cf39308e9f46319c90ee929744d737472411237b2096870b9445ed9fb` / `f15e45406bc6f30576c5b72191f602ba1f67a76ff713cd3548b56047122445e4`。
- v1交叉A/B：`ebc2fe6665260854fc4d08d6bb78838f3b84180c9cf372dab1c8383ce3192f24` / `7a197dfdb00de8fe378ec51d2d70e2554a866caf996a50462a58d12b6c776788`。
- 结论：仅`pass_as_pre_queue_proof_gate_only`，不是Innovation通过；v1最终内容SHA `c8b1650cb2e58d2d6bcb6329101c8c6526e462e4f8384cd7ae115262ea31539e`。

## Gate 0：冻结排他匹配失败

- 脚本SHA：`978e6d9ed7a83a7ef62c880b2ed74a3546294e46ada20790d96834d679be2250`。
- 结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/rtv/IDEA-213-GATE0/result.json@sha256:f1174d0f976a08b9f0cd8346beb8774926727691f43f3700556d3768fb4b1a07`。
- semantic-only：`H=68.750566`；RTV模拟`H=42.285810`，下降`26.464757`。
- RTV条件AUC seen/unseen=`0.467795/0.383265`；row-max=`0.481421/0.402414`。排他匹配比无约束控制更差。
- seen/unseen纠正—破坏=`98-549` / `114-929`。null affinity 0时中位真实匹配数为8/8，说明全部正余弦使弃权失效。
- 仅等信息head呈现排序线索：seen/unseen AUC=`0.645847/0.606018`，但默认阈值大量误翻。

## Rescue 1：seen-only选择性阈值也失败

- 脚本SHA：`6d97cbad73e346d3b31c4ae494be25a0b8cce7e6b65564877036340520c73901`。
- 结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/rtv/IDEA-213-R1-SELECTIVE-GATE/result.json@sha256:6528d773962704024f849fb4d75d95d850cf99e64e0a2690618369a50c9f9d91`。
- 固定seed7的80/20 seen train校准上，任何触发阈值都不能得到正的`纠正-破坏`，最优策略为永不翻转；official H保持语义基线、净收益0。

## 决策

RTV base与选择性阈值救援均在pre-queue阶段被否定，不创建模型分支、不运行28,228步、不登记Innovation。经验是：匹配约束不能从没有局部语义的patch中创造信息；下一方向若继续角色—patch路线，必须首先通过seen-only弱监督学习改善局部表示，再重新验证官方泛化，不能继续调Hungarian/null/阈值。
