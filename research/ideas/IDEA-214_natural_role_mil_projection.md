# IDEA-214 / Natural-Role MIL Projection (NRMP)

- status: `rejected_at_prequeue_gate`
- problem_category: `visual_grounding`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- source: IDEA-213 raw role-patch matching failure

## Hypothesis

NRMP用seen类bag标签和MIL学习rank-16残差投影，希望把无效的冻结CLIP patch-role cosine变成可迁移的自然角色局部证据。六个固定条件分别为NRMP、no-diversity、class-name、whole-prototype、wrong-role和same-class lexical-scaffold；全部训练2,000步后才加载official test。

- old_solution_path: frozen bad affinity -> matching/head。
- new_solution_path: seen bag-label MIL -> learned local role evidence -> candidate verification。
- principle_difference: 先学习局部证据表示，再决定如何匹配。
- non_equivalence_test: NRMP必须胜过同容量class-name、whole-prototype、wrong-role、lexical-scaffold和no-diversity控制。
- minimal_viability: unseen AUC>=0.60、H相对mean8 semantic基线+1、seen/unseen净纠正、locality优于控制。
- current_advantage: none；`performance_status=proof_of_path`。
- failure_boundary: bag标签可能只教会类级语义，diversity可能制造伪locality，粗patch可能缺少部位信息。

方法借鉴APN（NeurIPS2020）的class-level弱监督locality、RegionCLIP（CVPR2022）的region-text alignment、Attention MIL（ICML2018）和LAPS（CVPR2024）；不使用专家属性、伪region标签、教师或蒸馏。

## 双Agent审查

- Idea初审A/B：`5721d976ae3d8d7400eecfb995df59c178112b85463978db760d7d47e6597e06` / `e507b30b5d35122c0cd8e2bba6dcaf1e298e984c3c5037cec7c154700e6ac957`。
- Idea交叉及最终A/B：`ff4247819eab08a437d0a472b8e6cad38a298b4a3a8880772d9da86f3e5e6a83`、`65cd3ef58e91aa8b439cd34a4337dfa1059cd862da0b6bb99f0d09be417316b8`、`b6b19fea9ce8ffcfc7c89b1f54de1801205d3e1f2c3cad6cb4449ea7df849c41`、`4d74ed7ef2a535510fe64b3a968acc8b1d01e0b380f410b7675a3fa21d53c80b`。
- 最终脚本SHA：`250a50d43ecf891dafb74e06a3aab4d4acfa163f561878790c210b330267c06e`；双Agent代码复核P0/P1=0。

## Gate结果

结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/nrmp/IDEA-214-GATE0/result.json@sha256:e4a06160aaf0a332bb806b9f24fad5d5e1a531540076b5665edbc7a734bdd54c`。

- mean8 semantic基线：`H=68.750566`。
- NRMP：seen/unseen AUC=`0.737972/0.777722`，但`U/S/H=56.514227/52.083117/54.208271`；seen纠正/破坏=`168/433`，unseen=`273/651`。
- whole-prototype控制：unseen AUC=`0.777952`、`H=54.252420`，与NRMP持平且略高；class-name与lexical控制同样达到AUC约0.74-0.75。
- no-diversity：unseen AUC=`0.777892`、`H=54.212759`，与Full NRMP完全同级。
- locality、H净增、净纠正、控制优势等硬门均失败，`gate_pass=false`。

## Decision and learned evidence

NRMP被否定，不进入Innovation或完整训练。学习投影确实把pair排序AUC提高到0.78，但角色描述不优于重复whole-prototype，说明提升来自类级语义adapter而不是自然角色locality；零阈值又导致大量c1误翻。下一救援必须先从role和patch表示中扣除whole-class/global成分，再验证条件局部信息，不能继续调MIL温度、diversity或阈值。
