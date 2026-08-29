# IDEA-172：Text-Difference Active Evidence Acquisition

idea_id: IDEA-172
source_type: first_principles + rejected_visual_evidence_results + owner_direction + nearest_work_boundary
status: rejected
problem_category: visual_grounding
mechanism_tags: [active_perception, high_resolution_reobservation, text_difference_action, class_disjoint_gate]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs: [IDEA-160, IDEA-162, IDEA-163, IDEA-164, IDEA-165, IDEA-166, IDEA-168, IDEA-169, IDEA-170]
problem: 固定336全图及其缓存patch无法恢复缩小时丢失的细粒度像素；需要检验当前类别歧义能否主动请求一条新的原图高分辨率观测。
hypothesis: leader–challenger角色文本差异可定位一次原图crop，重新编码的高清观测能以同成本超过无关crop和原336 patch，并纠正class-disjoint错误。
core_change: B=1固定主动观察历史：Mean8父排序提出leader/challenger，角色文本差异选择6×6 patch粗位置，原图crop放大到336重新编码；crop只可交换父Top-1/Top-2，禁止全类重打分、融合、RL、B=2和多尺度。
old_solution_path: 固定336全图特征→点原型排序→argmax。
new_solution_path: 当前歧义→文本差异行动→原图高清新观测→一次离散Top-1/Top-2更新。
principle_difference: 父路径被动消费一次固定输入；候选路径由类别歧义控制下一条输入信号。
old_signal_or_primitive: 单张336全图及同一缩小图派生patch。
new_signal_or_primitive: 文本差异动态请求、由原始高分辨率图重新编码的一次局部CLIP观测。
paradigm_shift: 静态单观测判别变为一次主动视觉获取。
why_not_module: crop不融合父logit也不全类重排；若同成本固定/无关crop可复现，主动路径即失败。
closest_paradigm_work:
  - AdaptVision（CVPR 2026）已做主动视觉获取和crop工具。
  - CropVLM（CVPRW 2026）已无框动态zoom。
minimal_falsification: 100/50开发划分先检验25个固定高清crop的oracle上限，再以B=1 Active击败Static、Unrelated、Random、Center和Original-Patch同成本控制；任一失败停止。
non_equivalence_test: Active必须超过所有同成本crop控制及原patch读数；否则只是普通multi-crop。
minimal_viability: Oracle和真实B=1均应带来预注册纠错和macro增益。
current_advantage: none；Oracle存在上限，但真实文本差异行动相对Parent为-0.2pp且净纠正-1。
performance_status: rejected_at_proof_gate；未进入Innovation或正式TG+GTD训练，未报告H/U/S/ZS。
failure_boundary: B=1只能检查父前二；粗patch无法定位、文本差异不可视觉化或同成本控制有效时失败，禁止B=2、多crop、融合或RL补救。
paper_level_claim: 已拒绝；不得声称主动视觉或GZSL性能贡献。
owner_decision: 2026-08-30 owner回复“开始”，授权本卡B=1 proof-of-path诊断；不等同Innovation接纳。
adversarial_admission: 两名独立Agent完成方案交叉、代码交叉与GPU micro；代码路径P0=0/P1=0，但性能proof失败。

## 2026-08-30 结果

- commit=1374835caf91a2ab1279a3f7c1c9c37bd9fe574f；config SHA=e5fa093d809f76accd5661185f248ffc9683f38113d557eaecd5ba3fe4dc186b。
- 100源类4,702图校准；50目标类500图只推理评估。官方测试和formal true-unseen均未加载。
- Oracle：70个父错误且真类Top-2样本全部存在可纠正高清crop，macro从81.0%到95.0%，理论+14.0pp。
- 真实Active：80.8%，低于Parent的81.0%；纠正8、损坏9、净-1，正确父leader损坏率7.14%。
- Center crop为81.6%，高于Active；Active亦未超过Unrelated或Original-Patch，所有类别bootstrap关键CI下界不大于0。
- 结论：原始高分辨率信息存在，但文本差异粗定位无法可靠取得该信息，按预注册停止。
- 输出：/data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-172-active-evidence-acquisition-seed7/result.json@sha256:d3b4dfd8239e5e21990efde7efe8ef01ce19ca668cfbf5946399c58601ecf94b。
