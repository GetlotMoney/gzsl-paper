# IDEA-213: Class-Held-Out V/I Training

idea_id: IDEA-213
status: testing_non_innovation_tune_rescue
problem_category: learning_generalization
problem: TUNE013把head分类CE限制到seen类后，仍可能让V/I在普通seen批次上学习到只适合当前seen类别的视觉读出和关系强度，无法模拟official unseen没有图像梯度的迁移边界。
mechanism_tags: [one_text, class_held_out, first_order_meta, reader, interaction, tune]

hypothesis: 保持TG/GTD/S的普通seen训练和200类冻结推理不变，只把V/I的分类学习信号改成三折类别留出episode：每步用两折seen类做临时inner状态，再用剩余一折pseudo-unseen图像的outer CE更新正式Reader/alpha，可以让V/I更接近“从seen学习、迁移到未见类别”的训练合同。

old_solution_path: TUNE013中Reader和alpha会从同一普通seen batch的seen-only分类CE得到正式梯度；这些梯度直接奖励当前seen类别分类，而不是类别留出后的迁移表现。
new_solution_path: TUNE014每个episode按rank-modulo固定一折为pseudo-unseen，另外两折只在临时head副本上做一步inner更新；outer用临时状态处理pseudo-unseen图像，但分类候选轴是全部150个train seen类，从而同时包含pseudo-unseen正类与inner/meta-seen负类，采用一阶近似把outer梯度映射回正式Reader/alpha。S的raw_role_weights仍从普通seen CE学习，TG/GTD沿用TUNE013 parent loss，推理仍是一文本S/V/I的200类冻结前向。
comparison_scope: 相对V7-TUNE-013，仅改变V/I分类梯度来源；不引入真实unseen图像梯度，不改变正式评估或best-H选择口径。
principle_difference: V/I的学习对象从“拟合同批seen分类”改为“经过class-disjoint inner状态后，在held-out seen类上表现好”，用seen内部类别留出近似official unseen迁移。
non_equivalence_test: inner batch类别与pseudo-unseen outer类别必须不相交；outer CE候选轴必须是全部train seen类并包含inner/meta-seen列，这些列必须获得非零直接梯度；临时inner step不得改变正式参数；outer loss必须给正式Reader/alpha提供梯度；S仍由普通seen CE提供梯度；official test张量不得进入任何梯度路径。
minimal_viability: CPU单元micro路径能证明临时参数不正式step、outer loss给Reader/alpha梯度、pseudo-unseen不在inner、best_update=0不能过gate；正式CUB run另行由owner授权后执行。
minimal_falsification: 若完整CUB seed7 RUN后Full H仍低于TG+GTD+S retrained H 79.768159，或V/I关闭诊断不显示同checkpoint贡献，则该类别留出V/I训练不成立。
current_advantage: not_yet_tested
performance_status: proof_of_path
failure_boundary: 该实现是一阶近似，不是完整二阶双层优化；如果需要证明二阶meta梯度本身有效，必须新建实验并重新审查计算图和显存预算。
paper_level_claim: none
