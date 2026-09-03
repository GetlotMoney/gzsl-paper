# IDEA-215: De-Parent S/V/I Mask Diagnostic

idea_id: IDEA-215
status: proposed_diagnostic
problem_category: learning_generalization
problem: ABL002 显示完整框架下旧全局 V/I 贡献仅约 +0.25 H 且 I 与 V+I-off 逐值相同，无法区分“TG+GTD 本身很强、把收益空间占满（遮蔽）”与“V/I 确实没有独立可学内容 / CLS+一文本语义路径已到天花板”两种假说。
mechanism_tags: [de_parent, mean8, tg_gtd_mask, class_held_out, diagnostic]

hypothesis: 若把 TG+GTD 冻结并让 head 基于 Mean8 纯文本原型训练，S/V/I 在弱基线上的从头重训贡献会显著放大（说明其能力被 TG+GTD 遮蔽）；若仍几乎无贡献，则 CLS 全局特征 + 一文本语义路径已接近信息上限。因 Reader 的唯一 logits 通道是关系分支，Full−I-off 与 Full−V+I-off 逐值等价，本实验对 V/I 的判据按"V+I 联合"口径解释。

old_solution_path: TUNE013/TUNE014 在 TG+GTD 迁移后的原型上训练 S/V/I，TG+GTD 已把大部分可判别信息编入 base/role/compiled 矩阵。
new_solution_path: 冻结 TG+GTD，head 改为 Mean8 纯文本 base，S 用 seen-only CE、V/I 用一阶 class-held-out outer CE，五组条件从头重训，另加 B0 零训练 Mean8 基线。

principle_difference: 把“head 模块相对完整框架的边际贡献”改为“head 模块在去掉父迁移后的独立可学内容”，用于决策是否重构收益空间分配或转向新信号。

non_equivalence_test: source 必须全冻结（无 parent loss、无 sync）；head base 必须精确等于 tg_vpr.base_prototypes()（Mean8）而非迁移后 prototypes()；S/V/I 各条件 freeze/enable 合同与 ABL002 一致；B0 必须等于 Mean8 零训练直接分类；I-off 与 V+I-off 应逐值一致（架构耦合自检）。

minimal_viability: B0 可算出 Mean8 基线 H；五组完整 RUN 可执行、best_update>0、unseen 无梯度、逐条件 best-H 选择。

minimal_falsification: 若 Full−I-off（=Full−V+I-off，联合 V+I，重训差值）仍 ≤1 H 且 B1 Full 仍显著低于 formal V7，则遮蔽假说不成立，CLS+一文本语义路径到顶，转新方向。

current_advantage: not_yet_tested（诊断性，不申报 accuracy/speed/generality 优势）
performance_status: proof_of_path
failure_boundary: 弱基线上的贡献不代表完整框架下的独立增益；不得把本实验的 Full−off 差值包装成正式模块创新；Full−I-off 只按“V+I 联合”解释，不能单独归因 I。
paper_level_claim: none
