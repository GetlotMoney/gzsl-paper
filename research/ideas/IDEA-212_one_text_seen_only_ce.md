# IDEA-212: One-Text Seen-Only CE

idea_id: IDEA-212
status: testing_non_innovation_tune_rescue
problem_category: learning_generalization
problem: TUNE012 one-text Full 在 CUB 低于 TG+GTD+S，可能是新增head的200类训练CE把50个true-unseen类在每个seen训练batch里都当作负类压低，破坏最终200类联合竞争。
mechanism_tags: [one_text, seen_only_ce, tg_gtd, semantic, reader, interaction]

hypothesis: 保持TUNE012的一文本S/V/I结构、方向CE、训练预算、seed7原始初始化和200类推理不变，仅把最终head分类CE改成只在150个trainval seen logits上计算，可以减少对true-unseen原型的训练期负梯度，使CUB Full H回升到formal V7附近并保留关系负控优势。

old_solution_path: TUNE012用同一批seen训练图像计算200类head logits，然后对200类全轴做CE；true label之外的全部类，包括50个official unseen类，都会收到负类竞争梯度。
new_solution_path: TUNE013仍输出200类logits，但训练分类CE只取`logits[:, seenclasses]`并用`global_to_seen[y]`监督；方向CE继续训练Reader，推理和U/S/H/ZS仍使用完整200类logits。
comparison_scope: 本TRY相对V7-TUNE-012一文本Full路径只改head classification CE范围；正式V7 `b32a16f` 只作为CUB性能地板和框架身份锚点。
direction_ce_boundary: CUB一文本图预注册跳过seen类`[13,76]`，运行时若不一致直接失败。
principle_difference: 训练分类学习对象从“seen图像对全部200类排序”改为“seen图像只对可监督的150个seen类排序”，避免对无图像监督的unseen类施加分类负梯度。
non_equivalence_test: 在同一batch的head CE反向中，excluded unseen logits的梯度必须严格为0；评估时Full logits形状仍为200类，且ZS仍只在50个unseen类中竞争。
minimal_viability: CUB完整RUN可执行、best_update>0、unseen_images_used_for_gradient=false、Full H高于TUNE012 Full 77.405741，并报告相对TG+GTD+S retrained H 79.768159与formal V7 H 80.510432。
minimal_falsification: CUB seed7完整28228 updates后Full H仍低于TG+GTD+S retrained H 79.768159，或signflip/role_shuffle不低于Full。
current_advantage: not_yet_tested
performance_status: proof_of_path
failure_boundary: 如果仍下降，则问题不主要是all-class CE压制unseen，而更可能是一文本关系方向或Reader交互本身无法提供稳定迁移；不得继续把本改动包装为创新。
paper_level_claim: none
