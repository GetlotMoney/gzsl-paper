# IDEA-214：一文本关系视觉化算子

- 状态：`rejected_drop_tune015_contract_failed`
- problem_category：`learning_generalization`
- mechanism_tags：`[text_relation_operator, visual_centroid_alignment, graph_compile]`
- source_type：`code_analysis + first_principles + owner_hypothesis`
- 准确代码父条件：`35cefc52896c383e1ec75a3adc5f78d218d616a3`
- 问题：TUNE013的一文本关系仍依赖逐图Reader读取视觉方向，可能把“关系是否可由文本边迁移到视觉边”与“Reader是否学会读图”混在一起。
- old_solution_path：八角色文本构造边方向，固定编译关系图；逐图Reader从每张图像读出关系证据，再与`[norm(x), reader(x)] Q^T + b`共同分类。
- new_solution_path：八角色文本构造边方向`D_text`；共享identity-residual低秩算子形成`D_vis = normalize(D_text + A(D_text))`；只用trainval seen视觉中心差分监督seen-seen边；同一`A`外推到全部边后ridge编译为类别关系原型；推理为`norm(x) Q^T + b`。
- principle_difference：学习对象从“每张图像的关系读出”变成“文本关系方向到视觉类别差分的共享算子”，推理时不再有图像条件Reader。
- non_equivalence_test：导出checkpoint只能包含`q,bias`，不能包含Reader权重、edge_index、ridge_map或在线图求解；unseen相关边必须由同一个identity-residual A从文本方向生成，而不是使用unseen图像或逐图Reader。
- minimal_viability：初始`operator_up=0`时逐边`D_vis`精确等于原一文本`D_text`且范数为1；本地micro-batch中`operator_up/operator_down`梯度有限、不爆炸；seen-seen视觉对齐loss有限；导出部署logits与训练head full logits数值等价。
- current_advantage：CUB完整RUN后为负：Full H=79.437516，低formal V7 1.072915 H、低TUNE013 0.508281 H；删除Reader的共享算子无法替代逐图实例读出，seen视觉中心差分监督不足以迁移到unseen竞争。
- performance_status：`below_parent`
- old_signal_or_primitive：一文本方向加逐图Reader视觉读出。
- new_signal_or_primitive：trainval seen类别视觉中心差分作为边级视觉方向监督；共享identity-residual低秩线性算子作为关系迁移原语。
- paradigm_shift：基本学习对象从图像实例读出器变为类别关系方向的跨模态算子。
- why_not_module：它不是增加Gate、Head或校准项；它删除Reader并改变关系残差的生成路径，但在正式结果前只作为TUNE候选，不登记为Innovation。
- closest_paradigm_work：GTD-TST（IDEA-146）只学习单类Mean8到Value测地移动比例；本Idea学习类别对文本方向到视觉中心差分的共享算子，输出为图编译关系残差。
- minimal_falsification：若初始`D_vis`不能复现`D_text`、A在本地micro-batch梯度非有限或爆炸、unseen边不能由共享A产生、或导出不能化为`q,bias`单矩阵推理，则立即drop。
- problem_family：`unknown`
- shared_bottleneck：细粒度类别关系能否从seen视觉几何迁移到unseen竞争。
- reusable_capability：待验证。
- coverage_and_transfer：仅CUB预注册；跨数据集未知。
- frontier_shift：若成立，可能把Reader依赖的运行时成本转移到训练期关系编译。
- failure_boundary：seen视觉中心差分可能只学习到seen局部几何；identity-residual低秩A可能欠拟合；删除Reader会失去实例级视觉证据，Full H可能低于V7正式框架。
- paper_level_claim：若真实RUN成立，只能窄称“一文本关系可通过共享视觉化算子预编译为GZSL类别关系原型”，不能称首次。

## GTD重复性判断

不阻断。GTD的核心公式是单类`Mean8 -> Value`测地迁移Gate，训练目标是pseudo-unseen的角度比例蒸馏；TUNE015的核心公式是类别对`D_vis = normalize(D_text + A(D_text))`的共享identity-residual低秩算子，训练目标是seen-seen视觉中心差分对齐，并把结果图编译为关系残差。两者都使用seen视觉信息，但学习对象、监督形态和推理出口不同。

## 结果（TUNE015）

seed7完整28228 updates、201次official评估；best update=18753，U/S/H/ZS=78.273934/80.636215/79.437516/87.433225。相对formal V7 -1.072915 H、相对TUNE013 -0.508281 H。同checkpoint关闭差：i_off +0.107266 H（H=79.330250）、s_off +0.864793 H（H=78.572723）。程序decision=drop_tune015_contract_failed，停止此方向，不在该失败代码上继续堆叠。
