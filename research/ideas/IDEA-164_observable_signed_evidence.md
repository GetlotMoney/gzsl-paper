# IDEA-164：Observable Signed Evidence

idea_id: IDEA-164
source_type: experiment_result + first_principles + owner_hypothesis + nearest_work_boundary
status: proposed_owner_approved_for_gate1
problem_category: visual_grounding
mechanism_tags: [candidate_independent_observability, signed_evidence, fixed_reference_bank, causal_intervention, class_disjoint_transfer]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs:
  - IDEA-162
  - IDEA-163
problem: IDEA-162证明共享概念信号可学习迁移，但IDEA-163把每个类角色强行离散为support/refute/unobserved后，困难反例、Mean8纠错、安全性和删除解释全部失败。根因包括unknown仍由候选比较间接决定、有符号证据没有固定参考、可见性与分类目标不可辨识、attention没有因果锚点。
hypothesis: 只用一个跨类别共享Reader表达两个连续变量即可保留三态逻辑：类别无关角色可观察性`o_r(x)`与固定200类同角色文本参考下的有符号相对证据`d_c,r(x)`。为使分类梯度在结构上不能改变`o`，同一Reader的可观察性路径固定使用无trainable-adapter的冻结CLIP patch/text相似度，只保留每角色一个因果校准scale/bias；类别证据路径使用共享rank-64 adapter。若`d`使用固定参考而非随100/50候选集合变化，则`e=o*tanh(d/2)`应能在class-disjoint类别上区分可见支持、可见反驳和未观察，并给出可删除验证的区域。
core_change: 前六角色文本继续作为命题查询；所有200类同角色文本的固定、数值稳定leave-one-out LogMeanExp构成相对证据参考，`d_c,r=s_c,r-LogMeanExp_j≠c s_j,r`，Gate 1预注册`T_r=1`且不可搜索。每个角色的200类文本均值形成类别无关查询`q_bar_r`；同一Reader的冻结基底路径输出`o_r`并只允许6组因果校准标量更新，类别证据adapter不参与`o`。分类目标仍显式`stop_gradient(o)`；干预目标分别要求原图到证据区域干预的`o`下降和真类有符号`tanh(d/2)`下降均为正、且分别大于随机区域。三态概率由`P(U)=1-o`、`P(S)=o*sigmoid(d)`、`P(R)=o*sigmoid(-d)`唯一确定。
old_signal_or_primitive: IDEA-163使用候选与4个动态近邻离散判定三态，unknown/refute依赖局部候选集合，attention只有事后删除评估。
new_signal_or_primitive: 候选无关的连续可观察性、固定200类参考下的连续有符号证据，以及真实输入干预提供的因果监督；类别由六个连续证据状态而非单点原型或离散阈值状态表示。
paradigm_shift: GZSL从单点兼容度改写为“类别无关地判断角色是否被观察，再以固定语义参考判断观察结果对每个类别命题的正负证据”；未知由观测过程产生，支持/反驳由固定比较基准产生。
why_not_module: prompt、Reader和attention不是claim；核心是两个可辨识连续变量、固定参考证据原语和干预学习信号。若`o`可被分类梯度操纵、`d`随候选集合变化、删除影响不优于随机或最终只剩普通patch残差，则Idea不成立。
closest_paradigm_work:
  - PAPER-003、APN与TransZero均使用人工类级attributes进行局部化或视觉语义交互；本Idea只用现有文本角色描述，并把未观察从负属性中分离。
  - CREST使用attribute定位与Evidential Deep Learning表达不确定性；本Idea不使用人工attributes或独立证据Head，而以候选无关`o`和固定参考`d`生成严格归一的三态概率。
  - Counterfactual ZSL通过生成样本级反事实做seen/unseen一致性判断；本Idea的干预只用于学习/验证角色证据区域，不生成unseen视觉样本。
  - IDEA-163已经证明动态近邻离散三态和事后attention删除不成立，IDEA-164不能复用其运行代码或把失败公式改名。
minimal_falsification: Gate 1只使用CUB formal-seen图像，seed 7固定100类图像与标签用于Reader训练、50类图像与标签完全不进入梯度；与标准ZSL一致，全部200类冻结文本允许作为固定语义参考并参与可导证据标尺，必须明确披露为`all_class_text_reference_used_for_gradient=true`。一个rank-64共享Reader先以固定Top20%类别图像包证据竞争建立`s`；模型级反例必须证明一次纯分类更新后同图`o`逐值不变。从100类中固定250张图、每图轮换一个角色，分别用`q_bar_r` attention选择`o`区域、用真类`q_c,r` attention选择signed-d区域；随机同面积区域只由row+role哈希决定，在real/shuffled完全一致，若与证据区域重叠只会保守降低通过率。训练使用局部模糊，50类中的另250张图使用均值填充只评估。Gate 1要求：(1)候选无关`o_r`与固定200类`d`对候选轴排列逐值不变，在线原图/缓存patch逐图平均余弦均≥0.99；(2)只在`o≥0.5`的角色上，真命题`d>0`、4个困难命题最强值`<0`且真值更大的五选一准确率≥65%，可见角色覆盖≥30%，并且六个角色各自在class-disjoint图像维度上的标准差都≥0.02；(3)在250张class-disjoint图像上，删除各自声称区域造成的`o`下降至少0.01且优于随机的比例≥70%，signed-d下降至少0.01且优于随机的比例也≥70%。打乱对照固定相同rows/roles/random区域轨迹，只把全部1200个文本槽做无固定点错配；其signed-d失败按不经`o`筛选的全角色准确率判断。任一失败立即reject。
paper_level_claim: Gate 1、后续100/50净纠错及正式TG+GTD联合训练均成立后，才能窄化声称“用文本角色的候选无关可观察性与固定参考有符号证据，把GZSL点原型兼容度转化为连续证据状态更新”；`L_final=L_TG+GTD+beta*E`只能称为证据logit更新，在完成概率校准前不得称严格贝叶斯后验或声称首次。
evidence_refs:
  - research/ideas/IDEA-162_learnable_concept_readout_probe.md
  - research/ideas/IDEA-163_tri_state_evidence_predicate_set.md
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-162-learnable-concept-readout-seed7/result.json@sha256:4f73cbbd0308b9e96af1342df2f45bb2f89ed0ffb8ec1bf6001e835101b574af
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-163-tristate-predicate-seed7/result.json@sha256:3373601ac0736936d193136895c29178d58619ad585a7590d38f03c9db4bfc91
success_condition: Gate 1三项必须同时通过且打乱角色文本对照不得通过任一主门槛；通过只允许进入Gate 2，不等于范式、H或论文claim成立。
failure_condition: `o`受候选类别或类别证据adapter影响、`o`标准差<0.02、`d`不满足稳定固定200类参考、可见角色覆盖<30%、class-disjoint signed pairwise<65%、第二种干预下`o`或signed-d任一删除优势<70%或下降幅度<0.01、打乱对照通过任一内容门槛、real/shuffled采样/环境不匹配或patch同源平均余弦<0.99，均立即置为rejected并停止当前公式。
owner_decision: 2026-08-29 owner回复“行，开始尝试”，批准IDEA-164从FRAMEWORK-V4准确父commit独立分叉并只执行Gate 1。
