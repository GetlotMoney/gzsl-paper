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
hypothesis: 只用一个跨类别共享Reader学习两个连续变量即可保留三态逻辑：类别无关角色可观察性`o_r(x)`与固定200类同角色文本参考下的有符号相对证据`d_c,r(x)`。若`o`只接受离线真实干预监督且在分类损失中stop-gradient，`d`使用固定参考而非随100/50候选集合变化，则`e=o*tanh(d/2)`应能在class-disjoint类别上区分可见支持、可见反驳和未观察，并给出可删除验证的区域。
core_change: 前六角色文本继续作为命题查询；所有200类同角色文本的固定LogMeanExp构成相对证据参考，`d_c,r=(s_c,r-LogMeanExp_j≠c s_j,r)/T_r`。每个角色的200类文本均值形成类别无关查询`q_bar_r`，同一Reader输出`o_r=sigmoid(R(P,q_bar_r))`。分类目标使用`stop_gradient(o)`，不可通过降低可见性逃避错误；干预目标用原图区域模糊与同面积随机区域的完整`d/o`下降差训练可见性和证据区域。三态概率由`P(U)=1-o`、`P(S)=o*sigmoid(d)`、`P(R)=o*sigmoid(-d)`唯一确定，贡献固定为`e=o*tanh(d/2)`。
old_signal_or_primitive: IDEA-163使用候选与4个动态近邻离散判定三态，unknown/refute依赖局部候选集合，attention只有事后删除评估。
new_signal_or_primitive: 候选无关的连续可观察性、固定200类参考下的连续有符号证据，以及真实输入干预提供的因果监督；类别由六个连续证据状态而非单点原型或离散阈值状态表示。
paradigm_shift: GZSL从单点兼容度改写为“类别无关地判断角色是否被观察，再以固定语义参考判断观察结果对每个类别命题的正负证据”；未知由观测过程产生，支持/反驳由固定比较基准产生。
why_not_module: prompt、Reader和attention不是claim；核心是两个可辨识连续变量、固定参考证据原语和干预学习信号。若`o`可被分类梯度操纵、`d`随候选集合变化、删除影响不优于随机或最终只剩普通patch残差，则Idea不成立。
closest_paradigm_work:
  - PAPER-003、APN与TransZero均使用人工类级attributes进行局部化或视觉语义交互；本Idea只用现有文本角色描述，并把未观察从负属性中分离。
  - CREST使用attribute定位与Evidential Deep Learning表达不确定性；本Idea不使用人工attributes或独立证据Head，而以候选无关`o`和固定参考`d`生成严格归一的三态概率。
  - Counterfactual ZSL通过生成样本级反事实做seen/unseen一致性判断；本Idea的干预只用于学习/验证角色证据区域，不生成unseen视觉样本。
  - IDEA-163已经证明动态近邻离散三态和事后attention删除不成立，IDEA-164不能复用其运行代码或把失败公式改名。
minimal_falsification: Gate 1只使用CUB formal-seen图像，seed 7固定100类Reader训练、50类完全隔离评估，true-unseen图像不读取。一个rank-64共享Reader先以固定Top20%类别图像包证据竞争建立`s`；分类损失使用`stop_gradient(o)`。从100类中固定250张图、每图轮换一个角色，用Reader声称的Top区域产生局部模糊和随机同面积干预缓存，再以`L_state+L_causal`完成固定总更新；50类中的另250张图使用不同的均值填充干预只评估。Gate 1同时要求：(1)类别无关`o_r`不接收候选类别输入，代码反例证明候选顺序/数量变化时逐值不变；(2)固定200类参考的`d`在50类真实命题对同角色困难命题pairwise accuracy≥65%；(3)在250张class-disjoint图像上，删除Reader证据区域造成的完整`o`或`|d|`下降大于随机区域的比例≥70%。任一失败立即reject，不进入Gate 2，不调整层、prompt、Top-K、rank、Top20%或干预区域大小。
paper_level_claim: Gate 1、后续100/50净纠错及正式TG+GTD联合训练均成立后，才能窄化声称“用文本角色的候选无关可观察性与固定参考有符号证据，把GZSL点原型兼容度转化为连续证据状态更新”；`L_final=L_TG+GTD+beta*E`只能称为证据logit更新，在完成概率校准前不得称严格贝叶斯后验或声称首次。
evidence_refs:
  - research/ideas/IDEA-162_learnable_concept_readout_probe.md
  - research/ideas/IDEA-163_tri_state_evidence_predicate_set.md
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-162-learnable-concept-readout-seed7/result.json@sha256:4f73cbbd0308b9e96af1342df2f45bb2f89ed0ffb8ec1bf6001e835101b574af
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-163-tristate-predicate-seed7/result.json@sha256:3373601ac0736936d193136895c29178d58619ad585a7590d38f03c9db4bfc91
success_condition: Gate 1三项必须同时通过且打乱角色文本对照不得通过任一主门槛；通过只允许进入Gate 2，不等于范式、H或论文claim成立。
failure_condition: `o`受候选类别影响、`d`不满足固定200类参考、class-disjoint pairwise<65%、第二种干预删除优势<70%、打乱对照通过任一门槛或数据身份不完整，均立即置为rejected并停止当前公式。
owner_decision: 2026-08-29 owner回复“行，开始尝试”，批准IDEA-164从FRAMEWORK-V4准确父commit独立分叉并只执行Gate 1。

