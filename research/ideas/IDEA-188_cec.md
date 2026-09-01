# IDEA-188：Conditional Evidence Completion（CEC）

idea_id: IDEA-188
status: rejected_after_three_rescues
problem_category: visual_grounding
mechanism_tags: [semantic_condition, disjoint_visual_slots, held_out_completion]
base_framework: FRAMEWORK-V5
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
current_advantage: none
performance_status: below_parent

current_run: none

## 核心问题

现有GZSL以图文相似度和事后校正判别类别。CEC把问题改写为：类别语义能否根据图像中两组可见证据，预测被留出的第三组证据，并以真实证据的预测误差验证类别假设。

old_solution_path: `CLS↔文本原型相似度→图/角色校正→argmax`。

new_solution_path: `8角色文本→类别条件；pre-attention局部patch→3个互斥视觉槽；类别条件+另外两槽→预测留出槽；三槽预测误差→200类logits`。

principle_difference: 最终类别分数直接依赖当前实例的条件留出证据预测误差，不再只是图文余弦或OT加权。

old_signal_or_primitive: 单一CLS、点原型、PCLR关系分数和RSE加权。

new_signal_or_primitive: 首个视觉自注意力前、互斥分区得到的三个当前实例视觉证据槽，以及逐槽条件补全误差。

non_equivalence_test: no-tangent语义控制、one-slot视觉控制、同容量direct-compatibility交互控制；CEC-off同checkpoint；text/context/semantic/target shuffle反证。

minimal_viability: 开发集真实类条件能量优于hard wrong，且Full相对三项primary control的paired ΔH均为正；正式`H>=80`且同checkpoint `H_full-H_CEC-off>=0.20pp`。

minimal_falsification: 先运行无泄漏100/50开发Full；若补全权重近零、槽为空/复制、same-class target shuffle保留主要增益或direct control追平，则立即drop。

failure_boundary: pre-attention局部特征语义可能不足；三个槽可能只是空间分桶；基础CLS可能独立支撑分类；这些情况均不得包装为生成—验证成立。

why_not_module: CEC当前只登记为一个由S/V/I三个部署模块共同实现的求解路径创新；语义和视觉部件不能仅凭消融自动升级为另外两项范式创新。

paper_level_claim: 若控制和formal全部成立，仅声称“一次视觉编码下，类别语义条件的互斥视觉槽补全可作为GZSL验证分数”。

## 三模块

- S：8角色文本→local/unique/overall语义与128维类别条件；共享可微切向残差借鉴TG/GTD原理。
- V：最终CLS＋首个自注意力前576个局部patch→三个互斥视觉槽。
- I：每次留出一槽，用另外两槽和类别条件预测它；基础CLS分数减去补全误差得到logits。

没有V5教师、蒸馏、OT或PCLR推理。旧原理只作为分组、迁移和多证据动机。

## 当前实验事实

- D0在真实pre-attention资产上暴露argmax硬分区坍塌，至少一个槽仅分到约0.35%的patch；已停止并保留失败收据。
- R1只把分区改成每槽固定192个patch。4000次训练更新完成后，冻结dev门得到`U=72.6751 / S=79.0333 / H=75.7210 / ZS=78.6843`。
- R1槽位门通过：平均槽余弦`0.4137`、平均有效秩`2.6389`；但补全权重仅`0.012078`，Full与同checkpoint CEC-off逐项指标完全相同，`Delta H=0`。
- 因此R1只能证明三槽没有坍塌，不能证明类别条件补全参与了判别；决策为`drop_no_cec_signal_run_rescue2`。
- 冻结结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/cec/V5-TRY-002-R1-EVAL/metrics.json`，SHA256 `d4642ef6a5a57dab9cad00eb782dee84f68d9178e6c0a9d29620c203515fb3ca`。
- 只读cal诊断显示R1能量对seen困难错类的方向率为`57.45%`，对unseen仅`46.55%`。该结果提示语义泛化风险，但R1的补全训练梯度同时被`/768`削弱，不能提前否定恢复平方L2训练量纲后的R2。
- R2唯一方法变化是把每槽`mean(square residual)`恢复为预注册的`squared L2 sum`；balanced分区、S/V/I接口、seed、优化器、损失权重和4000更新均不变。
- 若R2仍在unseen端无方向，R3优先检验语义解空间：废除固定`6+1+1`聚合，使用类别名驱动的普通自然语言句子集合；禁止使用CUB专家属性、人工属性表或专家规则。
- R2训练把补全误差从`2.0078`降到`0.1208`，但冻结gate仍为`U=72.4349 / S=79.0333 / H=75.5904 / ZS=78.4440`，Full与off完全相同，1702张图无一项预测变化。
- R2能量方向率为seen `54.10%`、unseen `47.14%`；槽位门继续通过。因此量纲修复已排除“补全器没学会”的解释，剩余主要反例是当前`6+1+1`语义条件无法向unseen迁移。
- R2冻结结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/cec/V5-TRY-002-R2-EVAL/metrics.json`，SHA256 `1a700501c05ae6af88914a917792881f53f9ea25eb98e06522fe51f5b9cdf7b2`。
- R3只替换S：每类8条全类统一的整体观察句，仅替换canonical类别名；不使用CUB属性、旧角色句、图像、专家规则、LLM物种事实或网页。S输出一个768维类别原型和一个128维句云条件，V/I与R2逐值不变。
- R3语义资产SHA256为`57ed796222b395f8b4a8972539f61d2136283d1dfb38a8def3e9ea4ea470cb1a`；150类句云非退化率`100%`、有效秩`125.61`，纯文本PCA保留残差能量`99.25%`、投影拓扑相关`0.99992`。
- R3方案与代码经`/root/gev_a`、`/root/gev_b`独立审查和直接交叉：随机投影P1已由纯文本mean-direction＋127残差主成分关闭；最终`P0=0 / P1=0 / PASS_for_proof`。这只批准真实实验，不代表创新或H目标成立。
- R3固定4000步完成：分类CE从`2.7351`降到`0.3769`，补全误差从`1.9988`降到`0.1980`，句云条件残差均值`0.1922`且未饱和；但raw→adapted prototype topology仅`0.6264`，低于预注册`0.80`。
- 冻结评估在calibration阶段命中gamma上边界`1.0`，按合同在gate推理前停止。cal上Full与off全网格完全相同；边界点两者均为`U=70.3642 / S=73.9667 / H=72.1205 / ZS=75.9330`。
- R3失败收据：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/cec/V5-TRY-002-R3-EVAL/failure.json`，SHA256 `c2e97bc1654179bc78b7cd9ddad5e75aa6f0625dfea03667ba96498af2caf8df`；`gate_inference_executed=false`，禁止事后扩gamma网格救活。
- D0、R1、R2、R3均已结束；三次救援预算耗尽，CEC方案1关闭。下一对话不得继续调CEC，若继续总目标应从方案2的双Agent对抗设计开始。
- 2026-09-01 owner更新后续方案口径：`H=80`改为追求目标而非硬性通过线；语义、视觉、交互三个模块各自必须满足同checkpoint `H_full-H_module_off>=1.0pp`。该更新从方案2起执行，不追改CEC运行时合同，也不重开CEC；R1/R2的CEC Full与off均为`Delta H=0`，R3未执行gate，按新口径仍不达标。

## 双Agent对抗记录

review_date: 2026-09-01
review_agents: [/root/gev_a, /root/gev_b]
review_rounds: 3
review_status: owner_approved_after_max_rounds_as_proof_of_path

- Round 1否定异方差Gaussian：其能量可展开成加权余弦与类别偏置。
- Round 2要求条件预测器物理不读取target，并否定final CLIP patch上的伪held-out。
- Round 3改为pre-attention局部patch、互斥硬槽、target/context双detach和CEC-off正式门。
- 最后共同审查为`P0=0 / P1=5 / P2=3 / REVISE`；按三轮上限不再增加Agent轮次。
- owner在了解边界后明确回复“开始”，批准仅作为proof-of-path实现与实验；不等于Innovation或三项paper-core创新通过。


