# IDEA-133：角色引导局部视觉证据学习

idea_id: IDEA-133
source_type: owner_hypothesis
status: rejected
evidence_refs:
  - V2-CONFIRM-011本地RGVE-off观察：CUB端到端H=76.590293，现有主链主要依赖CLS全局特征
  - model/paper_v2.py代码观察：旧RGVE只有逐patch MLP、静态组权重和单一beta，缺少空间混合与独立局部分类目标
base_commit: 781926531af0190aaa082166fecfe762412adf0e
problem: 当前无专家FRAMEWORK-V2几乎只使用CLIP CLS完成分类，局部patch没有形成能够独立辨别细粒度类别的训练式视觉表示。
hypothesis: 在冻结CLIP patch上学习空间或部件级视觉证据，并用seen-only局部分类、注意力差异与CLIP保持损失训练，可使至少一种端到端或分阶段策略相对同资产视觉off提高至少2个H百分点，且不会造成8点以上的U或S退化。
core_change: 在现有TG-VPR→TST-NTR→CCGR全局原型链之后增加一个训练式角色引导patch证据分支；预注册四种候选结构用于确定唯一最终实现。
success_condition: CUB seed7最佳策略相对配对视觉off的Delta H至少+2.0，另一策略Delta H不低于-0.3，且U/S任一项不下降8点；通过后追加seed5/8和三数据集迁移。
failure_condition: 四候选、顺序调参和最多三次方法级补救后仍无策略达到+2 H，或2/3正式seed出现8点以上U/S退化。
experiment_queue: V2-TRY-148至V2-TRY-157为初始双策略架构筛选。
paper_core_innovation: false
current_result: 四结构seed7初筛完成；Spatial-RGVE分阶段H=78.457262、相对配对off提高1.561426，当前领先但未达到+2；Part Tokens提高1.278456，Multi-Scale提高1.084734，Confusion Refiner淘汰。150轮短模块式实验中，Visual5+Joint的全局最高H=77.461866，Visual10+Joint的全局最高H=77.843587，均未达到相对同资产视觉off提高2个H点的目标。
current_decision: V2-TRY-173保留为当前短模块式最佳观察，但不promote为正式创新；按owner要求在完成账本回填后暂停IDEA-133，不继续调参、补救、多seed或跨数据集实验。
modulewise_result: 50/50/50/50独立训练中，TG阶段最高H=72.230851，TST-NTR阶段提高到77.211374；CCGR阶段最高仅73.217216；Spatial Visual阶段从CCGR末态约73.039提高到76.675680，但未超过TST-NTR阶段。因此当前模块式全局best不包含CCGR或Visual，不能据此晋级视觉核心模块。
modulewise_decision: 暂停分数调参，先解决固定传递最后权重导致的TST过训练，并重新证明CCGR与Visual在冻结父链上的独立增益。
short_modulewise_result: owner指定的两个150轮RUN均完整结束且各保存151次official评估。共同的TST-NTR阶段最高H=77.211374；CCGR阶段最高H=77.052485，未超过TST-NTR。Visual5阶段最高H=77.389847，随后联合微调最高H=77.461866；Visual10阶段最高H=77.689593，随后联合微调最高H=77.843587。后者证明Visual与联合微调均有非零贡献，但相对TST-NTR阶段仅提高0.632213，未达到+2目标。
short_modulewise_decision: V2-TRY-172标记drop，V2-TRY-173标记keep但暂停；两个RUN均为Chen-style test-selected观察，true-unseen图像不进入梯度，且没有阶段内独立选择或拼接checkpoint。
module_strategy_matrix_result: V2-TRY-174至184在同一提交、CUB patch资产和seed7上完成无人工标注的一段式与均衡六段式累计模块矩阵。J条件H依次为Mean8 68.750566、TG 76.657659、TST 76.935667、NTR 77.012635、CCGR 76.590293、Visual 77.148551，对应增量为+7.907093/+0.278008/+0.076968/-0.422342/+0.558258。S条件TG为73.144710，TST提高到77.233743；NTR、CCGR、Visual三个独立RUN的全局best均仍为同一epoch55 TST_ONLY checkpoint，正式独立增量均为0。
module_strategy_matrix_decision: 当前只有TST在两种策略均为正；NTR仅在J下弱增益，CCGR在J回退且在S为0，Visual只在J下提高0.558258且未达到+2。均衡六段式完整M5的best尚未训练NTR/CCGR/Visual，不具备完整链主策略资格；按预注册计划进入有限弱模块调参与M5多seed策略确认，不promote正式实验目录。
strategy_stability_result: M5在seed5/7/8的一段式H为76.945755/77.148551/77.587678，mean/min/max/range为77.227328/76.945755/77.587678/0.641923；均衡六段式H为76.821410/77.233743/77.025836，mean/min/max/range为77.026996/76.821410/77.233743/0.412332。S-J逐seed为-0.124345/+0.085191/-0.561842，均值-0.200332，一段式2/3更高。
strategy_stability_decision: 三个六段式RUN的全局best全部停在TST_ONLY，均不具备完整M5主策略资格；一段式虽为唯一完整链候选，但seed5的U/S差8.860242达到淘汰线，尚不能宣布最终训练策略。owner在seed5/8策略对照完成后取消V2-TRY-189至192调参并要求停止；对应输出目录未创建，当前不继续S-compact、leave-one-out、跨数据集或更多seed。
hard1_handoff_result: V2-TRY-193至205按owner新口径把TST-NTR合并为一个复合模块，并完成J累计加入、五段式stage-best handoff与两种策略的四个单模块移除。五段式四次handoff的保存state SHA与下一阶段加载SHA逐项一致；TG/TST-NTR/CCGR/Visual stage-best H为72.230851/77.537976/77.178787/78.181266，最终Joint完整模型U/S/H/ZS为79.566842/76.948702/78.235875/86.897689。
hard1_module_result: J累计ΔH为TG +7.907093、TST-NTR +0.354976、CCGR -0.422342、Visual +0.558258；J单移除H降幅为TG +7.703174、TST-NTR -0.899084、CCGR +0.007028、Visual +0.558258。S累计ΔH为TG +3.480284、TST-NTR +5.307125、CCGR -0.359189、Visual +1.002479；S单移除H降幅为TG +8.806135、TST-NTR -0.038847、CCGR -0.305697、Visual +0.744451。
hard1_decision: 按累计加入和完整模型单移除均至少+1 H的硬标准，目前只有TG在J下通过两项；TST-NTR在S下累计通过但单移除为负，CCGR两种策略均为负贡献，Visual累计接近或达到1点但单移除不足1点。按owner固定顺序先对TST-NTR执行最多三次单变量调参；未通过前不调CCGR或Visual，也不建立正式消融目录。
hard1_tst_ntr_tune1: max_transport_step从1.5改为1.0后，J累计H=76.798905，相对TG仅+0.141246；J完整H=77.115038，比J的-TST-NTR基线78.047635低0.932597；S完整H=78.267919，比S的-TST-NTR基线78.274722低0.006803。三项均未通过硬门槛，进入第2次单变量调参max_ntr_delta=0.2。
hard1_tst_ntr_tune2: max_ntr_delta从0.1改为0.2后，J累计U/S/H/ZS=73.940527/80.649585/77.149473/84.892416，相对TG仅+0.491814 H；S完整U/S/H/ZS=78.422284/78.196627/78.309293/86.626625，相对S的-TST-NTR基线只下降0.034571 H。J完整RUN按owner指令在epoch153停止，停止前best U/S/H=74.400610/80.068654/77.130641，未形成final metrics。第2次调参未通过硬门槛。
owner_one_stage_reset: owner明确停止全部训练并取消第3次TST-NTR学习率调参；后续只采用一段式端到端训练，不再使用五段式、三段式或stage-best handoff作为候选策略。现有分阶段结果仅保留为历史诊断。基于一段式硬门槛，TG通过，TST-NTR、CCGR、Visual均未达到累计加入和单模块移除同时至少+1 H，当前框架需重新分析并替换失败模块。
