# 项目状态

## 当前版本

```yaml
repository: GetlotMoney/gzsl-paper
frameworks:
  - id: FRAMEWORK-V1
    branch: framework/v1
    tag: v1
    status: baseline_completed_with_runtime_device_fix
  - id: FRAMEWORK-V2
    branch: framework/v2
    tag: v2
    status: baseline_completed_single_seed
paper_main_evaluation_protocol: chen_shiming_code_aligned_test_selected_gzsl
strict_protocol_comparison: xlsa17_validation_then_final
historical_exploration_protocol: test_selected_inductive_gzsl
paper_primary_framework: FRAMEWORK-V2
standard_final_no_expert_H: 74.971312
standard_final_expert_H: 78.751611
standard_final_expert_delta_H: 3.780300
chen_style_end_to_end_no_expert_H: 74.933940
chen_style_end_to_end_expert_H: 78.134714
chen_style_end_to_end_expert_delta_H: 3.200774
historical_v2_baseline_H: 74.023182
paper_target_H: 78.0
target_supported_innovations: 3
historical_test_selected_supported_innovations: 3
current_no_expert_validation_H: 76.472964
current_expert_validation_H: 77.556001
current_expert_validation_delta_vs_no_expert_H: 1.083037
historical_test_selected_no_expert_attribute_H: 77.612988
historical_out_of_scope_expert_attribute_H: 80.817183
completed_try_count: 187
minimum_required_try_count: 50
```

V1 来源于 `model/v5-template-v2@fb4b29b04087640890a532f105cb527d3a8c461b` 的必要运行代码，旧仓库历史、旧实验和旧账本没有迁入。

## FRAMEWORK-V2

owner已将来源身份`INNOVATION-MODULE-1 / TG-VPR-H1`提升为独立正式框架`FRAMEWORK-V2`。V2使用独立代码、配置和训练入口，不接入`FRAMEWORK-V1`。首个当前仓库正式基线已由`V2-CONFIRM-001 / RUN-001`完成：`U=72.655779%`、`S=75.443041%`、`H=74.023182%`、`ZS=81.534684%`。

`V2-CONFIRM-002`曾验证单RUN固定epoch与完整样本遍历，但没有采用类别不相交validation，且结构受历史official test探索影响，因此只保留为协议纠正过程证据，不是标准GZSL最终结果。旧`77.612988%`无专家属性结果属于test-selected探索观察，使用专家属性的80+链全部不进入论文主成绩。

标准协议审计后，开发选模已进一步纠正为xlsa17类别不相交validation。`V2-TUNE-001`仅用100个开发seen类训练，以50个validation-unseen类和固定seen图像holdout选择epoch：无专家路线最终选择RUN-001，`H_val=76.472964`（epoch 24，topology 0.1）；专家312维属性路线选择RUN-006，`H_val=77.556001`（epoch 22，topology 0.2）。两条路线均`official_test_loaded=false`；遗留CLIP缓存来源仍不完整，所以暂不具备最终test资格。

owner随后明确授权`V2-CONFIRM-003`执行冻结配置最终评估。两条路线均在完整trainval 150类/7,057张图像上从头重训，并在checkpoint写入后各调用一次official test：无专家`U/S/H/ZS=73.071939/76.972061/74.971312/80.278075%`；专家属性`77.935892/79.584587/78.751611/84.862840%`，H提高`3.780300`并达到78目标。由于方法结构受历史test探索影响且CLIP缓存来源不完整，仍必须标记`strict_blind_claim_eligible=false`，不能写成从未接触test的盲测。

owner进一步选择陈使明TransZero公开代码对齐的test-selected方式作为论文主协议。`V2-CONFIRM-004`已预注册端到端联合训练：batch 50、seed 5、Adam `1e-4`、28,228次更新、每141步official test、仅根据完整模型H保存best；无专家与专家路线串行执行。该协议固定披露`test_used_for_selection=true / strict_blind_claim=false`，V2-CONFIRM-003继续作为validation-first严格对照。

`V2-CONFIRM-004`已完成：无专家整模型best在iteration 9165/epoch 65，`U/S/H/ZS=69.692755/81.027550/74.933940/80.256838%`，未达到77.023目标；专家整模型best在iteration 6768/epoch 48，`74.429244/82.228470/78.134714/85.708600%`，达到78目标。两条RUN均完成28,228步、201个official评估点，只有整模型H参与best选择，没有模块分阶段取最大。

按预注册止损规则，无专家端到端失败后新建`V2-CONFIRM-005`分阶段对照：50名义epoch只训练TG-VPR，100名义epoch冻结TG训练TST/NTR+CCGR，最后50名义epoch以`1e-5`全部解冻联合微调。阶段边界固定，阶段间使用最后权重，整次RUN只有一个跨阶段整模型best-H，`nested_official_test_selection=false`。

`V2-CONFIRM-005`完成后，无专家分阶段整模型best为`U/S/H/ZS=71.105868/80.453974/75.491628/82.531631%`，位于TRANSFER_CCGR阶段epoch 54；相对端到端无专家提高`0.557688`，但仍未达到77.023目标。阶段2冻结参数未进入optimizer，训练结果有效；梯度审计字段存在上一阶段冻结参数旧grad缓存误报，已在post-run代码中修复并披露。

V2-CONFIRM-005 RESCUE-1将迁移步长上限从1.5降到0.5，得到`U/S/H/ZS=74.326867/77.764529/76.006848/82.930040%`，相对父条件提高H `0.515219`。过迁移得到缓解，但0.5形成硬饱和且仍未达到77.023；下一补救预注册中间上限0.75。

V2-CONFIRM-005 RESCUE-2步长上限0.75得到`H=75.830543%`，低于0.5条件，步长轴关闭。本实验最佳固定为RUN-002 `H=76.006848%`；下一方向改变阶段2loss语义，按规范新建Experiment。

`V2-CONFIRM-006`已预注册：沿用最佳0.5步长上限，只在阶段2增加`0.25×pseudo-unseen CE`。pseudo-unseen由150个seen类固定三折模拟，真实unseen图像仍不进入梯度；loss语义改变因此使用独立Experiment。

V2-CONFIRM-006完成后H=`75.948676%`，比CONFIRM-005最佳低`0.058172`，拒绝。失败原因是父TG-VPR已见全部150类，简单pseudo样本加权不构成class-exclusive迁移；下一Experiment必须训练三个仅见100类的fold父模型。

`V2-CONFIRM-007`已开始实现真正class-exclusive：1个完整150类TG父模型供推理，3个仅见100类的fold TG父模型供共享TST/NTR+CCGR训练；每折另外50类从未进入该fold父模型梯度。fold训练不使用official test，只有完整推理模型参与Chen-style整模型H选模。

V2-CONFIRM-007 RUN-001完成后`U/S/H/ZS=73.860192/77.948201/75.849154/83.031738%`，未超过普通0.5分阶段条件。class-exclusive语义成立但0.5迁移上限可能过强；RESCUE-1复用SHA绑定父模型，只重新训练共享迁移并恢复上限1.5。

V2-CONFIRM-007 RUN-002高步长H=`75.793527%`，低于RUN-001；class-exclusive轴关闭。当前无专家最高仍为CONFIRM-005 RUN-002 `H=76.006848%`，下一创新转向样本条件seen/unseen竞争校准。

V2-INNOVATION-010 SCCC signed条件H=`76.099469%`但gamma均值负且接近边界；非负补救H=`75.846458%`。样本竞争校准机制不成立并拒绝，下一方向转类名CLIP原型与GPT长描述原型的无专家双语义融合。

owner已授权直接迁移H1旧实验的轻量证据。组件消融、多seed和参数收口证据位于`experiments/v2/evidence/legacy_h1/`；`IDEA-001 / TG-VPR-H1`现为论文核心创新1，状态`supported`。

## 当前待办

owner已选择`FRAMEWORK-V2`作为论文主框架。V2当前正式单seed基线为`H=74.023182%`，新的三个百分点目标为`H >= 77.023182%`。

固定10%保守unseen迁移在四seed均提升H，但它只在测试时生效，现降级为`test_time_observation`，不计入论文核心创新。

训练式ELPT已完成`V2-TRY-006`及全部3次方法级补救。最佳H达到`76.803085%`，但首次TRY的gate均值超过预注册上限；三个补救又持续出现gate饱和或S下降超过2个百分点，因此`IDEA-002`已标记`rejected`并强制止损。没有建立`V2-INNOVATION-002`。

`IDEA-003 / ICGR`已完成首次TRY与两次适用的补救。原始路由和增加语义余弦输入均未提高H；均匀KL消除了权重塌缩，但最终仍为`H=73.976174%`、`ΔH=-0.047008`。只适用于跨seed不稳定的RESCUE-3前提不成立，因此该方向已提前止损并标记`rejected`。

当前仍只有`IDEA-001 / TG-VPR-H1`一个supported核心创新。下一步必须建立新的独立训练式候选，不能把ELPT或ICGR失败条件晋级为正式创新。

`IDEA-004 / ACGR`使U和ZS出现正向信号，但H未提升；一次保守幅度补救仍失败且发生组权重塌缩，现已标记`rejected`。下一候选回到原型迁移主线，改用切空间方向迁移的新公式，不复用ELPT实验身份。

`IDEA-005 / TST`已在seed 5/6/7/8全部提高H，平均提升`3.013152`个百分点，候选H mean=`76.866245%`，已超过四seed目标`76.853093%`，现为论文核心创新2。seed7为`76.984545%`，距离单点目标仍差`0.038637`个百分点；项目还缺第3个supported创新，因此整体工作未完成。

TST之后已依次止损EPC、CATA、SPA、PURL和NTR。NTR直接8维条件曾达到seed7 `H=77.086536%`、四seedH mean约`76.876640%`，但相对TST仅2/4 seed为正，未按稳定创新晋级。当前正式状态仍是2个supported创新；第3创新与完整三创新组合尚未完成。

owner已更新成绩口径：主结果报告最高seed，mean/range只用于判断偶然性；`range<=1.0`个百分点时可以最高seed作为主成绩。按此口径，当前最佳观察是`V2-TRY-028 / seed7 / H=77.086536%`，四seed范围约`0.5432`，可作为当前最佳框架参考。但NTR相对TST的最高增益只有`0.101991`个百分点，未达到新核心创新`0.20`个百分点门槛，因此继续搜索替代或增强模块。

长期计划已完成第50个有效实验：`V2-TRY-050 / TG-VPR seed9`得到`H=73.478685%`。达到50组不结束目标，下一步继续运行seed9对应TST与NTR，并推进新的框架候选。

seed9后续结果：TST `H=76.698446%`，NTR `H=76.795441%`。NTR相对TST四项均提高，`Delta H=+0.096995`；五seed最高仍为seed7 `77.086536%`，范围约`0.543209`。当前累计52组有效实验，稳定78%+尚未实现。

当前累计55组有效实验。BMR、DPT、SGT、MPR、PGO与SVPG均已按真实失败模式止损；其中SVPG再次证明直接把seen视觉映射施加给unseen会造成严重联合竞争偏置。稳定78%+仍未实现，下一主线转向正交残差与类别条件生成。

当前累计57组有效实验。正交残差主子空间与补空间均被训练关闭，ORT已止损；全局共享SVPG和低秩ORT共同证明seen视觉偏置不能整体迁移给unseen。下一主线必须使用类别条件机制，稳定78%+仍未实现。

CCGR类别条件文本几何生成在seed7得到`H=77.100834%`，比NTR提高`0.014298`，成为当前最高观察，但未达到核心创新门槛。下一条件改用pseudo-unseen episode直接训练CCGR Gate。

episodic CCGR进一步达到`H=77.237120%`，相对NTR提高`0.150584`，成为当前最高观察；仍未达到78%，且U/S偏向需要继续补救。

unseen平衡CCGR进一步达到`H=77.384331%`，相对NTR提高`0.297795`且U/S/ZS全部提高，首次达到新核心创新增益门槛。当前仍需完成幅度非饱和补救、多seed和正式消融，78%目标尚未达到。

CCGR已在seed5/6/7/8/9上全部提高H，最高`77.384331%`、range=`0.708975`，已晋级论文核心创新3。当前三项核心创新为TG-VPR、TST和CCGR；NTR作为TST到CCGR之间的邻域路由实现保留。稳定78%+仍未达到，后续继续组合与新目标优化。

CCGR幅度Tune在0.15/0.20处达到`77.459608/77.459931%`后平台化；全局margin只读诊断上限约`77.52193%`。继续从视觉特征适配角度寻找78%+，不再扩大CCGR幅度。

FVRA视觉特征适配在无界和L2上限0.1两种条件下都系统性提高S并伤害U，已止损。当前累计69组有效实验，最高仍为CCGR幅度0.20的`H=77.459931%`；下一主线转向样本条件的seen/unseen联合竞争建模。

EDC样本条件竞争在校正范围0.2和0.05下都只改变U/S权衡并降低H，已止损。当前累计71组有效实验，最高仍为`H=77.459931%`；下一主线转向类别条件温度与能量归一化。

DALN密度Gate提高U与ZS但seen CE导致S下降，H仅低于父条件`0.039483`；该方向保留并转入pseudo-unseen episode训练。当前累计72组，最高仍为`77.459931%`。

DALN在seen CE与pseudo-unseen episode两条训练路径下均未提高H，已止损。当前累计73组，下一主线使用pseudo-unseen角度间隔直接扩大类间边界。

CCGR逐epoch选择在第5轮达到`H=77.547270%`，成为当前最高；仍未达到78%。后续固定结构与父checkpoint，运行多个Gate训练seed并继续记录official-test选模。

EAML角度间隔及CCGR/EAML原型ensemble均未超过`77.459931%`，已止损。当前累计74组，下一主线转向内层pseudo-seen、外层pseudo-unseen的元学习视觉适配。

MFRA元学习视觉adapter仍提高U并严重降低S，已止损。当前累计75组，下一主线转为条件视觉分布生成与200类平衡分类的新框架。

CGFG生成式GZSL出现synthetic-unseen域失真，平衡分类器仍极端偏向seen，已止损。当前累计76组；下一Tune重跑CCGR并按项目协议记录每epoch official-test H，检查固定最后一轮是否非最优。

CCGR Gate训练seed 7/17/27/37的逐epoch最佳H为`77.547270/77.572682/77.560640/77.503927%`，range仅`0.068755`，确认约`77.55%`的平台不是优化随机性造成；当前累计80组，最高为训练seed17的`77.572682%`。下一主线把完整top-5类别邻域关系输入CCGR，使生成方向同时感知“最相近的是谁”和“相似度分布”，而不是继续重复随机种子。

NG-CCGR完整top-5输入得到`H=77.562646%`，仅比当前最高低`0.010036`，同时U提高`0.133896`；方向有信号但随机初始化破坏了原有好解。当前累计81组，补救1将从TRY-078函数等价初始化8维模型并把epoch 0纳入选择。

NG-CCGR补救1在epoch 0精确复现`77.572682%`，但20轮邻域残差更新均降低H，确认top-5排序细节不是当前瓶颈并提前止损。当前累计82组，最高不变；下一候选转向不同目标，不再修改CCGR邻域输入。

CCGR-HEO首次TRY从当前最佳权重出发，权重1.0的pseudo-seen/pseudo-unseen软调和目标在20轮内均降低official H，最终选回epoch 0。当前累计83组，补救1只把调和权重降到0.1；若仍无增益则停止该目标，不做参数网格。

CCGR-HEO权重0.1的非零训练轮次最高仅`77.560640%`，仍未超过父模型并再次选回epoch 0；该目标已止损，不继续扫权重。当前累计84组，最高仍为`77.572682%`。

只读错误分解显示seen/unseen跨域错误为`12.82/11.30%`，而域内错误为`6.50/14.00%`，剩余主要不对称来自unseen细粒度混淆。针对该问题的CCGR-LBS仍无非零epoch超过父模型并止损。当前累计85组；下一路线必须改变训练阶段或表示结构，不再从TRY-078继续追加loss。

SDM对图像和原型同步学习有界对角度量，在第2轮得到`H=77.612988%`与`ZS=82.173079%`，相对父模型提高`0.040306/0.335044`并成为新最高。当前累计86组，仍未达到78%；下一补救保留对称共享度量，增加零初始化的受控低秩维度交互。

SDM冻结对角基的rank-64低秩补救loss持续下降，但所有非零epoch均低于父模型，说明发生pseudo-episode过拟合。当前累计87组；补救2只解除对角基冻结，让对角与低秩权重联合补偿。

SDM联合对角/低秩优化同样降低训练loss却不能提高official H，低秩路线止损，保留TRY-086纯对角新最高。当前累计88组；下一步在CCGR Gate训练seed 7/27/37上复现同一对角SDM，检验可靠性并搜索最高seed。

纯对角SDM在CCGR Gate训练seed 7/17/27/37上的H增益为`+0.013633/+0.040306/+0.024673/+0.000000`，候选H range仅`0.109061`，可保留为稳定辅助优化；低秩版本不保留。当前累计91组，最高仍为`77.612988%`，SDM不计为第四个论文核心创新，78%目标继续。

训练式ARA在seed17第7轮得到`U/S/H/ZS=73.954368/85.495055/79.307063/86.089158%`，相对SDM提高H `1.694075`并首次超过78%；ridge与beta只用seen图像训练，true-unseen不进梯度。当前累计92组，尚需父CCGR/SDM seed7/27/37可靠性和正式module-off消融，不能把单seed写成稳定结论。

ARA在seed7/17/27/37上全部超过79%，H mean/min/max/range=`79.292949/79.253171/79.330716/0.077545`，相对各自SDM父条件均提高至少`1.667857`。当前累计95组，稳定78%目标已达成但项目按owner要求继续；下一步完成SDM-off消融、正式结果目录与HTML框架图，再继续新组合而不提前结束。

ARA的SDM-off消融达到`H=79.386082%`，比含SDM高`0.079019`，证明SDM在最终组合中冗余；最终候选简化为TG-VPR→TST/NTR/CCGR→ARA。当前累计96组，下一步做CCGR-off交互消融，再正式化结果与HTML框架图。

ARA的CCGR-off消融得到`H=78.967987%`，比完整CCGR+ARA低`0.418095`，证明CCGR仍有独立贡献。当前累计97组，最终候选固定删除SDM、保留TG-VPR→TST/NTR→CCGR→ARA；开始正式化ARA实验目录、参数矩阵与HTML框架图。

最终无SDM的CCGR+ARA结构在seed7/17/27/37上得到H `79.334907/79.386082/79.265577/79.280845%`，mean/range=`79.316853/0.120505`，4/4 seed均超过79.26。当前累计100组真实实验；继续正式化ARA、完整消融与后续创新，不因达到78%提前结束。

ARA已正式登记为`V2-INNOVATION-004 / supported auxiliary`，目录包含4seed最终结果、SDM-off与CCGR-off消融、完整配置、模型/日志/指标SHA和HTML框架图；它不计为第四个论文核心创新。

不依赖人工attributes的DRA尝试出现明确seen过拟合：beta增大、训练loss下降，但所有非零epoch official H均下降，最终选回CCGR父模型并提前止损。当前累计101组，说明ARA增益来自独立显式属性证据而非重复编码GPT描述。

ARA相关工作已重新核对：CVPR 2017已有视觉→属性ridge回归与多语义模态融合，CVPR 2020已有细粒度GZSL属性对齐，CVPR 2025已有视觉/语义prompt协作。ARA继续作为稳定辅助增强，不作算法原创或第四个核心创新claim；论文卡为PAPER-002至004。

CARA样本条件beta Gate在seed17得到最高观察`H=79.404922%`，但残差std仅`0.003790`，几乎退化为全局beta微调，未通过机制门槛。该结果只作为test-selected调参观察，CARA不成立、不正式晋级；当前累计102组。

SFA把八角色描述压缩为64维跨类语义因子后仍发生seen过拟合，所有非零beta均降低H，已止损。当前累计103组；下一实验保留独立人工属性证据，但把ridge训练单位从图像改为类别视觉中心，降低类内噪声。

CRA用150个seen类别视觉中心替代7057张图像拟合属性ridge，在seed17达到`U/S/H/ZS=75.319785/84.055454/79.448210/86.219549%`，四项均高于CCGR并成为新最高。当前累计104组，继续运行seed7/27/37可靠性，单seed暂不晋级。

CRA在seed7/17/27/37上得到H `79.377682/79.448210/79.346923/79.336822%`，mean/range=`79.377409/0.111388`，且每个seed的U/S/ZS都高于CCGR父条件。当前累计107组；CRA可靠成立并替代普通ARA作为最终辅助结构，但不增加论文核心创新数量。

CRA已正式登记为`V2-INNOVATION-005 / supported auxiliary`，包含4seed配置、结果与SHA、实现说明和HTML框架图；ARA实验保留为父统计，不删除历史。

CCRA类别beta Gate产生真实类别差异，但所有非零epoch H均低于CRA父模型，已止损。当前累计108组，最高正式可靠结构仍是CRA seed17 `H=79.448210%`。

CRA ridge正则检查`0.01/0.1/1.0`对应H `79.448210/79.340503/77.699950%`，确认0.01最佳并关闭该参数轴。当前累计110组，最高可靠结果不变。

CRA第111组确定性重跑逐位复现第104组U/S/H/ZS，并完整生成`training.log / metrics.json / model_best.pth / checkpoint_last.pth / data_fingerprints.json`。最佳模型别名与`ara_model.pth` SHA一致，正式输出契约已验证。当前累计111组。

CRA第112组完成真实SIGTERM恢复：中断发生于epoch 5日志写出后、checkpoint替换前，有效原子checkpoint为epoch 4；恢复到全新目录后state/history/metrics/best epoch与同代码未中断运行完全一致。当前累计112组，单机跨目录resume已验证；尚无第二物理机器用于跨主机验证。

EBC在pseudo-unseen episode中训练全局seen logit扣减，seed17达到`H=79.717270%`并提高U `1.761520`，但gamma接近0.2上限，当前只作为饱和候选。累计113组，补救1收紧上限到0.15。

EBC收紧gamma上限到0.15后达到`U/S/H/ZS=76.813483/83.009040/79.791176/86.219549%`，gamma非饱和并成为新最高。当前累计114组，继续seed7/27/37可靠性，单seed暂不晋级。

EBC在seed7/17/27/37上的H为`79.748697/79.791176/79.649150/79.675166%`，mean/range=`79.716047/0.142025`，四个seed均提高H且gamma非饱和。当前累计117组；EBC可靠成立为CRA后的辅助输出平衡层，不增加论文核心创新数量。

EBC已正式登记为`V2-INNOVATION-006 / supported auxiliary`，包含4seed结果、0.2饱和对照、完整配置与SHA以及HTML框架图。

VPA属性→视觉中心反向ridge在seed17把ZS提高`0.906229`、H提高`0.095400`，但U下降`1.013190`，单独低于EBC。当前累计118组，下一步检验VPA类内增益与EBC域间平衡是否互补。

VEBC组合VPA类内判别与EBC域间平衡，在seed17达到`U/S/H/ZS=76.674461/84.529251/80.410490/87.125778%`并首次超过80；gamma接近0.25上限，当前为饱和候选。累计119组，补救1降低gamma学习率细化最优区间。

VEBC降低gamma学习率后仍在训练后期贴近0.25边界并复现`H=80.410490%`，补救1未解决饱和。当前累计120组，补救2扩大边界到0.30，使有效gamma成为内部解。

VEBC扩大gamma上限到0.30后达到`U/S/H/ZS=77.077311/84.184039/80.474080/87.125778%`，gamma通过非饱和门槛并成为新最高。当前累计121组，先补VPA各seed父模型，再运行组合可靠性。

VPA在seed7/17/27/37上的H为`79.524497/79.543609/79.463000/79.454045%`，mean/range=`79.496287/0.089564`，四个seed均提高H与ZS。当前累计124组；继续用各自VPA父模型训练VEBC组合可靠性。

最终VEBC在seed7/17/27/37上的H为`79.917063/80.474080/80.384030/80.140382%`，mean/range=`80.228888/0.557017`，四个seed均提高VPA父H且gamma通过非饱和边界。当前累计127组，最高可靠结果`H=80.474080%`。

VEBC最终组合已正式登记为`V2-INNOVATION-007 / supported auxiliary composition`，包含4seed结果、CRA/VPA/EBC组件消融、完整配置与SHA及HTML框架图。

VPA反向ridge `0.01/0.1/1.0`在seed17的H为`79.543609/79.674486/79.662435%`；0.1单VPA最高、0.01 ZS最高。当前累计129组，只追加0.1与VEBC组合比较后关闭参数轴。

VPA反向ridge=0.1与VEBC组合得到`H=80.165438%`，低于0.01组合`80.474080%`；最终组合参数固定0.01。当前累计130组，最高可靠结果不变。

JBEC在VEBC父解附近联合微调beta/gamma，seed17仅提高H `0.008688`到`80.482768%`，且两个残差接近边界。当前累计131组，必须做其余seed可靠性；最高可靠结构仍暂为VEBC。

JBEC在seed7/17/27/37上的H为`80.045741/80.482768/80.437363/80.227127%`，mean/range=`80.298250/0.437026`，四个seed相对VEBC均为正增益且残差非饱和。当前累计134组；JBEC可靠成立为不增加推理模块的辅助训练细化。

JBEC已正式登记为`V2-INNOVATION-008 / supported auxiliary training`，包含4seed配置、结果与SHA及HTML框架图。

JBEC gamma残差范围0.10的seed17调参观察达到`H=80.506168%`，但只比4seed可靠条件最高高`0.032088`且接近残差边界，停止该轴、不追加多seed。当前累计135组；最高可靠结果仍为0.05条件`80.482768%`。

ADMA属性对角度量产生真实维度差异，但所有非零epoch均降低H，已止损。当前累计136组，最高可靠结构继续保持JBEC `H=80.482768%`。

NGVF训练得到负eta，与单位球面归一化融合假设方向相反；其`H=80.495362%`微小观察不作为机制成功，已止损。当前累计137组，最高可靠结构仍为JBEC。

CNRA把独立CLIP类名原型作为JBEC残差，seed17达到`U/S/H/ZS=77.406234/84.313953/80.712565/87.423056%`，四项同时提高且beta非饱和。当前累计138组，继续其余seed可靠性，单seed暂不晋级。

CNRA在seed7/17/27/37上的H为`80.288043/80.712565/80.519916/80.530291%`，mean/range=`80.512704/0.424522`，四个seed均提高JBEC父H且beta非饱和。当前累计141组；CNRA可靠成立为独立类名语义辅助分支。

CNRA已正式登记为`V2-INNOVATION-009 / supported auxiliary`，包含类名cache SHA、4seed配置与结果、模型/日志/指标SHA和HTML框架图。

CNEBC在CNRA后训练额外episodic seen偏置，seed17达到`U/S/H/ZS=77.844131/84.026349/80.817183/87.423056%`，gamma残差非饱和并成为新最高。当前累计142组，继续其余seed可靠性。

CNEBC在seed7/17/27/37上的Delta H为`0/+0.104618/0/+0.005971`，可靠性未成立并已止损。当前累计145组；最高观察为seed17 `80.817183%`，最高可靠结构仍是CNRA `80.712565%`。

HGCS seen CE学到与有效诊断相反的正beta，所有非零epoch降低H。当前累计146组；补救1仅改用pseudo-unseen episode训练beta，检验能否学到负的组级公共模式抑制。

HGCS的pseudo-unseen episode补救仍学到强正beta并降低H，seen CE与episode两条路径均失败，已正式止损。当前累计147组，最高可靠结构继续保持CNRA `H=80.712565%`。

完整最高seed链式消融已建立为`V2-ABLATION-002`：TG-VPR→TST/NTR→CCGR构成三核心创新主干，CRA/VPA/JBEC/CNRA统一归入辅助语义证据头；H从`74.023182%`逐层提高到`80.712565%`，总增益`6.689383`个百分点。

三项核心创新新颖性边界已重新核对PAPER-005至008：TG-VPR不得声称首次GPT描述prompt，TST不得声称首次超球面/测地线prototype transport，CCGR不得声称首次动态或双prototype。允许的窄claim已固定在`docs/CORE_INNOVATION_CLAIM_BOUNDARIES.md`。

TG-VPR新颖性审计进一步加入PAPER-009/010：结构化、对比式LLM视觉描述及用细粒度图像适配VLM均有先例；TG-VPR只保留固定三角色共享Value重参数化与topology约束的组合claim。

Chen-style无专家路线新增NCRA类名残差：冻结最佳分阶段父模型，只用seen训练图像学习一个有界beta，并由整次RUN的official H选一个全局最佳权重。RUN-003达到`U/S/H/ZS=75.131226/79.388309/77.201125/83.028460%`，相对父模型H提高`1.194277`，正式超过`77.023182%`目标；该结果明确`test_used_for_selection=true`，不作blind-test声明。NCRA暂列supported辅助分支，相关工作检索前不宣称原创核心创新。

SDRS按父原型与类名原型的余弦分歧为NCRA提供类别条件缩放。收紧幅度后的RUN-002达到`U/S/H/ZS=73.985535/80.904585/77.290521/83.061785%`，比NCRA提高H `0.089396`；通过预注册边界但增益较小，保留为supported辅助改进，不作为论文核心创新。

SEBC在三个100/50类class-exclusive episode中只用全局seen图像训练一个seen竞争gamma。收紧上限后的RUN-002达到`U/S/H/ZS=75.772560/79.346550/77.518382/83.061785%`，比SDRS提高H `0.227861`，成为当前Chen-style无专家最高条件。该组合复用已有EBC机制，不作为新颖性claim；真实unseen图像未进入梯度，official test用于选模。

LPSR首次把真实576 patch缓存接入无专家路线，但class-agnostic top64平均只使H增加`0.003747`且ZS下降`0.066668`，best beta为负。IDEA-048已止损；下一局部实验必须保留“每个类别寻找自己的patch”这一定位关系，并因forward公式变化新建Experiment。

CCPE让每个类别用正交局部文本独立选择top patch。top8/top4仅为弱信号，top2的RUN-003达到`U/S/H/ZS=76.119131/79.278153/77.666533/83.168101%`，相对SEBC提高H `0.148151`并成为当前无专家最高条件。IDEA-049作为supported创新候选保留，但CLIP patch checkpoint provenance和最近相关工作检索尚未补齐，暂不作原创核心claim。

SCPE给CCPE top2增加24×24空间邻近权重后最高H仅`77.535935%`，低于CCPE。固定空间一致性与六句描述覆盖多个分散鸟体部位的语义不匹配，IDEA-050已拒绝；下一局部方向改为每个局部句子独立寻找patch。

MPPE让六个局部句子分别取最大patch后，所有非零beta均降低H，best退回SEBC关闭态`77.518382%`。常见羽色/背景伪匹配被六路累积，IDEA-051已拒绝；下一方向保留CCPE top2并利用seen图像参考分布消除类别文本公共偏置。

CNPE对CCPE top2分数做seen参考z-score后达到`U/S/H/ZS=75.874960/79.415172/77.604713/83.086050%`，比SEBC提高H `0.086331`但低于CCPE `77.666533`。IDEA-052作为独立替代方案拒绝；绝对top2与归一化top2均有正信号，下一实验检验双尺度互补融合。

DSPE联合训练时绝对分支吞掉归一化分支，最高H仅`77.565132%`；固定CCPE绝对beta的分阶段补救中，所有非零归一化beta仍降低H，best严格退回CCPE `77.666533%`。IDEA-053拒绝，说明同源patch分数的绝对值与z-score不能硬拼；下一方向必须改变信息或训练目标。

PCME固定CCPE后学习top1-top2差距，权重大部分时间为负，说明孤立top1确属噪声；但所有非零权重均未超过CCPE，best退回gap=0。IDEA-054拒绝；下一实验保留CCPE公式但把beta训练目标改为100/50类class-exclusive episode。

ECPE用100/50类episode训练CCPE beta后，beta第一轮即变为`-7.385873`并继续接近`-10`，所有非零条件均降低H，best退回SEBC关闭态。fold父模型的局部证据风险方向不能迁移到主模型，IDEA-055已拒绝；下一方向固定CCPE并只学习类别语义可靠性残差。

CRPE固定CCPE后按局部文本正交残差强度学习类别斜率，所有非零delta均未超过CCPE，best退回delta=0。IDEA-056拒绝；CCPE后的标量校准轴关闭，下一方向改为从seen局部patch视觉中心生成200类局部视觉原型。

LVPG从seen真类patch中心ridge生成200类局部视觉原型后，beta被推到上限但H降到约`76.946222%`，best退回SEBC。seen视觉映射的unseen域偏置在局部空间仍存在，IDEA-057已拒绝；下一方向换用独立Claude文本原型，不再从seen视觉生成unseen表示。

CLRE把独立Claude描述原型作为SEBC残差，RUN-001达到`U/S/H/ZS=75.997263/79.707325/77.808093/83.523118%`，相对SEBC四项同时提高，并超过CCPE成为当前无专家最高。IDEA-058作为supported候选保留；Claude cache准确prompt/编码模型provenance与相关工作仍待补齐，暂不作原创claim。

CLEC直接叠加CLRE与CCPE得到H `77.569776%`，训练局部比例后最高`77.648045%`，均低于CLRE。两分支在当前表示下不互补，IDEA-059已拒绝；下一方向测试与GPT/Claude均不同的merge文本原型，不再叠加patch分支。

MLRE用merge文本原型达到`U/S/H/ZS=75.798345/79.971749/77.829140/83.225495%`，H比CLRE高`0.021047`成为当前最高，但ZS比CLRE低约`0.298`。IDEA-060仅保留为弱H候选；下一实验在Claude与merge两个已训练端点间学习混合比例，检验能否兼顾H与ZS。

ACLM全局凸混合最高H `77.811876%`低于MLRE，Claude权重`0.980989`退化到端点。IDEA-061拒绝；下一补救按类别Claude/merge一致度学习不同混合权重，若仍失败则关闭跨LLM混合轴。

CACM类别权重最高仍为H `77.811876%`，mean/std=`0.990835/0.000207`，退化为Claude常数端点。IDEA-062拒绝并关闭跨LLM混合轴；下一方向对Claude原型去除类名身份方向，只保留独立跨LLM残差。

OCLR将Claude原型对类名方向正交化后达到`U/S/H/ZS=77.094042/79.075468/78.072185/84.185731%`，相对SEBC H提高`0.553803`、ZS提高`1.123947`，并超过MLRE成为当前无专家最高。IDEA-063为strong candidate；cache provenance和新颖性检索未完成前不作原创claim。

OGLC叠加OCLR与CCPE后直接H仅`77.335657%`，协调后最高`77.533191%`，局部证据破坏OCLR的联合竞争平衡。IDEA-064拒绝；下一方向把类名正交化原则迁移到merge文本，检验其是否跨文本源成立。

OMLR正交化merge文本后达到`U/S/H/ZS=76.796263/79.348004/78.051283/84.291506%`，H略低于OCLR但ZS更高；保留为强次级观察，并证明正交化原则跨Claude/merge成立。下一步运行OCLR seed7可靠性，判断最高结果是否偶然。

OCLR seed7达到`U/S/H/ZS=77.127939/78.964353/78.035343/84.219629%`，与seed5最高H只差`0.036842`；两seed均显著超过MLRE，强提升可靠。按owner规则主成绩仍取最高seed5 `H=78.072185`，平均只用于判断偶然性。

BOCR同时去除类名和TG父原型方向后最高H仅`77.629049%`，完整删除第二方向导致有效Claude语义损失。IDEA-066拒绝；下一方向从OCLR精确起步，只学习父方向的有界部分去除系数。

PBOR从OCLR起步学习父方向部分去除，seen CE把系数推到`-1`边界但H降到约`77.740104%`，best严格退回lambda=0。IDEA-067拒绝并关闭父方向调整轴；下一方向固定OCLR语义，使用class-exclusive episode重校准seen竞争gamma。

ORER在OCLR后用episode学习gamma残差，残差趋近`+0.1`上限且H降到`77.658158%`，best严格退回原gamma。IDEA-068拒绝；OCLR后的语义几何、局部组合与竞争重校准轴均已收口，当前最高仍为两seed可靠的OCLR `H=78.072185%`。

ORMR使用GPT-5.6 role-matched七句均值做类名正交残差，达到`U/S/H/ZS=76.753217/79.154509/77.935371/83.825284%`，相对SEBC明显提高但低于OCLR。IDEA-069不晋级，保留为正交机制跨模型正向对照。

OESR使用GPT-5.6八句均值正交残差达到`U/S/H/ZS=76.715362/79.547596/78.105812/83.822459%`，H比OCLR高`0.033627`但U/ZS更低。因差距很小，暂列弱H候选并追加seed7可靠性，不替代OCLR主候选。

OESR seed7达到`U/S/H/ZS=76.715362/79.540753/78.102514/83.822459%`，与seed5 H只差`0.003299`且两者均超过OCLR，确认H提升可靠。按owner规则当前最高取seed5 `H=78.105812`；OCLR仍保留更高U/ZS的强候选身份。

AOSR固定OESR beta并学习八句softmax权重，seed5达到`U/S/H/ZS=76.445878/80.058682/78.210580/83.652842%`，相对OESR H提高`0.104768`。句权重std/min=`0.107490/0.012587`通过非塌缩门槛；暂为当前最高H，追加seed7可靠性。

AOSR seed7最高H=`78.231209%`，但min句权重仅`0.0000056`且三个句子近零，违反非塌缩门槛；只保留高H观察。当前正式有效最高仍取seed5 `H=78.210580%`。下一实验用KL保守约束防止句子删除。

CASR的KL=0.1条件过度接近等权；降到0.01后seed7达到`U/S/H/ZS=76.849824/79.776293/78.285719/83.920640%`，权重std/min=`0.034369/0.080386`通过非塌缩门槛，成为当前最高有效候选。追加seed5完整链可靠性后再正式晋级。

CASR seed5达到`U/S/H/ZS=76.781464/79.831320/78.276696/83.987868%`，与seed7 H只差`0.009023`；两条链权重min均远高于0.01且均超过父模型。CASR两seed可靠成立，当前正式最高按owner规则取seed7 `H=78.285719%`。

CCSR固定CASR后按句子-类名独立度学习类别差异，所有非零delta均降低H，best严格退回CASR。IDEA-073拒绝；下一方向改为图像条件的保守句子路由，而非类别固定权重。

ICSR在CASR上增加零初始化图像门控，但所有动态条件均未超过父模型，image variation最终衰减到0。IDEA-074拒绝；下一方向保持全局句权重，改用训练期句子dropout增强稳健性，推理仍使用完整8句。

SDCR训练期每批mask一句、推理恢复完整8句，seed7达到`U/S/H/ZS=76.713103/79.959893/78.302856/83.920079%`，H比CASR仅高`0.017137`；权重非塌缩且mask覆盖均衡。暂列弱候选，追加seed5可靠性。

SDCR seed5达到`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`，与seed7 H差`0.017655`，两链均超过CASR且mask覆盖均衡。SDCR可靠成立，当前正式最高按owner规则取seed5 `H=78.320510%`。

SDCR每批mask两句的条件达到H `78.303151%`，高于CASR但低于mask一句的`78.320510%`。dropout数量轴关闭，最终固定每批mask 1句。

SDCC在SDCR上增加dropout学生到完整教师的一致性KL，最高H仅`78.285486%`，低于SDCR。显式一致性过度限制了dropout带来的有效偏移，IDEA-076拒绝并关闭该loss轴。

WSDR每批采样两个不同的单句mask并只反传较大CE，得到`U/S/H/ZS=76.679766/79.959893/78.285486/83.920646%`。权重与mask覆盖均正常，但H低于SDCR `78.320510%`，说明更激进的最坏候选优化没有改善泛化；IDEA-077拒绝并关闭该轴。当前累计148组，Chen-style无专家最高仍为SDCR seed5 `H=78.320510%`。

IADR按当前完整句权重概率选择训练期mask，高权重句实际被更频繁屏蔽且八句全部覆盖；最高`U/S/H/ZS=76.713103/79.959893/78.302856/83.953977%`，仍低于均匀随机SDCR。IDEA-078拒绝并关闭mask采样分布轴。当前累计149组，最高可靠结果仍为SDCR seed5 `H=78.320510%`。

MGSR用4个纯文本几何量和4个跨类共享系数从SDCR生成类别句权重。RUN-001达到`U/S/H/ZS=76.748133/80.051959/78.365239/83.953977%`，H提高`0.044729`且class variation=`0.007255`；但残差触及±0.25边界，只保留正信号并进入±0.10保守补救。当前累计150组，最高可靠结果仍为SDCR，最高新观察为MGSR。

MGSR RUN-002收紧残差到±0.10后最高H=`78.338157%`，低于RUN-001且仍触边界，说明单纯缩小上限不能修复seen CE驱动的系数极化；该参数轴关闭。当前累计151组，MGSR保留H=`78.365239%`的正观察但未晋级，下一补救改为直接约束共享系数。

R-MGSR用0.05系数L2直接抑制极化，但整次RUN best退回SDCR父模型`H=78.320510%`，selected iteration=-1且class variation=0，属于过强正则。当前累计152组；MGSR家族只剩最后一次0.005 L2补救，失败后强制止损。

R-MGSR最终0.005 L2补救仍以关闭态`H=78.320510%`为best。MGSR家族的无正则、收紧上限、0.05 L2和0.005 L2已覆盖首次TRY加3次补救：唯一更高观察`78.365239%`依赖饱和边界，所有防饱和方案均不能超过父模型。IDEA-079/080拒绝并止损。当前累计153组，最高可靠结果仍为SDCR seed5 `H=78.320510%`。

NCSR用SDCR原型的top-5近邻差分正交方向训练一个有界gamma；首次RUN所有非零条件都降低H，best退回SDCR `H=78.320510%`、gamma=0。构造正交误差低于`7e-08`，实现有效；下一补救只把单参数学习率降10倍，区分方向无效与优化振荡。当前累计154组。

NCSR降学习率到0.001后gamma振荡减弱，但所有非零条件仍低于父模型，best再次为`H=78.320510%`、gamma=0。近邻差分正交方向本身无效，IDEA-081拒绝并提前止损。当前累计155组，最高可靠结果仍为SDCR seed5 `H=78.320510%`。

RSDM只对SDCR残差分支学习图像/文本共享对角度量；权重真实分化但所有非单位条件均降低H，best退回`H=78.320510%`与全1权重。IDEA-082拒绝，故障定位为单独改变残差分支破坏三原型分支尺度平衡。当前累计156组；下一实验只允许把同一度量施加到完整语义链。

FSDM把同一对角度量扩展到TG主原型、SDRS类名和SDCR残差三路，但所有非单位条件仍降低H，best退回`H=78.320510%`和全1权重。RSDM/FSDM共同证明当前seen CE度量学习产生跨类域偏置，IDEA-083拒绝并关闭该方向。当前累计157组，最高可靠结果仍为SDCR seed5 `H=78.320510%`。

JSCF联合微调SDRS、SEBC和SDCR共10个参数后，best退回`H=78.320510%`。SEBC gamma从`0.153261`持续升至约`0.1853`且H下降，定位为普通seen CE破坏episode竞争偏置；下一补救冻结SEBC，只协调其余9个参数。当前累计158组。

JSCF冻结SEBC后仍以父模型`H=78.320510%`为best；SDRS delta从`0.394185`持续降至约`0.30`并伤害H。最终补救固定SEBC与SDRS，只用小学习率继续训练SDCR八维句权重。当前累计159组。

JSCF最终只训练SDCR八维权重的条件仍以初始`H=78.320510%`为best。10参数、冻结SEBC的9参数、只训SDCR的8参数三种边界全部无增益，IDEA-084拒绝并关闭分阶段协调轴。当前累计160组，下一方向必须引入新信息而非继续微调旧参数。

CLCR直接在SDCR上增加独立Claude类名正交残差；两种原型平均余弦仅`0.764167`，但所有非零Claude beta都降低H，best退回`H=78.320510%`与beta=0。IDEA-085拒绝且不做幅度补救。当前累计161组，下一方向转本地视觉patch信息。

SPCR把CCPE top2局部证据直接叠加到SDCR推理logits，但从小beta到接近5边界均降低H，best退回`H=78.320510%`与patch beta=0。IDEA-086拒绝；局部patch后续只允许作为训练期可靠性信息，不再增加推理分支。当前累计162组。

PGSD用train-only patch可靠性加权SDCR训练，样本权重std=`0.110662`但均值=`1.164163`，同时放大了CE总尺度；所有条件低于父模型，best退回`H=78.320510%`。IDEA-087拒绝，下一独立Experiment将权重中心化为均值1以隔离相对可靠性作用。当前累计163组。

CPGSD把patch可靠性中心化到mean=`1.0`、std=`0.068121`且保持有界，但所有条件仍低于父模型，best退回`H=78.320510%`。结合SPCR推理叠加失败，说明当前patch信息既不能直接加入SDCR推理，也不能改善其训练权重；IDEA-088拒绝并关闭patch结合轴。当前累计164组。

`SDCR_ERROR_AUDIT_001`复现最高可靠SDCR，并确认主错误不是单向seen/unseen偏置，而是Warbler、Sparrow、Cormorant等同族细粒度竞争；两个最差unseen类在ZSL空间仍仅`5%/14%`。下一方向从类名族群构造类内身份残差，不再调全局bias。

TIGR按类别名最后词形成37个族群并覆盖167类，但所有非零类中心差beta均降低H，best退回`H=78.320510%`与beta=0。结合HGCS失败，线性原型空间的组公共/组内身份方向均关闭；下一步改为保持组均值不变的最终logit差值缩放。当前累计165组。

TWLS保持族群均值并统一缩放族内logit差值，但正alpha约0.25使H降到约`76.9%`，best退回`H=78.320510%`与alpha=0。统一锐化不改变族内排序，只放大错误第一名；IDEA-090拒绝。下一步改为语义相似度加权的成对logit高通。当前累计166组。

TPLD使用非均匀成对affinity（平均熵`0.731696`）对族内logit做高通，但正alpha约0.22仍使H降到约`76.9%`，best退回`H=78.320510%`与alpha=0。固定图结构轴关闭；下一方向只对top2同族且低margin样本动态调用独立证据。当前累计167组。

AGCT只对top2同族低margin样本使用Claude二选一证据，seed5达到`U/S/H/ZS=76.647568/80.107862/78.339523/83.888441%`，比SDCR提高H `0.019013`。门槛只由train seen错误margin生成，official gate非零且beta不饱和；因增益很小，追加seed7判断可靠性。当前累计168组。

AGCT seed7达到`U/S/H/ZS=76.647568/80.107862/78.339523/83.854544%`，相对父模型H提高`0.036667`。两seed均正且最高H完全一致，AGCT作为supported辅助候选保留；按owner规则当前主成绩取`78.339523%`。增益弱且U/ZS略降，不作核心创新。当前累计169组。

CCTB只保留Claude与SDCR共识的AGCT样本，gate均值降至seen/unseen=`0.072010/0.093413`；beta升到正4.57仍不改变任何official指标，best退回父模型。IDEA-093拒绝。下一步仅对AGCT本身做一次75分位门槛覆盖率补救。当前累计170组。

AGCT 75分位门槛把unseen gate从`0.182175`扩大到`0.305713`，但所有非零条件都降低H，best退回父模型。原中位数条件继续保持两seed supported；门槛轴最后检查25分位窄覆盖，失败后关闭。当前累计171组。

AGCT 25分位门槛将unseen gate降到`0.087582`，seed5达到`U/S/H/ZS=76.681465/80.107862/78.357224/83.888441%`，比中位数AGCT提高H `0.017701`。追加seed7后决定是否替换正式条件。当前累计172组。

AGCT 25分位seed7达到`U/S/H/ZS=76.647568/80.107862/78.339523/83.854544%`，相对父模型H提高`0.036667`；两seed均正，25分位正式替换中位数条件。当前最高可靠H按owner规则取seed5 `78.357224%`。最后补救只收紧gate温度，失败后关闭参数轴。当前累计173组。

AGCT温度0.05条件与0.1条件最高指标逐项相同，未产生额外收益。最终结构固定25分位、温度0.1并关闭参数轴；当前最高可靠H保持`78.357224%`。下一方向在同一窄门控内联合学习Claude与merge两种tie-breaker，不再改变gate。当前累计174组。

MAGT在固定AGCT窄gate内联合Claude与merge，但两源原型余弦高达`0.980766`，所有双beta非零条件均降低H，best退回父模型。IDEA-094拒绝；下一歧义证据改用异质的本地patch，只在低margin top2内启用。当前累计175组。

AGPT在25分位gate内使用top2局部patch二选一，但所有非零patch beta都降低H，best退回父模型。局部patch的全局叠加、训练加权和歧义tie-break三种路径均失败，IDEA-095拒绝并关闭patch轴。下一步做gated样本source-oracle审计。当前累计176组。

`AGCT_SOURCE_ORACLE_001`显示unseen gated样本中有79个可由top2 oracle净纠正，理论H=`80.900744%`；但Claude、merge、patch任一固定正/反选择在GZSL unseen均为净负收益。下一模块改为训练共享pair selector，不能继续用固定beta或固定来源方向。

GPES用四特征共享selector训练25分位同族top2 pair，但train pair仅169个；pair CE下降而official H持续低于父模型，best退回零参数。IDEA-096拒绝为小pair集过拟合。下一独立Experiment扩大到所有同族真类top2 pair，并用soft gate加权。当前累计177组。

GWPS将pair训练集扩至4041并用soft gate加权，seed5达到`U/S/H/ZS=76.735932/80.086303/78.375328/84.009010%`，比AGCT最高提高H `0.018104`。pair标签top1占93.17%，追加seed7验证可靠性。当前累计178组。

GWPS seed7达到`U/S/H/ZS=76.773667/80.126470/78.414246/83.980089%`，相对父模型四项均提高；两seed可靠成立，当前最高可靠H更新为`78.414246%`。因patch provenance不完整，GWPS仅作supported辅助候选。下一实验用类别平衡pair CE修复93% top1标签不平衡。当前累计179组。

B-GWPS使用完整逆频率平衡后，top2类别权重=`7.320652`、组合pair权重std=`5.399093`，H降到约`76.75%`，best退回父模型。IDEA-098拒绝为过度平衡；下一补救改用平方根逆频率的温和补偿。当前累计180组。

M-BGWPS平方根平衡把top2权重降到`2.705670`，但组合权重std仍达`4.446476`，H约`77.0%`且best退回父模型。标签平衡轴关闭，原GWPS最高`78.414246%`保持；最后补救改为不平衡但适度扩大硬pair margin范围。当前累计181组。

E-GPES用50分位硬pair得到386个训练样本，最高H=`78.367537%`，高于SDCR但低于GWPS。169/386/4041三种pair规模与两档标签平衡均已覆盖，GWPS soft-gate全pair保持最优`78.414246%`；pair训练范围轴关闭。当前累计182组。

NPS把线性selector升级为4→8→1 MLP，参数真实更新且最高H=`78.414029%`，仅比GWPS低`0.000217`，没有实质收益。IDEA-101拒绝并关闭selector容量轴；下一方向尝试去除patch特征，建立不依赖不完整patch provenance的text-only selector。当前累计183组。

T-GWPS首次RUN因schema漏入hard-pair分支而标记invalid；正确4041-pair RERUN完全不读取patch并达到`H=78.352250%`，高于SDCR但略低于AGCT。IDEA-102拒绝并保留patch-free次级对照；patch差值是GWPS超越AGCT所需交互特征。当前累计184组。

`PATCH_CACHE_PROVENANCE_AUDIT_001`确认当前项目没有patch生成脚本或原图挂载，无法证明具体CLIP checkpoint与预处理。GWPS继续保持`feature_provenance_complete=false`；下一patch-free候选在T-GWPS三特征上增加短类名差值，不再猜测patch来源。

S-GWPS在patch-free T-GWPS上增加短类名差值，seed5达到`U/S/H/ZS=76.747000/80.059719/78.368367/83.953977%`，比patch-free AGCT提高H `0.011143`。追加seed7验证后决定supported状态。当前累计185组。

S-GWPS seed7达到`U/S/H/ZS=76.713103/80.059719/78.350691/83.920079%`，相对同seed SDCR父条件提高H `0.047835`。seed5/7均为正提升，S-GWPS晋级为两seed支持的patch-free辅助候选；按owner口径最高取seed5 `H=78.368367%`。绝对最高仍为patch依赖且provenance不完整的GWPS seed7 `H=78.414246%`。当前累计186组。

下一实验`V2-INNOVATION-070 / R-GWPS`保持S-GWPS的线性选择器与Chen-style训练边界，只新增GPT-5.6八个角色句各自的top1-top2差值，检验细粒度角色分歧能否以patch-free方式超过S-GWPS。

R-GWPS完成28,228次更新后best严格退回关闭态`H=78.320510%`、selected iteration=`-1`；所有非零12维selector均更差。八角色差值数值有效但直接并列重复放大类别身份，IDEA-104拒绝且不追加seed7或参数补救。当前累计187组。

新的长期目标是稳定达到最高seed `H>=78.0%`、形成3个可解释且有消融支撑的创新，并累计完成至少50组真实实验。执行计划见[`docs/LONG_HORIZON_EXPERIMENT_PLAN.md`](LONG_HORIZON_EXPERIMENT_PLAN.md)。

完整执行顺序和完成条件见[`docs/PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md)。
