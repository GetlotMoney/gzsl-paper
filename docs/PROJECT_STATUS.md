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
evaluation_protocol: test_selected_inductive_gzsl
paper_primary_framework: FRAMEWORK-V2
paper_baseline_H: 74.023182
paper_target_H: 78.0
target_supported_innovations: 3
supported_innovations: 3
current_seed7_H: 77.547270
current_multiseed_mean_H: 77.066040
current_best_observation_H: 79.448210
current_best_observation_seed: CRA_training_seed_17
completed_try_count: 110
minimum_required_try_count: 50
```

V1 来源于 `model/v5-template-v2@fb4b29b04087640890a532f105cb527d3a8c461b` 的必要运行代码，旧仓库历史、旧实验和旧账本没有迁入。

## FRAMEWORK-V2

owner已将来源身份`INNOVATION-MODULE-1 / TG-VPR-H1`提升为独立正式框架`FRAMEWORK-V2`。V2使用独立代码、配置和训练入口，不接入`FRAMEWORK-V1`。首个当前仓库正式基线已由`V2-CONFIRM-001 / RUN-001`完成：`U=72.655779%`、`S=75.443041%`、`H=74.023182%`、`ZS=81.534684%`。

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

新的长期目标是稳定达到最高seed `H>=78.0%`、形成3个可解释且有消融支撑的创新，并累计完成至少50组真实实验。执行计划见[`docs/LONG_HORIZON_EXPERIMENT_PLAN.md`](LONG_HORIZON_EXPERIMENT_PLAN.md)。

完整执行顺序和完成条件见[`docs/PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md)。
