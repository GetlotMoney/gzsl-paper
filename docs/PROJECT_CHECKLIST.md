# FRAMEWORK-V2 简化清单

## 2026-08-23 规范纠正

- [x] owner最终选择Chen-style test-selected为论文主协议，validation-first保留为严格对照。
- [x] Chen-style端到端流程固定为整模型统一H选模，不允许TG-VPR/TST/NTR/CCGR分别取最大。
- [x] V2-CONFIRM-004无专家/专家各完成28,228次更新和201个official评估点；无专家H=`74.933940`，专家H=`78.134714`。
- [x] 端到端专家达到78目标；无专家未达到77.023目标，失败结果完整保留。
- [x] V2-CONFIRM-005无专家分阶段H=`75.491628`，比端到端提高`0.557688`但仍未达到77.023目标；审计限制已披露。
- [x] V2-CONFIRM-005 RESCUE-1步长上限0.5达到H=`76.006848`，再次提高`0.515219`但仍未达标。
- [x] V2-CONFIRM-005 RESCUE-2步长0.75为H=`75.830543`，低于0.5，步长轴关闭。
- [x] 标准开发协议改为`train_loc` 100类训练、`val_loc` 50类类别不相交validation选模。
- [x] `V2-TUNE-001`划分固定为3724张梯度图像、978张val-seen、2355张val-unseen；official test不加载。
- [x] 无专家路线validation H=`76.472964`；专家312维属性路线H=`77.556001`，各自validation调优后Delta=`+1.083037`。
- [x] 两条路线都只使用来源一致的CLIP缓存；专家属性开关是唯一条件差异。
- [x] `V2-CONFIRM-003`按validation冻结配置完成最终重训/test：无专家H=`74.971312`，专家H=`78.751611`，Delta=`+3.780300`。
- [x] 两条最终RUN各只调用一次official test，`test_used_for_selection=false`；结果不用于回改当前方法。
- [x] 旧test-selected结果只作探索观察；人工专家属性链不进入论文主结果。
- [ ] 补齐遗留CLIP缓存准确checkpoint/预处理来源；当前结果保持`strict_blind_claim_eligible=false`。

以下旧清单保留为历史执行记录，其中test-selected或专家属性成绩不再代表当前论文主结果。

## 已完成

- [x] V2正式身份：`framework/v2 = v2 = 3dc078c...`。
- [x] V2基线：`H=74.023182%`。
- [x] 最终目标：`H >= 77.023182%`。
- [x] 创新1：`IDEA-001 / TG-VPR-H1`，状态`supported`。
- [x] H1论文关系、组件消融、多seed和参数收口证据已登记。

## 创新2

- [x] 固定10%测试时迁移四seed有效，但已降级为动机观察。
- [x] V2-TRY-006：3折pseudo-unseen训练ELPT gate。
- [x] 初试后完成3次方法级补救，仍未过门槛，已强制止损。
- [x] IDEA-002标记`rejected`；未建立`V2-INNOVATION-002`。

## 创新3

- [x] 建立`IDEA-003 / ICGR`最小记录和seed7 TRY。
- [x] 完成两次适用补救后仍无提升，已提前止损并标记`rejected`。
- [x] ACGR新候选及一次保守补救仍无提升，已止损。
- [x] IDEA-005 / TST在4/4 seed提升，已建立`V2-INNOVATION-002`并标记`supported`。
- [x] 建立第3个独立且连贯的训练式创新：IDEA-018 / CCGR五seed全部正增益。

## 最终组合

- [x] 完成TG-VPR → TST → NTR → CCGR的seed7链式消融，准确引用真实RUN。
- [x] 最终无SDM的CCGR+ARA结构4seed均达到`H >= 79.26%`；最高`79.386082%`、range=`0.120505`。
- [x] 当前CCGR Gate优化方差已做4个训练seed；H range=`0.068755`。达到78%后的最终结构仍需重新做多seed。
- [x] 纯对角SDM已做4个父CCGR训练seed；3组正增益、1组不变，候选H range=`0.109061`。
- [x] ARA正式实验、4seed最终结果、SDM/CCGR消融与HTML框架图已建立；标记为辅助增强而非第四个核心创新。
- [x] ARA相关工作原文与PDF SHA已登记；ridge属性映射和视觉/语义融合明确不作原创claim。
- [x] 三个创新形成连续逻辑：TG-VPR结构化语义 → TST安全迁移 → CCGR类别条件几何生成。
- [x] CRA类别中心ridge已完成4seed，H range=`0.111388`且U/S/ZS全部正增益。
- [x] CRA正式实验、4seed参数矩阵、模型/日志/指标SHA与HTML框架图已建立。
- [x] CRA标准输出五件套与确定性重跑已由V2-TRY-111真实验证。
- [x] CRA真实SIGTERM、原子checkpoint与新目录resume已由V2-TRY-112验证；跨物理机器仍待第二主机。
- [x] CRA ridge正则0.01/0.1/1.0已收口，固定0.01。
- [x] EBC四训练seed均提高H，range=`0.142025`且gamma非饱和。
- [x] EBC正式实验、饱和消融、参数矩阵与HTML框架图已建立。
- [x] VPA四训练seed均提高H和ZS，range=`0.089564`。
- [x] 最终VEBC四训练seed均提高H，mean=`80.228888`、最高=`80.474080`。
- [x] VEBC正式组合实验、组件消融、参数矩阵与HTML框架图已建立。
- [x] VEBC反向ridge组合对照已完成，最终固定0.01。
- [x] JBEC四训练seed均提高VEBC H，且beta/gamma残差非饱和。
- [x] JBEC正式训练实验、参数矩阵与HTML框架图已建立。
- [x] JBEC gamma残差轴已收口；0.10仅保留单seed微小增益观察，可靠条件固定0.05。
- [x] CNRA四训练seed均提高JBEC H，mean=`80.512704`、最高=`80.712565`。
- [x] CNRA正式实验、类名cache SHA、参数矩阵与HTML框架图已建立。
- [x] 最终80+完整链式消融与统一HTML数据流已建立，三核心创新和辅助语义证据头明确分层。
- [x] 三项核心创新最近工作与claim边界已登记，过宽“首次”表述已禁用。
- [x] TG-VPR结构化LLM描述最近工作PAPER-009/010已补充，claim再次收窄。
- [x] 累计至少完成50组真实实验；当前已完成147组，达到目标后仍继续运行。

## 发布

- [ ] push前核对分支、Tag和工作树；只有owner明确说`push`后执行。
- [ ] 数据、cache、checkpoint和原始大日志不进入GitHub。

## 当前唯一下一步

端到端专家已过78；若继续无专家提升，另建分阶段Experiment并保持整套模型统一Chen-style选模，不能覆盖V2-CONFIRM-004。
