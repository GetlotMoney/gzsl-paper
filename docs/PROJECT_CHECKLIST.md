# FRAMEWORK-V2 简化清单

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
- [x] 累计至少完成50组真实实验；当前已完成104组，达到目标后仍继续运行。

## 发布

- [ ] push前核对分支、Tag和工作树；只有owner明确说`push`后执行。
- [ ] 数据、cache、checkpoint和原始大日志不进入GitHub。

## 当前唯一下一步

执行`docs/LONG_HORIZON_EXPERIMENT_PLAN.md`；SDM方向已产生新最高，下一步增加受控低秩交互并继续验证。
