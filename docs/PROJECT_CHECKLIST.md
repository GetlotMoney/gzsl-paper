# FRAMEWORK-V2 简化清单

## 已完成

- [x] V2正式身份：`framework/v2 = v2 = 3dc078c...`。
- [x] V2基线：`H=74.023182%`。
- [x] 最终目标：`H >= 77.023182%`。
- [x] 创新1：`IDEA-001 / TG-VPR-H1`，状态`supported`。
- [x] H1论文关系、组件消融、多seed和参数收口证据已登记。

## 创新2

- [ ] 写一张`IDEA-002`：问题、来源、假设、唯一改动、成功/失败条件。
- [ ] 把第一个尝试写入`experiments/v2/EXPERIMENT_QUEUE.csv`。
- [ ] 每次服务器尝试只更新一行，失败`drop`、有效`keep`。
- [ ] 只有出现值得验证的候选时标记`promote`，再建立`V2-INNOVATION-001`正式目录和HTML图。
- [ ] 正式验证完成后标记`supported / revised / rejected`。

## 创新3

- [ ] 在创新2结果基础上写一张`IDEA-003`。
- [ ] 按同一最小实验流程验证，不增加额外文档层级。

## 最终组合

- [ ] 运行创新1、1+2、1+2+3及必要消融。
- [ ] 完整方法达到`H >= 77.023182%`。
- [ ] 最终组合做多seed，报告mean/min/max/range。
- [ ] 三个创新形成一条连续逻辑，并使用统一命名。

## 发布

- [ ] push前核对分支、Tag和工作树；只有owner明确说`push`后执行。
- [ ] 数据、cache、checkpoint和原始大日志不进入GitHub。

## 当前唯一下一步

`V2-TRY-002`已获得`H=75.587012%`并标记`promote`。下一步建立最小正式`V2-INNOVATION-001`，固定10%unseen残差迁移并做正式验证；不再搜索迁移强度。
