# FRAMEWORK-V2 简化清单

## 已完成

- [x] V2正式身份：`framework/v2 = v2 = 3dc078c...`。
- [x] V2基线：`H=74.023182%`。
- [x] 最终目标：`H >= 77.023182%`。
- [x] 创新1：`IDEA-001 / TG-VPR-H1`，状态`supported`。
- [x] H1论文关系、组件消融、多seed和参数收口证据已登记。

## 创新2

- [ ] 写一张`IDEA-002`：问题、来源、假设、唯一改动、成功/失败条件。
- [ ] 建一个`V2-INNOVATION-001`：`EXPERIMENT.yaml + config + PARAMETER_MATRIX.csv + result.md`。
- [ ] 代码结构变化时增加实验级`framework_diagram.html`。
- [ ] pre-run commit后去服务器运行，完成后做post-run result commit。
- [ ] 根据结果标记`supported / revised / rejected`。

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

创建`IDEA-002`：基于“seen原型被适配、unseen仍保持Mean8”的本地代码观察，写清可证伪假设和失败门槛。无需先找直接来源论文。
