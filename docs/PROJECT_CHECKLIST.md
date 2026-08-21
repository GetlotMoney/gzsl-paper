# FRAMEWORK-V2 项目清单

## 已锁定事实

- [x] 论文主框架由owner选择为`FRAMEWORK-V2 / TG-VPR-H1`。
- [x] `framework/v2 = v2 = 3dc078c0d52bf358bf24a26e48346c97de9e99ca`。
- [x] V2与V1保持两条独立代码和训练路径。
- [x] V2首个当前仓库正式基线完成：`U=72.655779%`、`S=75.443041%`、`H=74.023182%`、`ZS=81.534684%`。
- [x] 论文指标目标固定为`H >= 77.023182%`，即相对V2基线提高`3.00`个百分点。
- [x] 评估协议固定为`test_selected_inductive_gzsl`，不能描述为blind-test。
- [x] 论文最终需要三个获得证据和实验支持、逻辑连贯的核心创新点。

## P0：现在立即做——为创新1建立当前仓库证据

- [x] 核对《Enhancing CLIP with GPT-4: Harnessing Visual Descriptions as Prompts》本地原文。
- [x] 在仓库外固定PDF绝对URI和SHA256。
- [x] 创建`research/papers/PAPER-001_enhancing_clip_with_gpt4.md`。
- [x] 简单记录论文与TG-VPR-H1的改版关系；按owner要求不保留页码。
- [x] 分开记录论文事实和本项目推断。
- [x] 创建`research/ideas/IDEA-001_tg_vpr_h1.md`，把TG-VPR-H1登记为创新1候选。
- [x] IDEA-001记录证据、base commit、可证伪假设、成立/失败条件和论文主线角色。
- [x] owner明确授权直接迁移H1旧实验的轻量证据。
- [x] 迁入主方法、组件消融、多seed和最终收口四组证据及逐文件SHA。
- [x] 更新`research/IDEA_TREE.md`；IDEA-001现为`supported / paper_core_innovation`。

P0完成条件：`PAPER-001 → IDEA-001 → FRAMEWORK-V2`形成可追溯闭环。当前已完成。

## P1：创新2——seen几何向unseen可靠迁移

- [ ] 从当前代码和V2结果记录本地观察：H1只改写seen原型，unseen保持Mean8，可能存在迁移断层。
- [ ] 建立`IDEA-002`，设置`source_type: local_observation`，引用V2代码、`V2-CONFIRM-001`和H1消融证据。
- [ ] 提出可证伪的unseen迁移机制，而不是继续调H1的head、组权重、inner或topology。
- [ ] 预先写明预期改善U、S、H中的哪一项，以及失败门槛。
- [ ] Idea与本地证据完整后即可创建正式`V2-INNOVATION-001`，不以先找到论文为硬门。
- [ ] 在形成论文新颖性claim前检索最接近的相关工作；若找到论文，再补PAPER记录和差异说明。
- [ ] 代码改变后必须提供实验级`framework_diagram.html`，展示相对V2的数据流变化。
- [ ] 完成主条件、机制控制和必要消融；失败结果保留。

P1完成条件：创新2有可追溯证据、独立Idea、正式实验和可复核结果；论文新颖性claim前完成相关工作检索。

## P2：创新3——单图像动态证据选择

- [ ] 从图像与三组语义匹配的代码/实验现象建立本地观察，或引用新的论文/实验来源。
- [ ] 建立独立`IDEA-003`并记录`source_type`，说明它如何使用创新1的三组语义，并与创新2自然衔接。
- [ ] 预注册动态选择的输入、输出、作用位置、成功条件和失败条件。
- [ ] 只有证据和Idea完成后，才创建下一项正式V2 Innovation实验。
- [ ] 提供实验级HTML图、控制实验和消融，证明提升来自动态证据机制而非参数量或test搜索。

P2完成条件：创新3有可追溯证据、独立Idea、正式实验和可复核结果；论文claim前完成相关工作检索。

## P3：三创新组合与指标门

- [ ] 三个创新分别通过独立验证，不用失败或仅proposed节点凑数。
- [ ] 明确三个创新的共同研究问题、前后输入输出和互补关系。
- [ ] 运行创新1、1+2、1+2+3及必要baseline-off组合。
- [ ] 最终完整方法达到`H >= 77.023182%`；未达到时如实记录，不修改门槛制造成功。
- [ ] 对最终组合做预注册多seed验证，报告mean/min/max/range，不能只报告最好seed。
- [ ] 完成U/S/H/ZS、best epoch、数据身份、配置SHA、日志和模型SHA回填。

## P4：论文整体逻辑与命名门

- [ ] 用一句话说明论文只解决什么核心问题。
- [ ] 一个总方法名能够自然覆盖三个创新。
- [ ] 三个子创新名称风格统一、含义直接、容易写入标题和摘要。
- [ ] HTML总框架图展示连续数据流，不出现三个互不相干的外挂模块。
- [ ] Introduction中的三个贡献点与Method三个部分、Experiments三组证据一一对应。
- [ ] 每个论文claim都能回指PAPER证据或正式实验结果。

## Git与发布

- [ ] 将V2与实验分支发布到GitHub；此项必须等待owner明确说“push”。
- [ ] push前再次确认`framework/v2`与annotated Tag`v2`仍指向`3dc078c...`。
- [ ] 数据、cache、checkpoint、原始大日志和密钥不进入GitHub。

## 当前唯一下一步

开始P1：基于“seen原型被适配、unseen仍保持Mean8”的本地代码观察建立`IDEA-002`，写清可证伪假设和失败门槛；Idea完成后即可准备创新2实验，不必等待直接来源论文。
