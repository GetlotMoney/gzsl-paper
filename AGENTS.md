# gzsl-paper 协作规则

## 沟通与范围

- 默认中文，先讲结论。
- 每次只做一个明确交付物，不顺手扩张流程。
- 不确定的代码起点、实验变量或评估口径必须显式说明，不能猜。

## Git 管理

- `main` 是当前已接纳代码和轻量账本的默认分支，禁止 force-push。
- `framework/vX` 与 Tag `vX` 固定在同一正式框架 commit，均不移动。
- 不创建 `framework/vX-template-vN` 或 `model/vX-template-vN`。
- 每个实验由 owner 明确指定准确 base ref/commit；实验归属版本不自动决定代码起点。
- 实验分支使用 `exp/vX/<kind>/<id>-<slug>`；临时实现分支使用 `codex/<slug>`。
- 失败实验保留结果，但失败代码不合入正式框架。
- 只有 owner 明确接纳后，创新才能晋级为新的 `framework/vY` 与 Tag `vY`。

## 实验

- 一个 Experiment 回答一个研究问题；一个 Experiment 可以包含多组 Condition 和多行 RUN。
- 只改配置、参数、seed 或预注册控制条件：新增 RUN。
- 修改公式、模块输入、forward、loss、数据语义或评估语义：新建 Experiment。
- 每个 RUN 必须绑定准确 code commit、完整配置快照、seed、数据身份、U/S/H/ZS、日志和模型 URI。
- pre-run commit 只保存代码、配置和计划；post-run commit 只保存已经发生的结果。
- GitHub 不保存数据、cache、checkpoint、原始大日志或密钥。

## HTML 框架图

- 每个正式代码身份必须有可直接打开的 HTML 框架图；正式框架固定为 `experiments/vX/framework_diagram.html`。
- HTML 图必须记录准确 code commit，并说明输入、输出、关键模块、张量或数据流、loss、最终 logits/metric 出口和评估边界。
- 新增或改写 module、forward、loss、数据流、输入输出或评估语义时，所属 Experiment 必须新增或更新自己的 `framework_diagram.html`，并清楚标出相对 base commit 的变化。
- 只改参数、seed、epoch、文档或不改变计算语义的运行修复时，不重复复制框架图；实验必须链接所属框架图并记录准确代码差异。
- HTML 框架图是代码结构事实源之一；旧截图、聊天描述或只有 Markdown 的占位图不能替代它。

## 研究知识

- 新研究知识从本仓库重新建立；不迁移、不隐式复用旧 GTPJ 的论文笔记、idea tree、研究结论或编号。
- 论文、证据、idea 与实验之间的最小闭环以 `research/README.md` 为唯一规则入口。
- 外部论文需要重新核对原文并记录来源；旧聊天、旧笔记和 Agent 记忆不能直接作为证据。
- 正式创新实验必须绑定一个本仓库 `idea_id`，并引用支撑该假设的 `evidence_refs`。
- PDF 原件不提交 Git；仓库只保存来源、定位信息、必要摘要、文件 URI/哈希和研究判断。

## 三创新论文主线

- 项目目标是最终筛选并验证 `3` 个有实验支持的核心创新点；不能为了凑数量把未成立的想法包装成创新。
- 三个创新必须围绕同一个核心研究问题，形成清楚的因果链、递进关系或互补分工，并共同服务于一套完整方法。
- 每个创新仍须有独立的证据、可证伪假设和实验，但输入输出必须能自然衔接；不能把三个互不相关的模块拼接成论文框架。
- 最终论文必须能用“一个总方法名 + 三个统一风格的子创新名”顺畅表述；命名要简洁、含义直接、容易形成标题和缩写，不使用生硬拼词或彼此割裂的命名体系。
- 候选创新若无法解释它与另外两个创新及论文主线的关系，只能保留为独立候选或实验结果，不进入最终三创新框架。
- 三创新组合确定前必须检查整体动机、方法顺序、接口、训练目标、实验验证和论文叙事是否前后一致；局部指标提升不能替代整体逻辑成立。

## 训练与测试

- 协议固定为 `test_selected_inductive_gzsl`。
- 训练只使用 150 个 seen 类的训练图像；unseen 类图像不得进入梯度。
- test-seen/test-unseen 允许用于选择参数、epoch 和模型，但必须记录 `test_used_for_selection: true`。
- U/S/H 使用 200 类联合竞争；ZS 使用 50 个 unseen 类竞争。
- 不得把该协议描述为 blind-test 或 test-free model selection。

## 交付

- 修改代码后运行最小相关测试。
- 完成时说明改了什么、验证了什么、未验证什么和真实剩余风险。
