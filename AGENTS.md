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

## 训练与测试

- 协议固定为 `test_selected_inductive_gzsl`。
- 训练只使用 150 个 seen 类的训练图像；unseen 类图像不得进入梯度。
- test-seen/test-unseen 允许用于选择参数、epoch 和模型，但必须记录 `test_used_for_selection: true`。
- U/S/H 使用 200 类联合竞争；ZS 使用 50 个 unseen 类竞争。
- 不得把该协议描述为 blind-test 或 test-free model selection。

## 交付

- 修改代码后运行最小相关测试。
- 完成时说明改了什么、验证了什么、未验证什么和真实剩余风险。
