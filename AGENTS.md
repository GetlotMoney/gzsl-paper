# gzsl-paper 协作规则

## 沟通与范围

- 默认中文，先讲结论。
- 每次只做一个明确交付物，不顺手扩张流程。
- 不确定的代码起点、实验变量或评估口径必须显式说明，不能猜。

## Git 管理

- `main` 是当前已接纳代码和轻量账本的默认分支，禁止 force-push。
- 公共工作流规范、已接纳代码和已接纳轻量账本进入`main`；未接纳候选不得借文档合并混入`main`。
- `framework/vX` 与 Tag `vX` 固定在同一正式框架 commit，均不移动。
- 每个新分支必须保留此前全部`experiments/v1/`至`experiments/vX-1/`账本作为只读历史；不得因切换版本删除、移动、复制、重编号或改写旧版本账本。
- 当前版本的新TRY只写入`experiments/vX/EXPERIMENT_QUEUE.csv`；跨版本证据直接引用原路径、原RUN和原commit，不复制成当前版本结果。
- 失败候选分支只保留代码与实验留痕，不作为下一候选的代码基线；继续寻找同版本新模块时，必须由owner确认准确父commit，并从该commit新建独立`exp/vX/<kind>/<module>`分支，禁止在失败候选代码上继续堆叠。
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

## 代码两轮对抗审查

- 凡新增或修改module、forward、loss、数据流、资产生成、训练器或评估语义，服务器smoke、训练和正式运行前必须完成两轮独立Agent对抗审查；普通机器测试不能替代。
- 第一轮Agent只读审查预冻结代码，主动尝试证伪公式、父条件、数据边界、指标、关闭路径和可复现性；所有P0/P1必须修复并以测试证明关闭。
- 第二轮必须由另一名独立Agent审查准确post-fix commit、本地与服务器clean状态及真实资产/结果；只有明确报告无P0/P1并写出“第2轮通过”，代码才能冻结和启动。
- 第二轮若发现P0/P1，修复后原签字失效，必须由新的独立Agent对新的准确commit重新执行第二轮；不得沿用旧commit结论。
- 当前Experiment必须记录两轮审查对象commit、发现、修复commit、最终结论和证据URI；审查后相关代码树继续变化则签字自动失效。

### 五分钟无缺陷审核窗口

- 两轮审核开始前，主Agent必须一次性准备准确diff、相关测试、一次本地完整测试、资产/配置确定性校验和服务器真实micro-batch证据；Agent不得各自重复跑整仓测试或重新计算同一批全量SHA。
- 无P0/P1的正常审核窗口最多5分钟：Round 1目标2分钟，只审公式、父条件、数据/指标边界、关闭路径和复现性；Round 2目标3分钟，只审准确代码身份、本地/服务器clean状态并重放一个真实micro-batch。
- 两名Agent可以并行预读同一冻结代码身份；Round 2只有在Round 1确认无P0/P1后才能正式签字。Round 1发现P0/P1时，并行预读立即作废。
- Round 2可在系统或仓库外临时目录执行不产生正式结果的只读/可清理micro-batch审核；正式server smoke、训练、资产发布和RUN仍必须等第2轮通过。
- 每个Agent到时必须返回`通过`、`P0/P1发现`或`证据不足`之一，禁止无限等待；`证据不足`视为不通过。发现P0/P1后的修复时间不计入原审核窗口，修复完成后以新代码身份重新开始一个5分钟窗口。
- 纯配置、生成后资产、实验队列、结果回填和文档提交不重复启动两轮代码Agent；它们使用schema、SHA、shape、dtype、数量、关闭路径和结果文件校验。只有相关代码树改变才使代码签字失效。
- Git commit不能在自身文件中预写自身SHA；pre-run队列可在冻结commit产生后由后续账本commit绑定该运行commit，正式RUN必须在记录的clean commit执行并在结果中回写实际commit。纯账本提交不得冒充代码变化。

## 训练与测试

- 协议固定为 `test_selected_inductive_gzsl`。
- 训练只使用 150 个 seen 类的训练图像；unseen 类图像不得进入梯度。
- test-seen/test-unseen 允许用于选择参数、epoch 和模型，但必须记录 `test_used_for_selection: true`。
- U/S/H 使用 200 类联合竞争；ZS 使用 50 个 unseen 类竞争。
- 不得把该协议描述为 blind-test 或 test-free model selection。

## 交付

- 修改代码后运行最小相关测试。
- 完成时说明改了什么、验证了什么、未验证什么和真实剩余风险。
