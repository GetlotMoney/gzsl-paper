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
- 探索阶段统一写入当前框架的`EXPERIMENT_QUEUE.csv`；每个快速尝试只占一行，不建立独立目录。
- 快速尝试至少记录idea、准确code commit、config、唯一改动、seed、U/S/H/ZS、状态、决策和仓库外输出URI。
- 失败尝试标记`drop`并保留一行；有效候选标记`keep`；只有值得详细验证的候选标记`promote`后才建立正式Experiment目录。
- 每个正式`experiments/vX/`必须固定包含`tune/`、`ablation/`、`innovation/`和`confirmation/`四类目录，并各有`INDEX.md`；没有实验时索引明确写“当前无实验”。
- 只改配置、参数、seed 或预注册控制条件：新增 RUN。
- 修改公式、模块输入、forward、loss、数据语义或评估语义：新建 Experiment。
- 每个 RUN 必须绑定准确 code commit、完整配置快照、seed、数据身份、U/S/H/ZS、日志和模型 URI。
- 新实验默认只要求`EXPERIMENT.yaml`、`configs/RUN-xxx.yaml`、`PARAMETER_MATRIX.csv`和`result.md`四类文件。
- `PARAMETER_MATRIX.md`、逐RUN evidence页、README、implementation和module_source均为按需文件，不再作为普通实验默认门槛。
- 只有代码结构、模块、forward、loss、数据流或评估语义改变时，才额外要求实验级`framework_diagram.html`。
- pre-run commit 只保存代码、配置和计划；post-run commit 只保存已经发生的结果。
- 多个快速尝试可以共用一次结果回填提交，不要求每个TRY单独写post-run文档。
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
- 创新证据不强制来自论文；可以来自论文、正式实验结果、代码观察、指标异常、第一性原理分析或明确记录的owner假设。
- 没有直接论文时可以先建立可证伪Idea并做实验；但在形成论文新颖性claim前必须检索最接近的相关工作并说明区别。
- PDF 原件不提交 Git；仓库只保存来源、定位信息、必要摘要、文件 URI/哈希和研究判断。

## 三创新论文主线

- 项目目标是最终筛选并验证 `3` 个有实验支持的核心创新点；不能为了凑数量把未成立的想法包装成创新。
- 三个创新必须围绕同一个核心研究问题，形成清楚的因果链、递进关系或互补分工，并共同服务于一套完整方法。
- 每个创新仍须有独立的证据、可证伪假设和实验，但输入输出必须能自然衔接；不能把三个互不相关的模块拼接成论文框架。
- 最终论文必须能用“一个总方法名 + 三个统一风格的子创新名”顺畅表述；命名要简洁、含义直接、容易形成标题和缩写，不使用生硬拼词或彼此割裂的命名体系。
- 候选创新若无法解释它与另外两个创新及论文主线的关系，只能保留为独立候选或实验结果，不进入最终三创新框架。
- 三创新组合确定前必须检查整体动机、方法顺序、接口、训练目标、实验验证和论文叙事是否前后一致；局部指标提升不能替代整体逻辑成立。
## 代码两轮对抗审查

- 凡新增或修改module、forward、loss、数据流、资产生成、训练器或评估语义，服务器smoke、训练和正式运行前必须完成两轮独立Agent对抗审查；普通机器测试不能替代。
- 第一轮Agent只读审查预冻结代码，主动尝试证伪公式、父条件、数据边界、指标、关闭路径和可复现性；所有P0/P1必须修复并以测试证明关闭。
- 第二轮必须由另一名独立Agent审查准确post-fix commit、本地与服务器clean状态及真实资产/结果；只有明确报告无P0/P1并写出“第2轮通过”，代码才能冻结和启动。
- 第二轮若发现P0/P1，修复后原签字失效，必须由新的独立Agent对新的准确commit重新执行第二轮；不得沿用旧commit结论。
- 当前Experiment必须记录两轮审查对象commit、发现、修复commit、最终结论和证据URI。签字身份固定为`最终RUN commit + 审查声明路径的tree hash + config SHA + 资产manifest SHA + 环境/GPU fingerprint`；其中任一语义身份变化则签字失效。

### 十分钟完整审核目标

- 两轮审核开始前，主Agent必须一次性准备准确diff、相关测试、一次本地完整测试、资产/配置确定性校验和服务器真实micro-batch证据；Agent不得各自重复跑整仓测试或重新计算同一批全量SHA。
- 正确性和完整性优先于速度；审核必须覆盖规定项并形成完整结论，禁止为了满足时长跳过检查、缩小范围或降低通过标准。
- 在共享证据齐全且没有P0/P1时，力争10分钟内完成两轮完整审核；该时长是优化目标，不是强制截止线，正确性始终优先。
- 两名Agent可以并行预读同一冻结代码身份；Round 2只有在Round 1确认无P0/P1后才能正式签字。Round 1发现P0/P1时，Round 2本轮签字资格作废，但仍须审完已分配范围并把其余问题纳入同一次汇总。
- Round 2可在系统或仓库外临时目录执行不产生正式结果的只读/可清理micro-batch审核；正式server smoke、训练、资产发布和RUN仍必须等第2轮通过。
- 超过10分钟时，Agent必须立即汇报剩余检查项、变慢原因和当前发现，同时继续审核直到完整结论；不能因到时直接签字或停止。`证据不足`视为不通过，补齐证据后继续。发现P0/P1时速度目标自动让位于修复和复核，修复完成后以新代码身份重新审核。
- 纯队列、结果回填和文档提交在签字身份不变时不重复启动两轮代码Agent。纯配置仅限已审schema内参数值且不启用新forward/loss/objective/评估路径；否则仍是语义变化。生成后资产使用manifest、SHA、shape、dtype、数量、关闭路径和结果文件校验。
- Git commit不能在自身文件中预写自身SHA；pre-run队列可在冻结commit产生后由后续账本commit绑定该运行commit，正式RUN必须在记录的clean commit执行并在结果中回写实际commit。纯账本提交不得冒充代码变化。

### 一次性缺陷汇总与并行复核

- 审核开始后冻结一个准确代码身份；主Agent先把公式、训练、评估、数据/资产、GPU运行、checkpoint和输出合同拆成清楚的审查矩阵，再把能独立检查的范围并行交给多个只读Agent。
- Agent发现首个P0/P1后不得提前结束自己的分工；必须继续审完已分配范围，一次性返回完整`P0/P1/P2`清单。除非存在安全或破坏性风险，不允许边审边改。
- 主Agent必须等待所有并行Agent到达汇合点，去重并合并完整清单，然后只制作一个集中修复补丁；禁止“发现一个、修一个、重新跑全流程”的串行循环。
- 修复后生成一个新的冻结代码身份；多Agent并行复核同一最终diff和受影响合同。未变化的代码树、完整测试、资产manifest/SHA和父指标证据按审查声明路径的tree hash复用，不重复整仓测试、全量大文件哈希或双卡完整预载。
- 每个不同的objective/forward/loss路径至少执行一次真实GPU micro-batch；相同路径不跨GPU重复。共享闭环必须覆盖梯度、ZS、动态停止和checkpoint roundtrip，第二GPU只验证设备路径及其特有差异。
- 共享证据包必须绑定代码commit与相关tree hash，至少包含准确diff、测试摘要、配置SHA、资产身份、micro-batch结果、checkpoint恢复和正式输出不存在证明；所有Agent引用同一份证据。
- 若任一Agent发现P0/P1，本轮不得签字；“一次集中修复”指每个审核周期一个批次。复核若发现新的P0/P1，所有Agent仍先审完分工并汇总，再进入下一批集中修复；不得为了只修一次而错误签字。

## 训练与测试

- owner选择的论文主协议为`chen_shiming_code_aligned_test_selected_gzsl`：使用`trainval_loc`的150类/7,057张图像训练，并反复评估official test选择整模型最大H。
- Chen-style固定披露`test_used_for_selection: true / unseen_images_used_for_gradient: false / strict_blind_claim: false`，不得描述为Xian validation-first或blind-test。
- 首个Chen-style主实验固定端到端联合训练，TG-VPR、TST/NTR、CCGR和可选属性残差不得分别使用test选择checkpoint；只有整套模型H参与best选择。
- 分阶段训练允许作为后续新Experiment，但必须记录冻结边界；若每阶段分别看test，额外标记`nested_official_test_selection: true`。
- TransZero代码对齐采样固定为batch 50、200名义epoch、28,228次更新、每步独立`randperm(7057)[:50]`、每141步official评估。
- V3候选探索允许使用预注册动态筛选，最多150名义epoch并在固定里程碑停止；它只负责淘汰候选。所有胜出累计条件与最终单模块移除证据仍必须固定200名义epoch。
- CLIP与经典ResNet-101属于不同视觉特征设置，必须记录准确checkpoint、预处理和缓存生成脚本，并只与相同骨干基线直接比较。
- 现有validation-first结果继续保留为严格协议对照，不删除、不覆盖。
- U/S/H 使用 200 类联合竞争；ZS 使用 50 个 unseen 类竞争。
- 所有Chen-style结果必须保存完整official评估历史、best-H对应同一checkpoint的U/S/H/ZS及独立best-ZS观察，不能跨checkpoint拼接数字。

## 交付

- 修改代码后运行最小相关测试。
- 完成时说明改了什么、验证了什么、未验证什么和真实剩余风险。
