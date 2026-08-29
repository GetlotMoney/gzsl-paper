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
- “准确父commit”指新候选要比较和继承的最后一个已接纳、可复现代码身份，不是Git时间线上最近的commit，也不是上一个失败TRY的commit。只有前一候选经owner明确接纳后，它的接纳commit才可以成为下一候选父条件；否则各候选必须从owner确认的同一正式父条件独立分叉。
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
- `research/ideas/`是跨版本持续积累、可反复检索和引用的Idea总库；`proposed / testing / supported / revised / rejected`全部永久保留。提出、实施、运行、补救、跨数据集确认或得到新证据后，都必须持续回填同一Idea卡和`research/IDEA_TREE.md`，不得只写TRY而遗漏Idea。
- 每张新Idea卡必须先按“解决什么问题”填写一个`problem_category`、一个具体`problem`和可选`mechanism_tags`；`IDEA_TREE.md`按问题类别维护轻量索引，不能只按版本或编号堆放。
- 同一核心问题、假设和唯一改动未变时，一个Idea可关联多个TRY、RUN、seed、参数和数据集确认；公式、输入、学习信号、表示原语或核心可证伪假设发生变化时必须创建新`IDEA-xxx`并引用旧Idea。旧编号不复用，`rejected`历史不得改写成新的`proposed`。
- 外部论文需要重新核对原文并记录来源；旧聊天、旧笔记和 Agent 记忆不能直接作为证据。
- 正式创新实验必须绑定一个本仓库 `idea_id`，并引用支撑该假设的 `evidence_refs`。
- 创新证据不强制来自论文；可以来自论文、正式实验结果、代码观察、指标异常、第一性原理分析或明确记录的owner假设。
- 没有直接论文时可以先建立可证伪Idea并做实验；但在形成论文新颖性claim前必须检索最接近的相关工作并说明区别。
- PDF 原件不提交 Git；仓库只保存来源、定位信息、必要摘要、文件 URI/哈希和研究判断。

## 范式级创新准入

- 本项目以后只把“范式级候选”登记为`innovation`或`paper_core_innovation`。普通调参、消融、确认、工程修复和必要对照仍可执行，但不得包装为创新点。
- “范式级创新”固定定义为：改变模型**从什么新信号学习**，或改变方法所操作的**表示原语**。它必须改变问题的基本学习对象，而不是只替换已有决策流程、网络部件或分数计算方式。
- 满足下列至少一项，才可进入范式候选：
  - 学习信号来源发生变化，例如从seen类别标签扩展到合法的unseen文本自监督、概念级监督、因果干预信号或其他父框架不存在的证据；
  - 表示原语发生变化，例如从单点原型变为连续场、条件分布、可复用概念组合或其他具有不同数学对象与学习规律的表示；
  - 学习问题被重新形式化，并由新的可证伪假设、训练目标和评估合同支撑；仅改变推理顺序或决策规则不算重新形式化。
- 下列改动默认不属于范式级创新，除非能证明它同时引入了新的学习信号或表示原语：
  - 增加Gate、Head、Adapter、残差、校准项、margin、正则项或辅助loss；
  - 更换Top-K、注意力、池化、路由、重排、融合权重或MLP结构；
  - 增加更多patch、crop、view、prompt、文本源、尺度或预训练模型后继续做同一种相似度打分；
  - 只改变训练日程、采样、优化器、参数量、速度或显存；
  - 只凭`ΔH`上涨声称创新。指标成立是必要证据，但不能把模块级改动升级成范式。
- 范式级不等于复杂、大模型、多模块或新名词。一个很小的改动若真正改变监督来源或表示原语，可以是范式候选；一个很大的系统若仍在相同输入和点原型上做打分，仍然只是工程组合。
- 每张新Idea卡在进入`EXPERIMENT_QUEUE.csv`前必须额外写清：
  - `old_signal_or_primitive`：父框架依赖的旧学习信号或表示原语；
  - `new_signal_or_primitive`：候选引入的新学习信号或新表示原语；
  - `paradigm_shift`：一句话说明基本学习对象如何改变；
  - `why_not_module`：说明它为什么不是Gate、重排、聚合器、校准或同信号新loss；
  - `closest_paradigm_work`：重新核对的近期最接近原始论文及本项目的实质区别；
  - `minimal_falsification`：在正式实现前能够最快否定核心范式假设的真实实验；
  - `paper_level_claim`：若成立，论文能够提出的窄而准确的范式claim，禁止写“首次”而无系统检索证据。
- 缺少上述任一项，或`new_signal_or_primitive`仍与父框架相同，候选只能作为普通快速尝试、诊断或工程改动，不能登记为Innovation。
- owner必须在创建创新分支前明确确认该候选通过范式准入；代码实现、指标上涨、Agent审核或实验完成都不能替代owner的范式确认。
- 范式候选仍必须接受真实结果淘汰。若最小证伪不支持新信号或新原语的有效性，应立即drop，不得退化成模块调参后继续冒充范式创新。

## 三创新论文主线

- 项目目标是最终筛选并验证 `3` 个有实验支持的核心创新点；不能为了凑数量把未成立的想法包装成创新。
- 最终三个核心创新中的每一个都必须独立通过“范式级创新准入”；三个模块级技巧即使组合后涨点，也不能作为三创新论文主线。
- 三个创新必须围绕同一个核心研究问题，形成清楚的因果链、递进关系或互补分工，并共同服务于一套完整方法。
- 每个创新仍须有独立的证据、可证伪假设和实验，但输入输出必须能自然衔接；不能把三个互不相关的模块拼接成论文框架。
- 最终论文必须能用“一个总方法名 + 三个统一风格的子创新名”顺畅表述；命名要简洁、含义直接、容易形成标题和缩写，不使用生硬拼词或彼此割裂的命名体系。
- 候选创新若无法解释它与另外两个创新及论文主线的关系，只能保留为独立候选或实验结果，不进入最终三创新框架。
- 三创新组合确定前必须检查整体动机、方法顺序、接口、训练目标、实验验证和论文叙事是否前后一致；局部指标提升不能替代整体逻辑成立。
## 一轮双Agent交叉审查

- 凡新增或修改module、forward、loss、数据流、资产生成、训练器或评估语义，服务器smoke、训练和正式运行前必须完成一轮双Agent交叉审查；普通机器测试不能替代。
- 两名独立Agent同时只读审查同一个冻结commit。交流前各自必须先完成独立检查和完整`P0/P1/P2`初始清单，主动尝试证伪公式、父条件、数据边界、指标、关闭路径和可复现性；不得一开始互相抄结论。
- 两名Agent完成独立检查后必须相互交换一次完整清单，并各自回复一次对另一清单的补充、异议和最终判断。只要求这一轮交叉交流，不再串行重复第二轮完整审核。
- 只有两名Agent最终都明确报告`P0=0 / P1=0`并写出“双Agent交叉审查通过”，代码才能启动；任一Agent证据不足、不同意或仍有P0/P1均视为不通过。
- 若发现P0/P1，两名Agent仍须审完、完成一次交叉交流并汇总全部缺陷；主Agent一次性集中修复。修复后形成新的冻结commit，并重新执行一轮完整双Agent交叉审查；旧签字全部失效，不要求另找所谓“第二轮Agent”。
- 当前Experiment必须记录两名Agent审查的同一commit、双方初始清单、一次交叉交流、修复commit、最终结论和证据URI。签字身份固定为`最终RUN commit + 审查声明路径的tree hash + config SHA + 资产manifest SHA + 环境/GPU fingerprint`；其中任一语义身份变化则签字失效。

### 十分钟完整审核目标

- 双Agent审核开始前，主Agent必须一次性准备准确diff、相关测试、一次本地完整测试、资产/配置确定性校验和服务器真实micro-batch证据；Agent不得各自重复跑整仓测试或重新计算同一批全量SHA。
- 正确性和完整性优先于速度；审核必须覆盖规定项并形成完整结论，禁止为了满足时长跳过检查、缩小范围或降低通过标准。
- 在共享证据齐全且没有P0/P1时，力争10分钟内完成独立检查、一次交叉交流和最终双签；该时长是优化目标，不是强制截止线，正确性始终优先。
- 任一Agent可在系统或仓库外临时目录重放不产生正式结果的只读/可清理micro-batch；双方复用同一真实证据，不跨GPU机械重复。正式server smoke、训练、资产发布和RUN必须等双Agent最终通过。
- 超过10分钟时，Agent必须立即汇报剩余检查项、变慢原因和当前发现，同时继续审核直到完整结论；不能因到时直接签字或停止。`证据不足`视为不通过，补齐证据后继续。发现P0/P1时速度目标自动让位于修复和复核，修复完成后以新代码身份重新审核。
- 纯队列、结果回填和文档提交在签字身份不变时不重复启动双Agent代码审查。纯配置仅限已审schema内参数值且不启用新forward/loss/objective/评估路径；否则仍是语义变化。生成后资产使用manifest、SHA、shape、dtype、数量、关闭路径和结果文件校验。
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
