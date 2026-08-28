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
## 最小两轮代码审核

- 只有新增或修改`module / forward / loss / 数据流 / 训练器 / 评估语义`才启动两轮审核；纯配置、队列、结果和文档不重复审核。
- 审核前一次性准备且只准备：准确RUN commit/config、相关专项测试、一次完整测试、一次真实GPU micro-batch、正常checkpoint保存与恢复。相同证据两名Agent共用。
- Round 1只审公式、初始化、梯度归属、数据边界、Full/Off指标和关闭路径；Round 2只审准确身份、服务器clean、真实GPU出口和正常resume。两名Agent并行预读，Round 1通过后Round 2签字。
- 正常checkpoint只需证明`save → weights_only load → resume后下一batch/LR一致`。除非发生真实损坏或安全任务，不做恶意篡改、fuzz、逐张量hash链、optimizer/RNG字节攻击或任意字段穷举。
- 每种不同objective只跑一次GPU micro-batch；不跨两张GPU重复，不重复完整测试，不重复大资产SHA，不为证据另建文档层级或审核线程。
- 发现P0/P1后，两名Agent先各自审完既定范围并一次性交齐清单；主Agent只做一批集中修复。复核只审最终diff和受影响合同，未变化证据全部复用。
- 两轮在证据齐全时目标5–10分钟。命令或自造入口连续失败两次必须停止，改走现有训练/测试入口，禁止继续换壳重试。
- 审核记录只写现有Experiment的`REVIEW.md`：准确commit、config SHA、证据URI、P0/P1结论和“第2轮通过”。不新增receipt、证据页、状态机或额外目录。
- 只有两轮均无P0/P1才能启动正式RUN；修复后语义commit变化才使签字失效。Git提交不得在自身文件中预写自身SHA，纯账本提交不得冒充代码变化。

## 正式RUN运行闭环

- 正式RUN启动后1分钟内必须确认：真实Python子进程存在、占用预期GPU、服务器HEAD与RUN commit一致、config SHA一致、checkpoint已出现且`update>0`；任一项失败立即按真实错误处理，不能把launcher PID当训练已启动。
- RUN结束时必须确认：进程已退出、`metrics.json`与完整评估历史存在、`stop_reason=completed_fixed_150`、`total_updates=21171`、`history_length=152`；缺任一项均不得报“完成”。
- 双卡队列必须逐卡接力：前一RUN完成后立即检查该卡的下一RUN；若下一RUN输出目录不存在且GPU空闲，直接启动并再次执行启动后1分钟确认。不得只看前一组结果而漏启动后一组。
- 正式结果必须确认`loaded_training_checkpoints=[]`，并验证RUN commit、config SHA、资产身份以及同一best-H checkpoint的U/S/H/ZS和Full/Off指标一致；不得跨checkpoint或跨commit拼接结果。
- 上述运行闭环属于轻量确定性检查，不新增Agent、测试、审核轮次、守护进程、状态机或文档层级；只在现有RUN日志、checkpoint和结果文件上核对。

## 训练与测试

- 协议固定为 `test_selected_inductive_gzsl`。
- 训练只使用 150 个 seen 类的训练图像；unseen 类图像不得进入梯度。
- test-seen/test-unseen 允许用于选择参数、epoch 和模型，但必须记录 `test_used_for_selection: true`。
- owner选择的论文主协议为`chen_shiming_code_aligned_test_selected_gzsl`：使用`trainval_loc`的150类/7,057张图像训练，并反复评估official test选择整模型最大H。
- Chen-style固定披露`test_used_for_selection: true / unseen_images_used_for_gradient: false / strict_blind_claim: false`，不得描述为Xian validation-first或blind-test。
- 首个Chen-style主实验固定端到端联合训练，TG-VPR、TST/NTR、CCGR和可选属性残差不得分别使用test选择checkpoint；只有整套模型H参与best选择。
- 分阶段训练允许作为后续新Experiment，但必须记录冻结边界；若每阶段分别看test，额外标记`nested_official_test_selection: true`。
- 一段式训练定义：除冻结CLIP图像/文本资产外，禁止加载训练好的TG或模块checkpoint；所有启用的TG、GTD、MMT、BD、视觉模块从本RUN初始状态开始，并从update 1起在同一连续训练时间线同步更新。`warmup`只表示学习率预热，不表示加载权重。
- 当前V3 owner协议固定batch 50、150名义epoch、21,171次更新、每步独立`randperm(7057)[:50]`、每141步official评估。TG-only、累计条件和单模块移除均使用相同总预算，不再自动升级为200轮。
- CLIP与经典ResNet-101属于不同视觉特征设置，必须记录准确checkpoint、预处理和缓存生成脚本，并只与相同骨干基线直接比较。
- 现有validation-first结果继续保留为严格协议对照，不删除、不覆盖。
- U/S/H 使用 200 类联合竞争；ZS 使用 50 个 unseen 类竞争。
- 不得把该协议描述为 blind-test 或 test-free model selection。

## 交付

- 修改代码后运行最小相关测试。
- 完成时说明改了什么、验证了什么、未验证什么和真实剩余风险。
