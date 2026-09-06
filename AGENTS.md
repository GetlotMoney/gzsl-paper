# gzsl-paper 协作规则

## 1. 当前主线

- 当前目标是研究一套全新的、原生三分支 GZSL 框架。
- 三条分支必须地位对等：各自形成不可替代的类别证据，彼此在最终 logits 前交换信息，并在同一次前向与反向传播中协同优化。
- 不把“主分支 + 辅助分支”、简单 logits 加权、额外 Gate/Head/loss 或约束包装成三分支。
- 当前只确定研究动机和终极目的；第三分支、通信机制、公式、代码身份和实验条件尚未确定，不能写成已经成立的方法。
- 新框架不以 PSR/ER 为强制基座。旧框架中有证据支持的部分可以迁移，但必须服务于三支对等协同，不能直接堆叠旧模块。
- `tmp/worktrees/v8-triadic-native` 等旧原型只算工程探索，不是正式框架，也不能因 smoke 跑通就视为正式实验。
- `论文框架仓库/` 只保存 owner 明确要求封装的交付快照；日常研究不得自动同步或覆盖其中内容。

## 2. 做事原则

- 先质疑问题是否真实，再跑最小闭环；一次实验只回答一个问题，一次修改只引入一个关键变量。
- 事实优先。假设、已实现、已运行和已证明必须分开写，不为完成任务假装成功。
- 能复用就不新增，能删除就不堆叠；只有能防止当前具体错误的流程、文件、依赖或 Agent 才能增加。
- 用户确认“开始、继续、批准”后直接执行已确认事项，不重复规划。
- 不确定代码起点、数据边界、实验变量或评估口径时必须明确说明，不能猜。

## 3. Git 与代码身份

- `main` 只接纳正式代码和轻量账本，禁止 force-push；未接纳候选不得混入 `main`。
- 正式框架的 `framework/vX` 与 Tag `vX` 必须固定在同一 commit，发布后不移动。
- 每个实验由 owner 指定准确父 commit。未接纳或失败候选不得成为下一候选的代码基线；失败结果保留，但失败代码不合入正式框架。
- 新研究主线实验分支使用 `exp/<track_id>/<kind>/<id>-<slug>`；历史 `exp/vX/...` 与编号保留。临时实现分支使用 `codex/<slug>`。
- 历史 `experiments/v1/` 至当前版本账本只读保留，不删除、复制、重编号或改写；跨版本证据直接引用原路径和原身份。
- pre-run commit 只保存代码、配置和计划；结果只能在真实运行后写入 post-run 账本。Git 不保存数据、cache、checkpoint、原始大日志或密钥。
- 只有 owner 明确接纳后，候选才能晋级为新正式框架。

## 4. 研究知识与创新

- 本项目独立建立研究知识，不迁移或隐式复用旧 GTPJ 的笔记、结论和编号。
- 外部论文必须重新核对原文并记录来源；旧聊天、旧笔记和 Agent 记忆不能作为论文证据。PDF 原件不提交 Git，只记录定位、摘要、URI 和哈希。
- `research/ideas/` 永久保存 Idea 的 `proposed/testing/supported/revised/rejected` 历史；同一假设持续回填，核心机制或学习信号改变时新建 Idea，旧编号不复用。
- 创新必须同时区分：`method_originality`（是否有不可还原的新机制）与 `method_maturity`（是否已在准确率、成本或通用性上形成真实优势）。
- 只增加 Gate、Head、Adapter、注意力、融合、正则、辅助 loss、更多 prompt/view 或参数变化，默认不算创新；除非有独立机制、不可还原对照和真实优势。
- 创新 Idea 至少写清：旧解法、新解法、原理差异、不可还原测试、最小可运行证据、最小证伪实验、当前真实优势和失败边界。
- 独立新框架以同骨干、同文本、同训练和评估协议的公平冻结 CLIP 基线为比较对象。准确率候选的最小证伪要求 seed7 matched fixed200 至少 `+0.5 H`；最终确认要求 3 个 seed 平均至少 `+0.5 H` 且每个 seed 不为负。若主张成本或通用性优势，H 不得低于公平比较对象，并须有对应真实证据。
- Idea、代码或单次涨点都不能替代 owner 对创新资格和正式框架的确认。

## 5. 实验与交付物

- 新研究主线在首次实际登记时使用 `experiments/<track_id>/EXPERIMENT_QUEUE.csv`，不为探索提前分配正式 V 版本；已有 Experiment 保留原路径。每个 TRY 一行，记录 Idea、准确 code commit、配置、唯一改动、seed、U/S/H/ZS、状态、决策和仓库外输出 URI。
- 只改已审配置、参数或 seed 时新增 RUN；修改公式、forward、loss、数据语义或评估语义时新建 Experiment。
- 每个正式 RUN 必须绑定 code commit、完整配置、seed、数据/资产身份、U/S/H/ZS、评估历史、日志和模型 URI，失败结果也要保留。
- 正式实验的最小文件是 `EXPERIMENT.yaml`、`configs/RUN-xxx.yaml`、`PARAMETER_MATRIX.csv` 和 `result.md`；其他文件只在能防止当前错误时增加。
- module、forward、loss、数据流或评估语义改变时，实验必须有可直接打开的 `framework_diagram.html`，标明准确 commit、输入输出、关键数据流、loss、logits/metric 出口和评估边界。纯参数或文档变化不复制框架图。

## 6. 必要的双 Agent 审查

### 新创新 Idea

- 新 Idea 准备成为 innovation、论文核心候选、进入活跃主线或创建实验分支前，必须由两名 Agent 独立审查同一准确草稿及直接证据。
- Agent A 主审第一性原理、机制自洽、数据边界和最小证伪；Agent B 主审近期工作、不可还原性、泄漏、控制实验、统计门和复杂度。
- 两者先独立给出完整 `P0/P1/P2`、`pass/revise/reject` 和最强反例，再交换完整原文并各自回应。主 Agent 只做一次集中修订，不逐问题串行打补丁。
- 最多 3 轮。只有双方对同一版本均为 `P0=0/P1=0/P2=0` 且明确 `pass`，才能称为“双 Agent 对抗通过”；通过仍不能替代 owner 确认和真实实验。

### 代码进入服务器前

- 新增或修改 module、forward、loss、数据流、资产生成、训练器或评估语义后，服务器 smoke 和正式 RUN 前必须由两名 Agent 只读审查同一冻结 commit。
- 两者先独立完成完整 `P0/P1/P2`，再交换一次完整清单并回应。只有双方最终均为 `P0=0/P1=0` 才能启动；P2 记录但不阻断。
- 若存在 P0/P1，先汇总全部缺陷，再集中修复并冻结新 commit；旧签字失效，只复核受影响范围和合同。
- 审查从一开始并行，默认 15 分钟内完成并启动真实 RUN。只保留准确 diff、数据/test 边界、必要 manifest 与一次真实 GPU micro-batch；不重复计算未变化的 SHA、测试或父结果。确因正确性超过 15 分钟时，说明不可省原因和剩余步骤后继续。
- 当前 Experiment 记录审查 commit、双方清单与回应、修复身份和最终结论。代码、配置、资产 manifest 或评估语义改变时签字失效；纯账本回填不重复审查。

## 7. 训练与评估协议

- 论文主协议为 `chen_shiming_code_aligned_test_selected_gzsl`：CUB 使用 `trainval_loc` 的 150 类/7,057 张图像训练，并用 official test 反复选择整模型最大 H。
- 固定披露：`test_used_for_selection: true`、`unseen_images_used_for_gradient: false`、`strict_blind_claim: false`；不得描述为 validation-first 或 blind test。
- TransZero 代码对齐采样固定为 batch 50、200 名义 epoch、28,228 次更新、每步独立 `randperm(7057)[:50]`、每 141 步 official 评估。若新实验改变这些条件，必须明确登记为新的比较条件。
- 只有整套模型 H 参与 best 选择；分阶段分别查看 test 时必须标记 `nested_official_test_selection: true`。
- CLIP 与 ResNet-101 是不同视觉特征设置，只能与相同骨干、checkpoint、预处理和缓存协议的基线直接比较。
- GZSL 的 U/S/H 使用全部 seen+unseen 类联合竞争；ZS 只在 unseen 类内竞争。所有主结果保存完整 official 评估历史，并从同一 best-H checkpoint 报告 U/S/H/ZS；不得跨 checkpoint 拼数字。
- 同一 checkpoint 的分支关闭只能称为推理关闭诊断。正式消融必须让 Full 与 Full-minus-branch 从同一准确父条件分别重新初始化并完整训练，保持 seed、预算、优化器、评估频率和选择口径一致。

## 8. 完成标准

- 修改后只运行能直接证明结果的最小测试或真实样例。
- 完成时说明改了什么、验证了什么、未验证什么和真实剩余风险。
- 不自动删除数据，不自动 push、发布或部署。

## 9. 长期研究记录合同（2026-09-06 owner 确认）

- README 是唯一当前入口；IDEA_TREE 管问题与关系，Idea 卡管假设和演化，原 TRY/RUN 管运行事实，LEDGER_INDEX 管准确位置。不得多处复制成绩维护平行事实源。
- 主线、问题、Idea、Experiment、RUN、框架候选、正式版本分开。关系用编号和引用表达，不要求每个对象新建目录。
- Idea 跨主线和框架共用；同假设换条件仍用原卡并追加实验，旧失败边界不覆盖；核心机制或学习信号/假设改变才新建 Idea。
- 工作状态、证据状态、接纳状态分开；不机械改写旧 status。supported 不等于正式接纳，暂停不等于机制否定。
- 实验开工前明确 research_mode、owner 指定的 base_commit、reuse_scope、change_scope、comparison_ref 和唯一问题。base_commit 与 comparison_ref 分开，独立新框架允许只复用可信工程基础。
- 代码起点、实际运行 code_commit 和保存结果的 post-run commit 分开；不得用最新分支 tip 或记录快照冒充训练代码。
- 新记录只在有实际内容时创建；历史编号与原路径不修改。新编号先查跨分支记录，不能只看当前目录。
- 本次账本/GitHub整理授权不等于未来任意 push、发布或删除授权；公开同步只包含核对后的指定内容，不批量上传候选分支和未提交草稿。
