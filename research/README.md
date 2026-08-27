# 研究知识与实验闭环

## 目标

本目录只解决一个问题：让每个新实验都能追溯到明确证据来源和一个可证伪的研究假设，并让实验结果反过来更新该假设。证据来源可以是论文，也可以是本地实验、代码观察、指标异常、第一性原理分析或owner明确提出的假设。

owner已选择`FRAMEWORK-V2`作为论文主框架。当前研究目标是在相同数据、划分和`test_selected_inductive_gzsl`评估口径下，使H从V2正式单seed基线`74.023182%`提高到至少`77.023182%`，即提高`3.00`个百分点。

论文方法目标是从证据和真实实验中筛选出 `3` 个相互关联、能够共同构成完整方法的核心创新点。数量是筛选目标，不是造模块指标；证据不足或实验不成立时不得为了凑满三个而降低标准。

固定闭环：

```text
真实问题或证据 → 一张Idea卡 → 实验清单快速尝试 → 成功后正式Experiment
```

新模块从准备到判定必须逐项执行[`docs/MODULE_WORKFLOW_CHECKLIST.md`](../docs/MODULE_WORKFLOW_CHECKLIST.md)。准备研究的模块登记在`IDEA_TREE.md`和对应Idea卡；具备代码与配置、准备真实运行时再写入当前框架的`EXPERIMENT_QUEUE.csv`。

本仓库从空白研究知识层开始。旧 GTPJ 的论文笔记、idea tree、研究结论和编号不迁移、不引用，也不能通过聊天记忆隐式恢复。需要使用同一篇论文时，必须重新核对原文并在本仓库重新登记。

例外：owner于2026-08-22明确授权直接迁移H1相关旧实验的轻量证据。该授权只覆盖`INNOVATION-024`、`ABLATION-014`、`TUNE-005`和`TUNE-006`，不自动扩大到其他旧实验、Idea或论文笔记。

## 最小结构

有第一份真实内容时再创建对应文件，不预建空记录：

```text
research/
├─ IDEA_TREE.md
├─ papers/
│  └─ PAPER-001_<slug>.md
└─ ideas/
   └─ IDEA-001_<slug>.md
```

编号分别从 `PAPER-001` 和 `IDEA-001` 开始，按创建顺序递增，不复用已使用编号。

## 本地知识库与 RAG

论文卡片只负责记录论文来源，不是创新开工硬门。用于核对的PDF、解析全文、批量图片和检索索引保存在仓库外；登记论文时固定所核对PDF的仓库外绝对URI和SHA256。

本地PDF与正式出版页面、DOI或arXiv都可以作为论文来源，但必须重新核对原文。RAG只是一层可选检索工具，不能把模型回答直接当证据。没有论文时，Idea可以直接引用本地实验、代码或指标记录。

## 论文记录

每篇论文只建立一个记录，至少写清：

- `paper_id`、标题、作者、年份和会议或期刊；
- DOI、arXiv 或出版社等可复查来源；
- `source_checked_at`；
- 所核对 PDF 的仓库外绝对 URI 和 SHA256；
- 与当前研究的简短关系，以及论文没有直接证明的本项目改动；
- 页码、章节、表格、公式位置和`PAPER-xxx-Cxx`细粒度证据均为可选，只在后续论文claim需要严格逐条审计时补充；
- 自己的推断必须单独标记，不能写成论文事实。

PDF 原件、批量图片和大体积解析缓存不进入 Git。

## 创意树

`IDEA_TREE.md` 是研究问题到 Idea、实验和结果的轻量索引，不复制论文证据或实验正文。每个 Idea 节点必须链接对应 `IDEA-xxx` 卡片；没有证据和可证伪假设时不得创建正式 Idea 节点。

创意树从本项目重新建立，不迁移旧 GTPJ idea tree。节点只表达以下关系：

```text
研究目标 → 问题/机制 → IDEA-xxx → Experiment → 结果与状态
```

基线完成前，创意树只保留研究目标和 `pending_baseline` 状态，不提前编造候选改动。

## Idea 记录

每个Idea只需一张卡，并且必须能被实验否定。最少写清：

```yaml
idea_id: IDEA-xxx
source_type: paper | local_observation | experiment_result | code_analysis | first_principles | owner_hypothesis
evidence_refs:
  - <至少一个可追溯来源>
base_commit: <准确commit>
problem: <真实问题>
hypothesis: <可证伪假设>
core_change: <唯一核心改动>
success_condition: <成立条件>
failure_condition: <失败条件>
status: proposed | testing | supported | revised | rejected
```

实验和结果路径发生后再补；Idea完成后先进入当前框架的`EXPERIMENT_QUEUE.csv`，只有`promote`候选才建立正式Experiment。论文角色、命名候选和与其他创新的接口只在进入最终三创新组合时补，不作为Idea开工门槛。

证据不足时可以保存为 `proposed`，但不能直接宣称创新成立。

## 三创新组合与命名

最终入选的三个创新必须同时满足：

1. 共同回答一个核心研究问题，而不是分别解决三个无关问题；
2. 方法关系能够用一条连续逻辑解释，例如“表示建立 → 条件适配 → 结构校准”，具体角色由真实 Idea 决定，不能提前编造；
3. 前一创新的输出或结论能自然支撑后一创新，或者三者具有明确的互补分工；
4. 每个创新都有独立消融或控制实验，同时整套方法有组合结果；
5. 三者能够统一在一个总方法名下，子名称风格一致、含义直接、容易写进标题、摘要和框架图。

以下情况视为框架割裂，不能进入最终三创新组合：

- 只能用“另外再加一个模块”解释其存在；
- 与其他创新没有共享问题、数据流或机制关系；
- 必须使用完全不同的术语体系才能描述；
- 单独涨点但破坏整体接口、训练逻辑或论文叙事；
- 名称只能靠生硬缩写、堆词或事后包装得到。

创意树可以保留超过三个候选，但只有状态、证据、实验和组合逻辑都通过的三个节点才能标记为 `paper_core_innovation`。如果不足三个，应继续寻找和验证，不得把 `proposed` 或失败节点补位。

## 进入实验

当前三项核心创新的允许与禁止claim见[`docs/CORE_INNOVATION_CLAIM_BOUNDARIES.md`](../docs/CORE_INNOVATION_CLAIM_BOUNDARIES.md)。

正式 Innovation 实验的 `EXPERIMENT.yaml` 必须包含：

```yaml
idea_id: IDEA-xxx
evidence_refs:
  - <PAPER、RUN、代码或本地观察引用>
base_commit: <准确 Git commit>
```

正式实验不以“先找到论文”为硬门。没有直接论文时，只要本地证据清楚、假设可证伪、base commit和成功/失败条件完整，就可以开始实验。论文检索最迟必须在对外声称“新颖”或开始正式论文写作前完成。

参数、seed、epoch 或预注册开关变化继续作为同一 Experiment 的新 RUN；模块公式、输入、forward、loss、数据边界或评估语义变化必须新建 Experiment。具体执行规则继续以 `docs/EXPERIMENT_PROTOCOL.md` 为准。

## 结果回填

实验完成后必须把结果路径写回对应 idea，并根据真实证据更新状态：

- `supported`：结果达到预先写明的成立条件；
- `revised`：原假设不完整，但结果支持一个更窄的新解释；
- `rejected`：达到失败条件或关键机制没有得到支持。

失败结果不能删除，也不能事后修改成立条件来制造成功。新的论文或实验结果可以产生下一个 idea，但必须建立新编号和新的可证伪假设。
