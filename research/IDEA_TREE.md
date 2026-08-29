# GZSL 研究主线索引

本文件只回答“每个版本是什么、哪些 Idea 在哪里验证、现在走到哪一步”。公式、长结果和运行证据分别保存在 Idea 卡与实验账本中，不在这里复制。旧版V2长树已原样保存在 [`archive/IDEA_TREE_V2_LEGACY.md`](archive/IDEA_TREE_V2_LEGACY.md)。

## 身份关系

```text
研究问题 → IDEA-xxx → Vx-TRY-xxx → 正式Experiment → owner接纳 → FRAMEWORK-Vy
```

- Idea：可证伪研究假设，卡片位于 [`research/ideas/`](ideas/)。
- TRY：快速真实尝试，唯一状态源是对应版本的 `EXPERIMENT_QUEUE.csv`。
- Experiment：值得详细验证后建立的正式目录。
- Framework：owner接纳并冻结的正式代码身份，对应 `model/frameworks/vX/`、`framework/vX`与Tag `vX`。

## 按问题分类的初步索引

这里先索引正式主线和当前仍有直接复用价值的关键Idea；V2完整历史继续由对应Idea卡、V2队列和旧树保存。历史Idea在被重新检索或继续实验时补分类，不根据文件名机械猜测。

| 问题类别 | 要解决的问题 | 当前关键Idea |
|---|---|---|
| `semantic_representation` | 类别语义和原型如何形成更有结构的表示 | [`IDEA-001 / TG-VPR-H1`](ideas/IDEA-001_tg_vpr_h1.md) |
| `cross_class_transfer` | seen知识如何可靠迁移到unseen原型 | [`IDEA-005 / TST`](ideas/IDEA-005_tst.md)、[`IDEA-146 / GTD-TST`](ideas/IDEA-146_gtd_tst.md) |
| `visual_grounding` | 实例级局部视觉证据能否支持或反驳类别语义 | [`IDEA-133`](ideas/IDEA-133_visual_evidence_learning.md)、[`IDEA-158 / GAVE`](ideas/IDEA-158_gave.md)、[`IDEA-159 / RGT`](ideas/IDEA-159_rgt.md)、[`IDEA-160 / full-resolution concept grounding`](ideas/IDEA-160_full_resolution_concept_grounding.md)、[`IDEA-161 / intermediate-patch concept signal`](ideas/IDEA-161_intermediate_patch_concept_signal.md)、[`IDEA-162 / learnable concept readout probe`](ideas/IDEA-162_learnable_concept_readout_probe.md)、[`IDEA-163 / tri-state evidence predicate set`](ideas/IDEA-163_tri_state_evidence_predicate_set.md)、[`IDEA-164 / observable signed evidence`](ideas/IDEA-164_observable_signed_evidence.md)、[`IDEA-165 / constrained evidence graph search`](ideas/IDEA-165_constrained_evidence_graph_search.md) |
| `class_competition` | 细粒度候选和seen/unseen联合竞争如何避免错误修正 | 当前无已晋级V4 Idea；相关V2历史按需检索 |
| `learning_generalization` | 训练目标与选择规则如何迁移到未见类别 | 当前无已晋级V4 Idea；相关V2历史按需检索 |
| `reliability_robustness` | 如何识别不可靠证据并保持关闭路径 | GAVE、RGT包含该机制标签，但主类别仍为`visual_grounding` |
| `evaluation_diagnostic` | 资产、缓存、评估和诊断合同是否可信 | 作为诊断证据记录，不包装为论文创新 |

每张新Idea仍保存在`research/ideas/IDEA-xxx_<slug>.md`；本表只做检索入口，不复制公式、结果或状态事实。

## FRAMEWORK-V1

- 状态：历史正式框架。
- 正式模型：GTPJ。
- 代码：[`model/frameworks/v1/`](../model/frameworks/v1/)。
- 实验入口：[`experiments/v1/`](../experiments/v1/)。

## FRAMEWORK-V2

- 状态：历史正式框架。
- 正式模型：TG-VPR-H1。
- 正式来源：[`IDEA-001`](ideas/IDEA-001_tg_vpr_h1.md)，状态`supported`。
- 历史论文候选：[`IDEA-005`](ideas/IDEA-005_tst.md)，状态`supported`，不等同于V4的GTD。
- 代码：[`model/frameworks/v2/`](../model/frameworks/v2/)。
- 214条快速尝试与正式Experiment：[`experiments/v2/`](../experiments/v2/)。
- 未接纳代码只保留在对应实验分支与准确commit，不进入`main`。

V2的大量失败与辅助候选不在本索引逐项展开；准确状态保留在 [`experiments/v2/EXPERIMENT_QUEUE.csv`](../experiments/v2/EXPERIMENT_QUEUE.csv) 和对应Idea卡中。

## FRAMEWORK-V3-EXPLORATION

- 状态：`closed exploration`。
- 身份边界：没有正式`framework/v3`分支或Tag `v3`，不能包装成正式框架。
- 快速尝试：[`experiments/v3/EXPERIMENT_QUEUE.csv`](../experiments/v3/EXPERIMENT_QUEUE.csv)。

| Idea | 模块 | 最终状态 | 主要去向 |
|---|---|---|---|
| [`IDEA-133`](ideas/IDEA-133_visual_evidence_learning.md) | 角色引导局部视觉证据 | rejected | 保留为V3候选历史 |
| [`IDEA-144`](ideas/IDEA-144_fmc_sr.md) | FMC-SR | rejected after fixed150 rescues | 停止该方向 |
| [`IDEA-146`](ideas/IDEA-146_gtd_tst.md) | GTD-TST | supported | 晋级为FRAMEWORK-V4正式GTD |

## FRAMEWORK-V4

- 状态：`active`。
- 正式方法：TG+GTD。
- 正式来源：[`IDEA-146`](ideas/IDEA-146_gtd_tst.md)。
- 代码：[`model/frameworks/v4/`](../model/frameworks/v4/)。
- 当前问题：利用实例级局部视觉证据改善细粒度竞争，同时不破坏seen准确率与GTD关闭路径。

| Idea | 模块 | 状态 | TRY | 源分支 |
|---|---|---|---|---|
| [`IDEA-158`](ideas/IDEA-158_gave.md) | GAVE | weak signal only，未晋级 | V4-TRY-001 | `exp/v4/innovation/innovation-001-gave` |
| [`IDEA-159`](ideas/IDEA-159_rgt.md) | RGT | rejected before training | V4-TRY-002 | `exp/v4/innovation/innovation-002-rgt` |
| [`IDEA-160`](ideas/IDEA-160_full_resolution_concept_grounding.md) | 576-patch概念落地oracle | rejected before queue | 无；pre-queue最小证伪 | 无；未创建创新分支 |
| [`IDEA-161`](ideas/IDEA-161_intermediate_patch_concept_signal.md) | 中间层576-token直接读取oracle | revised：只否定裸余弦读取 | 无；pre-queue双卡1000图诊断 | 无；未创建创新分支 |
| [`IDEA-162`](ideas/IDEA-162_learnable_concept_readout_probe.md) | 自然prompt＋共享学习型读取探针 | supported signal only，待owner范式准入 | 无；pre-queue三步诊断 | 无；未创建创新分支 |
| [`IDEA-163`](ideas/IDEA-163_tri_state_evidence_predicate_set.md) | 三态视觉证据谓词集 | rejected before GZSL training | 五项最小证伪门槛全部失败 | `exp/v4/innovation/innovation-003-tri-state-evidence-set` |
| [`IDEA-164`](ideas/IDEA-164_observable_signed_evidence.md) | 候选无关可观察性＋固定参考有符号证据 | rejected at Gate 1 | `o`退化高常数、signed-d迁移与双因果删除均失败 | `exp/v4/innovation/innovation-004-observable-signed-evidence` |
| [`IDEA-165`](ideas/IDEA-165_constrained_evidence_graph_search.md) | 共享概念证据图＋容量约束精确搜索 | owner已批准主条件＋2次补救 | 待运行 | 待从`52088f69`创建独立分支 |

准确数字、commit、配置和输出URI见 [`experiments/v4/EXPERIMENT_QUEUE.csv`](../experiments/v4/EXPERIMENT_QUEUE.csv)。

## 三创新论文门槛

最终目标仍是三个围绕同一核心研究问题、各自有独立证据且能够自然串联的创新。当前正式主线只有TG+GTD；GAVE未晋级、RGT已拒绝，不能为了凑数量写成最终三创新框架。
