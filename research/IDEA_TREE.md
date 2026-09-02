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
| `visual_representation` | 如何保留候选特定的36patch视觉证据 | [`IDEA-218 / CCMVE`](ideas/IDEA-218_class_conditional_mil_visual_expert.md)：互补性成立但共同投影破坏unseen几何 |
| `semantic_representation` | 类别语义和原型如何形成更有结构的表示 | [`IDEA-001 / TG-VPR-H1`](ideas/IDEA-001_tg_vpr_h1.md) |
| `cross_class_transfer` | seen知识如何可靠迁移到unseen原型 | [`IDEA-005 / TST`](ideas/IDEA-005_tst.md)、[`IDEA-146 / GTD-TST`](ideas/IDEA-146_gtd_tst.md) |
| `visual_grounding` | 实例级局部视觉证据能否支持或反驳类别语义 | [`IDEA-133`](ideas/IDEA-133_visual_evidence_learning.md)、[`IDEA-158 / GAVE`](ideas/IDEA-158_gave.md)、[`IDEA-159 / RGT`](ideas/IDEA-159_rgt.md)、[`IDEA-160 / full-resolution concept grounding`](ideas/IDEA-160_full_resolution_concept_grounding.md)、[`IDEA-161 / intermediate-patch concept signal`](ideas/IDEA-161_intermediate_patch_concept_signal.md)、[`IDEA-162 / learnable concept readout probe`](ideas/IDEA-162_learnable_concept_readout_probe.md)、[`IDEA-163 / tri-state evidence predicate set`](ideas/IDEA-163_tri_state_evidence_predicate_set.md)、[`IDEA-164 / observable signed evidence`](ideas/IDEA-164_observable_signed_evidence.md)、[`IDEA-165 / constrained evidence graph search`](ideas/IDEA-165_constrained_evidence_graph_search.md)、[`IDEA-167 / conditional information evidence`](ideas/IDEA-167_conditional_information_evidence.md)、[`IDEA-168 / concept-specific region interaction`](ideas/IDEA-168_concept_specific_region_interaction.md)、[`IDEA-169 / contrastive concept interaction`](ideas/IDEA-169_contrastive_concept_interaction.md)、[`IDEA-171 / hypothesis-conditioned visual completion`](ideas/IDEA-171_hypothesis_conditioned_visual_completion.md) |
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
| [`IDEA-165`](ideas/IDEA-165_constrained_evidence_graph_search.md) | 共享概念证据图＋容量约束精确搜索 | rejected after two rescues | capacity1/2均-0.2pp，2×2区域0pp | `exp/v4/innovation/innovation-005-constrained-evidence-graph` |
| [`IDEA-166`](ideas/IDEA-166_text_conditioned_visual_distribution.md) | 文本条件低秩视觉分布 | rejected after main＋2 rescues | LOO信号成立，但三条件均比Point低约2.66pp、净纠正-57 | `exp/v4/innovation/innovation-006-text-conditioned-distribution` |
| [`IDEA-167`](ideas/IDEA-167_conditional_information_evidence.md) | 条件信息增益最小充分证据 | revised before run；未执行 | 过宽Gate被拆分，永久保留历史 | 无运行分支 |
| [`IDEA-168`](ideas/IDEA-168_concept_specific_region_interaction.md) | 共享文本概念的跨区域非加性交互 | rejected at Gate 0 | 四项概念特异性门全失败；符号稳定但不优于对照 | `exp/v4/innovation/innovation-008-concept-region-interaction` |
| [`IDEA-169`](ideas/IDEA-169_contrastive_concept_interaction.md) | 固定Attention的同角色概念对比交互 | rejected at Gate 0 | 仅60对/13类；覆盖与三项效应门失败 | `exp/v4/innovation/innovation-009-contrastive-concept-interaction` |
| [`IDEA-170`](ideas/IDEA-170_content_aware_inpainted_interaction.md) | 内容感知补全的跨区域交互 | rejected at Gate 0 | 两种补全均胜随机、均不胜困难对照；方向关闭 | `exp/v4/innovation/innovation-010-content-aware-inpainted-interaction` |
| [`IDEA-171`](ideas/IDEA-171_hypothesis_conditioned_visual_completion.md) | HCVC：候选条件视觉补全 | 双Agent范式准入通过；proof-of-path未运行 | 无；Gate 0合同已冻结 | 无；未创建实现或实验分支 |
| [`IDEA-172`](ideas/IDEA-172_text_difference_active_evidence_acquisition.md) | 文本差异主动高清取证 | rejected at proof gate | Oracle+14pp，但真实行动-0.2pp、净纠正-1 | `exp/v4/diagnostic/diagnostic-001-active-evidence-acquisition` |

准确数字、commit、配置和输出URI见 [`experiments/v4/EXPERIMENT_QUEUE.csv`](../experiments/v4/EXPERIMENT_QUEUE.csv)。

## 三创新论文门槛

最终目标仍是三个围绕同一核心研究问题、各自有独立证据且能够自然串联的创新。当前正式主线只有TG+GTD；GAVE未晋级、RGT已拒绝，不能为了凑数量写成最终三创新框架。
