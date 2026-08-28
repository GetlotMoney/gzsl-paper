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
- 未接纳代码：[`model/candidates/v2/`](../model/candidates/v2/)。

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

| Idea | 模块 | 状态 | TRY | 代码 |
|---|---|---|---|---|
| [`IDEA-158`](ideas/IDEA-158_gave.md) | GAVE | weak signal only，未晋级 | V4-TRY-001 | [`idea_158_gave`](../model/candidates/v4/idea_158_gave/) |
| [`IDEA-159`](ideas/IDEA-159_rgt.md) | RGT | rejected before training | V4-TRY-002 | [`idea_159_rgt`](../model/candidates/v4/idea_159_rgt/) |

准确数字、commit、配置和输出URI见 [`experiments/v4/EXPERIMENT_QUEUE.csv`](../experiments/v4/EXPERIMENT_QUEUE.csv)。

## 三创新论文门槛

最终目标仍是三个围绕同一核心研究问题、各自有独立证据且能够自然串联的创新。当前正式主线只有TG+GTD；GAVE未晋级、RGT已拒绝，不能为了凑数量写成最终三创新框架。
