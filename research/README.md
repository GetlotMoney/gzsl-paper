# 研究知识：保存、复用与演化

当前目标与下一步只维护在 [项目入口](../README.md)；本文件规定可长期复用的记录方式。

## 对象与唯一职责

| 对象 | 回答的问题 | 存储 |
|---|---|---|
| 主线 / 问题 | 为什么研究、缺少哪个答案 | `IDEA_TREE.md` 的主线与问题节点，不为每个问题建目录 |
| Idea | 什么机制可能有效、为何、如何证伪 | `ideas/IDEA-xxx_<slug>.md` |
| Experiment / RUN | 怎么比较、实际发生什么 | 原队列或正式实验目录，见实验规范 |
| 框架候选 | 哪些机制组成完整方法、缺什么证据 | 有具体方案后沿用 `proposals/`，不提前建立空候选 |
| 正式版本 | owner 接纳的准确实现 | 固定 branch/tag 与原框架记录 |

## Idea 存储

Idea 属于项目总库，不属于某个 V 版本。一个假设只维护一张卡，可被多主线引用；换框架不复制卡片。编号永久不复用、不重编号；新编号必须查跨分支 [记录索引](RECORD_INDEX.csv)，不能只数当前文件夹。

每张卡至少保存：问题、假设、核心机制、来源、成立与失败条件、实验引用、当前结论、适用边界和按日期追加的演化记录。保留 `method_originality` 与 `method_maturity` 的区分，未知写未知。

```yaml
idea_id: IDEA-xxx
title: 中文短名
track_refs: [主线标识]
problem: 要回答的问题
hypothesis: 可证伪假设
core_change: 核心机制
evidence_refs: []  # 论文或项目内直接证据；不能用聊天代替论文证据
reuse_refs: []     # 引用旧 Idea/实验，注明复用知识、机制或代码部件
success_condition: 预先确定
failure_condition: 预先确定
work_status: planned  # planned / active / paused / closed
evidence_status: untested  # untested / partial / supported / rejected / inconclusive
acceptance_status: unaccepted  # unaccepted / candidate / accepted
experiment_refs: []
failure_boundary: 尚未验证
history: []
```

模板只是有真实内容时的最小字段示意，不批量给旧卡填猜测状态。旧 `status: proposed/testing/supported/revised/rejected` 原样保留；复用时追加三种状态及时间，不抹去旧结论。工作结束不等于科学假设被否定；有信号不等于创新成立或正式接纳。

## 三种复用

1. **知识复用**：引用旧结论、反例、诊断合同，不继承旧代码。
2. **同一机制跨条件验证**：核心假设不变，原 Idea 增加实验引用，明确新框架、数据与协议；不把旧失败状态改回 proposed，不用新结果覆盖旧条件下的结论。
3. **机制或假设改变**：核心机制、学习信号或可证伪假设改变，建立新 Idea，引用旧编号并说明差异。不是因为换 seed 或普通参数就分配新 Idea。

代码复用独立记录在实验的 `reuse_scope`，精确到源 commit 与路径；引用旧 Idea 不代表已获准把失败候选作为代码基线。

复用前直接读卡片及关联实验，确认失败属于机制反例、实现缺陷、成本不合适，还是特定条件下无收益。旧 `REUSE_INDEX.md` 仅作历史导航，不能替代上述证据核对。

## 分类与更新

沿用问题类别：`semantic_representation`、`cross_class_transfer`、`visual_grounding`、`class_competition`、`learning_generalization`、`reliability_robustness`、`evaluation_diagnostic`。类别用于检索，不预先规定网络部件。

- 开始：明确主线、问题、假设和判断条件；Idea 阶段允许代码起点待定。
- 运行前：把准确代码起点、比较基线、配置和资产身份写入实验；未确定不得运行。
- 结束：原账本写结果，Idea 写解释和失败边界，树只改关系与状态链接，首页只改下一步。
- 正式组合：由整体机制与证据决定，不要求凑三个创新，不把模块数量当创新数量。

## 论文来源

外部论文必须重新核对原文；`papers/` 保存来源、相关内容和适用边界，PDF、解析全文和缓存放仓库外并记录 URI 与必要哈希。本项目不检索或隐式复用受保护旧项目知识。具体创新准入与代码审查沿用 [AGENTS.md](../AGENTS.md)，本次记录整理不触发新创新审查。
