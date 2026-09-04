# GZSL 跨分支总账索引

本文件只回答“记录在哪里”，不复制代码、结果正文或大文件。实验事实仍以对应 Idea 卡、队列行、Git commit 和仓库外结果 URI 为准。

## 一条记录如何追溯

```text
Idea 卡 → 版本队列/正式 Experiment → 实验分支 → RUN commit → 代码入口/配置 → 仓库外结果
```

- Idea 卡记录问题、假设、状态和结论；不复制完整代码。
- Git commit 是完整代码快照；候选代码留在实验分支，正式接纳后才进入 `model/frameworks/vX/`。
- `experiments/vX/EXPERIMENT_QUEUE.csv` 是快速尝试的运行账本。
- checkpoint、原始日志和大结果保存在仓库外，账本只记录 URI 和哈希。

## 正式框架身份

| 身份 | 状态 | 冻结引用 | 正式代码入口 | 账本入口 |
|---|---|---|---|---|
| FRAMEWORK-V1 | 历史正式框架 | `origin/framework/v1` / `v1` | `model/frameworks/v1/` | `experiments/v1/` |
| FRAMEWORK-V2 | 历史正式框架 | `framework/v2` / `v2` | `model/frameworks/v2/` | `experiments/v2/` |
| V3 | closed exploration，不是正式框架 | 无正式branch/tag | 候选commit | `experiments/v3/` |
| FRAMEWORK-V4 | 历史正式框架 | `framework/v4` / `v4` | `model/frameworks/v4/` | `experiments/v4/` |
| FRAMEWORK-V5 | 历史正式框架 | `framework/v5` / `v5` | `model/frameworks/v5/` | `experiments/v5/` |
| V6 | development，不是正式框架 | 无正式branch/tag | `model/frameworks/v6/`及对应候选commit | `experiments/v6/`（在 `main`） |
| FRAMEWORK-V7 | 当前论文正式框架 | `framework/v7` / `v7` | `model/frameworks/v7/`；训练入口复用`model/frameworks/v6/train_compiled_pclr.py` | `experiments/v7/`（在 `main`） |

## Idea 总库入口

| 范围 | 位置 | 说明 |
|---|---|---|
| IDEA-001～172 | `research/ideas/`、`research/IDEA_TREE.md`、`research/archive/IDEA_TREE_V2_LEGACY.md` | 已进入当前分支历史；V2详细状态以V2队列为准 |
| IDEA-186 | `research/ideas/IDEA-186_pairwise_contrastive_laplacian_reasoning.md` | PCLR主线历史 |
| IDEA-188～202 | `main:research/ideas/` | V5/V6及FRAMEWORK-V7形成过程；当前V5诊断分支不含这些文件 |
| IDEA-203～212 | 未发现可核实Idea卡 | 编号空档，不补造 |
| IDEA-213～224 | 各自的 `exp/v6/diagnostic/idea-*` 分支 | 诊断/候选验证，均未进入正式框架 |
| IDEA-225 | 未发现可核实Idea卡 | 编号空档，不补造 |
| IDEA-226～227 | 独立专家属性诊断分支 | 见下表 |
| IDEA-228～231 | 未发现可核实Idea卡 | IDEA-232提到229～231，但Git分支、历史、reflog均未找到，记为断链 |
| IDEA-232 | 当前分支的Idea卡与V5队列 | 专家属性CRR Level 1通过，Level 2待运行；不是范式创新 |

## 最近诊断与专家属性记录

| Idea | 状态 | 分支 | 代码/结果身份 | 结论 |
|---|---|---|---|---|
| IDEA-213 RTV | rejected | `exp/v6/diagnostic/idea-213-rtv-gate` | `f74bfea` | role-region transport gate失败 |
| IDEA-214 NRMP | rejected | `exp/v6/diagnostic/idea-214-nrmp-gate` | `1cf3533` | natural-role MIL projection失败 |
| IDEA-215 CRG | rejected | `exp/v6/diagnostic/idea-215-crg-gate` | `6b95bd8` | conditional residual grounding失败 |
| IDEA-216 CAEF | rejected | `exp/v6/diagnostic/idea-216-caef-gate` | `1c33344` | conflict-aware evidence fusion失败 |
| IDEA-217 PMVE | rejected | `exp/v6/diagnostic/idea-217-pmve-gate` | `d050093` | patch-MIL visual expert失败 |
| IDEA-218 CCMVE | revised / proof_of_path | `exp/v6/diagnostic/idea-218-ccmve-gate` | `04e8b87` | 保留错误互补证据，未晋级 |
| IDEA-219 PADC | rejected | `exp/v6/diagnostic/idea-219-padc-gate` | `06da50b` | below parent |
| IDEA-220 RG-DCF | rejected | `exp/v6/diagnostic/idea-220-rgdcf-gate` | `241d9f8` | OOF可靠性未转成H增益 |
| IDEA-221 SCLV | rejected | `exp/v6/diagnostic/idea-221-sclv-gate` | `b64e503` | full gate失败 |
| IDEA-222 PLLRV | rejected | `exp/v6/diagnostic/idea-222-pllrv-gate1a` | `56a1e91` | proof_of_path_failed |
| IDEA-223 R-PLLRV | rejected | `exp/v6/diagnostic/idea-223-rpllrv-gate1a` | `50f1422` | proof_of_path_failed |
| IDEA-224 HCLR | rejected | `exp/v6/diagnostic/idea-224-hclr-gatea` | `619bb23` | stage A失败 |
| IDEA-226 AELI | revised / proof_of_path | `exp/v6/diagnostic/idea-226-aeli-gate1a` | `fe3b507` | 三态专家属性信号可学，但Gate1a失败 |
| IDEA-227 HAEL | rejected | `exp/v6/diagnostic/idea-227-hael-gate1a` | `c51b5ae` | 类别目标改善CBA，但unknown监督塌缩 |
| IDEA-229～231 | missing_record | 未找到 | 未找到 | 不能当作已核实实验 |
| IDEA-232 CRR | testing / keep | `exp/v5/diagnostic/idea-232-crr` | RUN `33e18da`；结果回填 `dd7ae23` | 对称OOF `+1.0859pp`，Level 1通过，待Chen-style Level 2 |

## 最近V7实验位置

`main`中的V7队列和四类INDEX尚未回填这些独立分支，因此先以分支为真实位置：

- Tune：`exp/v7/tune/tune-001-*` 至 `exp/v7/tune/tune-015-*`。
- Ablation：`exp/v7/ablation/ablation-002-*` 至 `ablation-004-*`。
- Confirmation：`exp/v7/confirmation/confirmation-001-multidataset`。
- Diagnostic：`exp/v7/diagnostic/diagnostic-001-clrv-patch`。

这些分支当前均未被 `main` 包含；在账本正式回填前，不把 `main` 中“当前无实验”的V7 INDEX当作最新状态。

## 已知待补项

1. IDEA-229～231没有可核实卡片、分支或commit；只能标缺失，不能重建结论。
2. `main` 的V6队列有两行未绑定 `code_commit`；完成态的V6-TRY-002需要补准确身份，planned态V6-TRY-005需在实际运行前冻结。
3. V7近期实验仍分散在独立分支，`main`中的V7队列与INDEX尚未回填。
4. 本索引建立在当前V5诊断分支；合入何处由owner按“未接纳候选不得混入main”的规则另行决定。
