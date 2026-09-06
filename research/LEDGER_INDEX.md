# 跨分支总账：证据与代码位置

当前主线见 [README](../README.md)。本索引保存位置，不替代原始结果，不因代码分支名称判定实验成功。

## 本次盘点（2026-09-06，同步前快照）

- 328 个本地分支；扫描分支 tip 中的 Idea、队列、Experiment、参数矩阵、result、框架记录与候选文档。
- 找到 227 个不同 Idea 编号，最大 IDEA-237；索引共 1073 条不同路径/内容版本，不能把条数当实验数。
- [RECORD_INDEX.csv](RECORD_INDEX.csv)：记录类型、编号、路径、blob、源 commit、源分支、同内容引用数。`declared_status` 是原文首个 status 字段，可能过时，不能代替整卡最终结论。
- [CODE_SNAPSHOTS.csv](CODE_SNAPSHOTS.csv)：每个分支的准确快照与远端同名分支状态。`no_remote_branch` 不等于 commit 一定未上传；分支存在也不证明训练完成。
- 8 张卡片有本地未提交内容版本，仅登记位置，未把草稿导入正式卡或发布为结果。

## 如何取回

在 RECORD_INDEX 按 Idea 编号或实验路径检索，取该行完整 source_commit 与 path：

```powershell
git show <source_commit>:<path>
git show <code_commit>:<代码路径>
```

索引的 source_commit 是保存记录的快照，不一定是实际训练 code_commit；后者从原 RUN 账本读取。公开远端可达的提交可用 GitHub `/blob/<commit>/<path>` 浏览；仅本地提交必须从本地 Git 读取。`local_uncommitted` 只能从原工作目录读取。

## 正式历史版本

以下是已存在的正式冻结身份，不因当前转向新研究而撤销；也不自动成为新实验父条件。

| 版本 | 冻结 commit | 引用 |
|---|---|---|
| V1 | `7d842e5c0e5554409eedb3097fea5130a848c9e4` | `framework/v1` / `v1` |
| V2 | `3dc078c0d52bf358bf24a26e48346c97de9e99ca` | `framework/v2` / `v2` |
| V4 | `52088f69d7ac4e574e7b63c28b21ac0da7789933` | `framework/v4` / `v4` |
| V5 | `52b511d77b4ad048f35b40dc3cbd9afd092167e9` | `framework/v5` / `v5` |
| V7 | `b32a16f848c34f8e09d03b27d2f22ed445b9a295` | `framework/v7` / `v7` |

V3 是已关闭探索；V6 是开发阶段；现有 V8 名称仅为原型。无正式标签，不补造、不移动历史引用。

## 当前缺口与证据边界

- 当前扫描未找到卡片的编号：IDEA-147, IDEA-176, IDEA-177, IDEA-178, IDEA-181, IDEA-225, IDEA-228, IDEA-229, IDEA-230, IDEA-231。这里只检查本地分支 tip 与当前目录，不声称遍历已删除引用、全部历史或服务器；不补造结论。
- 原总账曾将 IDEA-203～212 标缺失，本次已在跨分支扫描找到相关卡片；应按 RECORD_INDEX 的准确位置读取，旧“缺失”判断不再作为当前事实。
- `main` 的 V6-TRY-002 未绑定 code_commit，V6-TRY-005 是旧计划且无 code_commit。原记录保持不变；未补到直接证据前，不猜测实际执行代码。
- V7/V8 等分散账本通过 source commit 统一可查，本次没有复制原队列或跨分支拼接指标。
- 未复核服务器作业、仓库外 checkpoint/日志可用性或全部科学结论；旧 running/planned 只表示原记录状态。
- 本次同步范围为管理文档、定位索引及已有正式身份；不删除实验分支、工作树，不批量发布未接纳候选分支。

## GitHub 与本地分工

`main` 为管理入口及已接纳代码；正式版本固定 branch/tag；候选分支保留原代码与证据。现有本地 main 在整理前较远端领先 88 个历史提交，正式身份同步与本次文档提交分别核验。不要用 `push --all`、force-push 或移动标签“整理”历史。

更新索引使用现有 Python 与 Git，无新增依赖、定时任务或后台服务：

```powershell
python tools/index_research_records.py
# 从 main 工作目录执行时，可显式加入有未提交卡片的工作目录：
python tools/index_research_records.py --workspace D:/Backup/Documents/ChatGPT/gzsl-paper
```

该命令只查询 refs 和轻量记录，覆盖两份派生 CSV，不改原卡、队列、代码或分支；终端输出本次统计。新增目录或重算科学结论均不在其范围。CSV 的 observed_at_utc 表示实际盘点时间。

索引是带日期的盘点，不是实时服务。新证据落盘后更新原记录及索引；不因索引旧值把历史分支 tip 当当前 RUN。

<details>
<summary>整理前总账原文：仅供历史追溯，以上盘点覆盖旧定位判断</summary>

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
| IDEA-232 | 当前分支的Idea卡与V5队列 | 专家属性CRR Level 1通过、Level 2因U下降拒绝；不是范式创新 |

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
| IDEA-232 CRR | rejected_level2_unseen_degradation | `exp/v5/diagnostic/idea-232-crr` | pre-run `3d6b1a45`；result SHA `926ee47c...` | H `+2.849050`但U `-0.638064`，命中失败门，专家属性主线关闭 |

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

</details>
