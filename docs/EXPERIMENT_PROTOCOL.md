# 实验与代码身份规范

## 记录归属

当前状态从 [README](../README.md) 进入；[Idea 规范](../research/README.md) 保存假设；本文件规定实验与代码的最小合同。

- 历史 `experiments/v1/` 至现有版本路径、编号、结果只读保留，不复制、迁移或改写。
- 继续既有 Experiment 使用原身份；需新实验时按新规则登记，并引用原证据。
- 新独立研究按 `track_id`（研究主线）登记，例如 `triadic-native`，不把探索阶段叫正式 V8。第一次有真实计划才创建 `experiments/<track_id>/EXPERIMENT_QUEUE.csv`。
- 新实验分支为 `exp/<track_id>/<kind>/<id>-<slug>`；旧 `exp/vX/...` 保留；临时实现仍用 `codex/<slug>`。分支是工作位置，不是正式框架。
- 总索引只保存记录位置。原队列/参数矩阵是运行事实源，结果正文解释结果；同一 RUN 不另抄成新成绩。

## 开工前必须回答

| 项目 | 内容 |
|---|---|
| research_mode | `incremental` 增量 / `replacement` 替换 / `independent` 独立框架 |
| base_commit | owner 指定的准确 40 位代码起点，不能写“最新 main” |
| reuse_scope | 哪些已核实部件复用，源 commit 与路径；不复用写无 |
| change_scope | 本次新增、替换、移除什么 |
| comparison_ref | 公平比较基线的准确 commit、配置/RUN、数据与评估条件 |
| question / variable | 唯一研究问题与关键变量 |

代码起点和比较对象是两个字段：从旧仓库复用数据加载器，不等于继承其方法；独立新框架也不要求重写成熟基础设施。

增量或替换从明确正式父条件分叉；独立框架从 owner 认可的准确工程起点复用必要基础代码，单独实现新方法。失败后的新候选回到指定父条件，不沿失败候选堆叠；同候选修 bug 可继续但必须冻结新 code commit。

## TRY 与正式 Experiment

快速 TRY 一行对应一个代码/配置条件、一个 seed、一次执行。新主线行至少记录：

```text
attempt_id,track_id,idea_id,experiment_id,research_mode,base_commit,reuse_scope,
change_scope,comparison_ref,code_commit,config_ref,seed,asset_ref,evaluation_protocol,
test_used_for_selection,unseen_images_used_for_gradient,strict_blind_claim,
status,U,S,H,ZS,history_uri,log_uri,model_uri,output_uri,decision
```

代码或结果尚未产生时留空并明确 planned；不得把计划 commit、草稿哈希或分支 tip 当成真实运行代码。诊断不适用 U/S/H/ZS 时留空并链接诊断量及判定，不伪造零分。

正式验证沿用最小四文件：`EXPERIMENT.yaml`、`configs/RUN-xxx.yaml`、`PARAMETER_MATRIX.csv`、`result.md`。代码起点等实验级字段写在 EXPERIMENT.yaml，参数矩阵每行只记录 RUN 级差异和引用。快速 TRY 晋级后保留原记录，正式实验引用原 TRY，不把同一次运行计为两个独立证据。

只改已审配置、普通参数或 seed：新增 RUN。修改公式、forward、loss、数据或评估语义：新 Experiment；核心研究假设改变时同时新建 Idea。纯文档/工程修复不自动冒充新创新。

## 代码保存：pre-run → RUN → post-run

1. pre-run 提交代码、完整配置和计划，不写虚构成绩。准确 `code_commit` 由该提交产生，在后续账本提交中绑定，不能让提交包含自己的哈希。
2. 服务器使用准确 code commit、冻结配置及资产身份运行，仓库外独立输出目录不得覆盖。语义变化按 AGENTS 审查要求执行。
3. post-run 提交真实指标、评估历史与日志/模型 URI、研究决定；仍引用实际运行的 code commit，不能换成 post-run tip。

Git 保存代码、配置、轻量账本与必要 manifest；不保存数据、cache、checkpoint、大日志或密钥。跨分支证据用完整 commit + 路径定位；分支名仅作辅助。main 接纳记录不代表接纳候选代码。

## 审查与框架图

唯一审查标准见 [AGENTS 第6节](../AGENTS.md#6-必要的双-agent-审查)。两名 Agent 独立审同一冻结 commit、交换一次完整清单并回应；代码运行门为双方 P0/P1=0，P2记录。15分钟是默认反馈目标，不以超时自动替代正确性判断。未变化证据不重复检查；纯账本不启动独立 Agent 审核。

module、forward、loss、数据流或评估语义变化时必须提供可直接打开的 `framework_diagram.html`，注明 commit、输入输出、关键流、loss、logits/metrics 出口和评估边界。快速 TRY 必要图可放同主线现有位置并从行引用，无需增加空正式目录；纯参数与文档复用原图。

## 训练、比较与决定

当前论文主协议为 `chen_shiming_code_aligned_test_selected_gzsl`，固定披露 `test_used_for_selection: true`、`unseen_images_used_for_gradient: false`、`strict_blind_claim: false`。CUB 为 trainval 150类/7057图，batch50、200名义epoch、28228次更新、每步独立 randperm(7057)[:50]、每141步 official 评估。改变条件需明确登记新比较条件。

U/S/H 在全部 seen+unseen 联合竞争，ZS 只在 unseen 内竞争；均来自整模型同一 best-H checkpoint，保存完整评估历史。分阶段选 test 必须标 `nested_official_test_selection: true`。不同骨干/checkpoint/预处理/文本/训练预算不能直接排名。

独立新框架按 AGENTS 使用公平冻结 CLIP 比较：准确率候选 seed7 matched fixed200 至少 +0.5 H；最终3 seed均值至少 +0.5 H且每个seed不负。成本/通用性主张需 H 不低于公平对象并有直接证据。旧实验的 +1H、+0.2H 等门仅解释旧结果，不回写或重新宣判历史。

推理关闭仅为诊断；正式消融 Full 与 Full-minus-branch 独立重新初始化、完整训练，预算、seed、优化器和选择口径一致。独立新框架不强制任何分支关闭都返回某个旧框架。

结束时更新实际运行状态及 keep/drop/promote 决定、Idea解释和下一步。失败、中断、无效结果均保留并标适用边界；已有记录未找到证据时写未核实。promote 表示进入正式验证，不代替 owner 接纳正式框架。
