# Agent A initial review — FRAMEWORK-V7 promotion

## 结论

- Reviewer：Agent A
- 阶段：FRAMEWORK-V7 晋级代码审查阶段1，独立只读审查
- 冻结 commit：`568b01ec8a8e48ffe78336a6fc99f7708de03cbc`
- promotion source：`2f7837266f4077b3fb7e40927fc6571499a76747`
- source RUN：`V6-TRY-006`
- source RUN commit：`8de7cebda0235ab12e1b4b8f669134c8f4e2c075`
- V6 code-review commit：`b707b0c4671051244cebf4f8404299fc016b281e`
- 测试证据：主Agent声明 `32 passed`；本阶段未重跑测试，未运行服务器
- 最终判断：`revise`
- P0：0
- P1：2
- P2：4

运行代码层面，`model/frameworks/v7` 已基本满足独立无图部署：部署模型只持有 `Reader + Q + b` 六个导出张量，前向只执行 `h(x) @ Q.T + b`，没有 V6 模块、在线 Top-K、关系边、Laplacian solve、cap/std 或 late ensemble 依赖。评估口径也按 200 类 GZSL 和 50 类 ZS 实现。

阻断点不在 logits 计算，而在正式晋级身份：当前 V7 账本/HTML 图没有把“V7 部署代码冻结身份”准确记录清楚，并且 IDEA-201 顶部状态与底部晋级记录互相矛盾。按项目规则，正式 framework 的代码身份与 HTML 图是事实源；这两个问题会让后续 `framework/v7` / `v7` tag 的审计身份不可信，因此判 `revise`。

## 已读取范围

已读取：

- `AGENTS.md`
- `README.md`
- `research/README.md`
- `research/IDEA_TREE.md`
- `research/ideas/IDEA-201_compiled_pclr.md`
- `model/frameworks/v7/__init__.py`
- `model/frameworks/v7/model.py`
- `model/frameworks/v7/evaluate.py`
- `config/framework_v7.yaml`
- `experiments/v7/FRAMEWORK.yaml`
- `experiments/v7/EXPERIMENT_QUEUE.csv`
- `experiments/v7/framework_diagram.html`
- `experiments/v7/innovation/INDEX.md`
- `experiments/v7/ablation/INDEX.md`
- `experiments/v7/tune/INDEX.md`
- `experiments/v7/confirmation/INDEX.md`
- V6 账本：`experiments/v6/FRAMEWORK.yaml`、`experiments/v6/EXPERIMENT_QUEUE.csv`、`experiments/v6/innovation/INDEX.md`、`experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/EXPERIMENT.yaml`、`result.md`、`CODE_REVIEW.md`、`MICRO_BATCH.md`、`PARAMETER_MATRIX.csv`
- `tests/frameworks/v7/test_v7_deployment.py`
- `2f7837266f4077b3fb7e40927fc6571499a76747..568b01ec8a8e48ffe78336a6fc99f7708de03cbc` 的 diff

阶段1没有读取 V7 的 B 审查文件。过程说明：一次宽范围关键词检索误匹配到历史 V6 review 文件的少量行；这些不是本轮 V7 B 文件，也未作为 V7 独立判断依据。后续检索已排除 `reviews/**`。

## 关键审查结论

### 1. 独立部署是否只用 Reader / Q / b

通过。

- `V7DeploymentModel` 的 constructor 只接收并注册六个 buffer：`q`、`bias`、`reader_in_weight`、`reader_in_bias`、`reader_out_weight`、`reader_out_bias`。
- `from_export()` 要求 export 字段集合与这六项严格相等；缺字段或多字段都会拒绝。
- `forward()` 只计算 `image=normalize(x)`、`readout=Reader(x)`、`cat(image, readout) @ q.T + bias`。
- `state_dict()` 测试明确排除 `edge_index`、`relation_embeddings`、`incidence`、`laplacian_map`。

结论：部署代码没有 V6 training module、online graph、Top-K、边张量、Laplacian solve、cap/std、late role/gamma 运行依赖。

### 2. checkpoint 身份

代码路径部分通过，正式账本身份有 P1。

- `load_v7_checkpoint()` 硬检查 checkpoint 内部身份：`experiment_id == V6-TRY-006`、`code_commit == 8de7cebda0235ab12e1b4b8f669134c8f4e2c075`、`config_sha256 == 73a812268b18e9f46a2cedf59acdabb8ef0cdb13388ec83b5f23b73475e4239b`，并要求 `export` 是 dict。
- `config/framework_v7.yaml` 绑定 `source_checkpoint_sha256: a551de9d...f2207c9`，并记录同一个 source RUN commit 与 source training config SHA。
- V6 RUN 账本记录 `model_best_sha256: a551de9d...f2207c9`，与 V7 config 一致。

但 `experiments/v7/FRAMEWORK.yaml` 的 `reviewed_code_commit` 仍是 `b707b0c...`，即 V6 训练前代码审查身份，不是当前新增 V7 部署/评估代码所在的冻结 commit `568b01e...`；同时 V7 HTML 图写的是 `Formal commit: pending promotion freeze`。这不影响 loader 运行，但影响正式 framework 身份签字，列为 P1-1。

### 3. evaluator U/S/H/ZS

通过。

- evaluator 从 asset config 读取固定 CUB tensors，`seen` 来自 `train_labels`，`unseen` 是 200 类全集减 seen。
- seen 与 unseen 的 GZSL 预测都对 200 类 logits 直接 `argmax`。
- ZS 只对 unseen columns 做 `argmax` 后映射回全局 class id。
- `per_class_accuracy()` 对每个 class 平均，`H = 2*S*U/(S+U)`。
- `EXPECTED_METRICS` 与 V6 result / V7 framework / V7 config 中的 U/S/H/ZS 一致：`77.606910 / 83.639657 / 80.510432 / 88.473403`。

未发现 U/S/H/ZS 计算口径偏离项目协议或跨 checkpoint 拼接。

### 4. 无 V6/在线图运行依赖

通过。

- `model/frameworks/v7/model.py` 不 import V6、V4 PCLR、relation asset 或 graph solver。
- `evaluate.py` 只从 V4 训练工具复用 `load_config/load_assets` 作为数据加载入口；部署 logits 仍来自 V7 model。
- V7 config 明确声明 `online_top_k: false`、`online_relation_edges: false`、`online_laplacian_solve: false`。

注意：V7 作为晋级框架仍依赖 V6-TRY-006 产出的导出 checkpoint，这是来源依赖，不是部署期在线图依赖。

### 5. 指标/数据边界

通过。

- V7 config、V7 framework、README、IDEA-201 和 V6 result 均披露 `test_used_for_selection: true`、`strict_blind_claim: false`、`unseen_images_used_for_gradient: false`。
- V6 result 明确记录 C-PCLR 低于内部 matched online-V5 和正式 V5，不把原始程序 `drop_gate_b_contract_failed` 改写为准确率门通过。
- V7 文档把论文父基线固定为 TG+GTD，并保留内部 V5/matched online-V5 为开发事实；这一点是诚实披露，不是指标篡改。

风险边界：若后续论文文字把 V7 写成“超过 V5/PCLR-RSE”或“blind-test”，将与账本冲突；当前 V7 文档没有这样写。

### 6. 论文父 TG+GTD 与内部来源并存

基本通过，但受 P1-2 影响。

当前账本清楚区分：

- 论文父框架：TG+GTD / `TUNE-002-RUN-030`，`H=79.070015`；
- repository/internal history：正式 V5 `H=81.068777`、matched online-V5 `H=80.818096`，C-PCLR 相对二者分别低 `0.558345 H` 和 `0.307664 H`；
- owner 晋级理由：把 S/V/I 统一方法作为论文框架，父基线改为公开方法父 TG+GTD，而不是内部 V5。

该并存方式是可接受的，但 IDEA-201 顶部仍保留“不是 Innovation、不是授权创建分支/实验”的旧身份句，与 `status: promoted_framework_v7` 和底部 FRAMEWORK-V7 晋级记录冲突，需要修正。

## P0

无。

## P1

### P1-1：V7 正式框架身份没有准确绑定当前冻结代码 commit

证据：

- 当前审查冻结 commit 是 `568b01ec8a8e48ffe78336a6fc99f7708de03cbc`，该 commit 新增 `model/frameworks/v7/*`、`config/framework_v7.yaml` 和 `experiments/v7/*`。
- `experiments/v7/FRAMEWORK.yaml` 记录 `reviewed_code_commit: b707b0c4671051244cebf4f8404299fc016b281e`，这是 V6-TRY-006 训练代码审查身份，不包含当前新增的 V7 deployment/evaluator。
- `experiments/v7/framework_diagram.html` 仍写 `Formal commit: pending promotion freeze`。
- 项目规则要求正式 HTML 图记录准确 code commit；正式 framework 也应能从账本恢复准确代码身份。

影响：

- 不会改变 V7 logits 计算，但会使正式 `framework/v7` / `v7` tag 的审计签字不闭合。
- 后续读者无法区分：V6 训练代码审查身份、V6 RUN 产物身份、V7 promotion source、V7 deployment/evaluator 代码身份。

关闭条件：

- 在 V7 账本中明确拆分并记录至少四个身份：
  - `promotion_source_commit: 2f7837266f4077b3fb7e40927fc6571499a76747`
  - `v6_reviewed_training_code_commit: b707b0c4671051244cebf4f8404299fc016b281e`
  - `source_run_commit: 8de7cebda0235ab12e1b4b8f669134c8f4e2c075`
  - `v7_deployment_code_commit` 或 `framework_v7_freeze_commit`：当前应指向 `568b01ec8a8e48ffe78336a6fc99f7708de03cbc`，若后续因审查文件/账本修订产生新冻结 commit，则指向最终 commit。
- `experiments/v7/framework_diagram.html` 不得保留 `pending promotion freeze`，必须记录准确代码身份；若不能在自身 commit 内预写自身 SHA，则在后续纯账本 commit 中绑定最终冻结 commit，并在本轮结论中不得声称 V7 身份已完全闭合。

### P1-2：IDEA-201 当前身份文字自相矛盾

证据：

- `research/ideas/IDEA-201_compiled_pclr.md` frontmatter 已改为 `status: promoted_framework_v7`、`performance_status: above_paper_parent_owner_promoted_v7`。
- 同一文件开头仍写“当前身份：双Agent独立复审后的 `proof_of_path / revise` 草稿……不是 Innovation 登记，也不授权创建分支、实现代码或启动实验。”
- 文件底部又记录 owner 已将 S/V/I 统一方法晋级为 `FRAMEWORK-V7`。

影响：

- 这会破坏研究知识源的一致性：同一 Idea 同时声称“已 promoted_framework_v7”和“仍不授权进入实验/框架”。
- 该矛盾不影响 V7 部署代码执行，但会影响论文父条件、Idea 状态、创新登记和后续实验分叉依据的可信度。

关闭条件：

- 修正 IDEA-201 顶部当前身份，使其与 owner promotion 一致：例如写明“当前由 owner 晋级为 FRAMEWORK-V7；历史 Gate B 程序 drop、低于 V5/matched online-V5 和新颖性/成本缺口保留在审核记录中。”
- 保留历史 `proof_of_path/revise`、`drop_gate_b_contract_failed` 和内部 V5 负差值，不得删除或改写；只把“当前身份”更新为现在的 V7 晋级状态。

## P2

### P2-1：V7 config 自身 SHA 未记录

证据：`config/framework_v7.yaml` 当前 SHA256 为 `e7eff7dd2cd1498cefc77128a73c26f07d4c493eda3febc25ca28d730d139b5a`，但 V7 framework 账本未记录该 config 自身 SHA。

影响：不影响 evaluator 运行；但正式框架签字若要复现，应绑定部署 config SHA。

关闭条件：在 V7 framework 或 review 汇总中记录 `framework_v7_config_sha256`。

### P2-2：V7 evaluator 只验证 Full 指标，不复放 S/V/I off

证据：`model/frameworks/v7/evaluate.py` 只加载 exported full head 并复现 Full `U/S/H/ZS`；S/V/I off delta 只来自 V6 result / framework 账本。

影响：不阻断 V7 部署入口，因为 V7 的目标是正式 Full deploy/eval；但如果后续需要把 S/V/I 作为 V7 可执行验收，当前 evaluator 不能单独重放 off 条件。

关闭条件：保留为账本证据即可；若需要可执行 V7 ablation，再增加明确的 export-off 或 checkpoint-off 评估入口并重新审查。

### P2-3：V7 评估路径依赖绝对 `/data/...` source path，Windows 本地无法直接复现

证据：`config/framework_v7.yaml` 的 `source_checkpoint` 和 `asset_source_config` 都是服务器绝对路径。

影响：符合既有仓库外大文件策略；不影响服务器复现，但本地 Windows worktree 不能直接运行 `evaluate.py`。

关闭条件：正式交付时注明评估需在含对应 warehouse 路径的服务器环境运行；或后续增加路径重映射配置，不改变默认身份。

### P2-4：缺真实部署成本指标，效率 claim 仍只能是结构性 claim

证据：V7 framework / README / IDEA-201 均说明部署成本、延迟、显存、吞吐尚未完成。

影响：当前不能写“已证明更快/更高效/Pareto 优势成立”；只能写“部署路径移除了在线图推理，只执行 Reader 与 `hQ^T+b`”。

关闭条件：按 IDEA-201 的最小成本证据，同硬件同 dtype 报告 Parent、R4 online relation head、V7 exported head 的 batch=1 与评测 batch p50/p95 latency、吞吐、显存和模型大小。

## 最强反例

最强代码/合同反例是：在真实服务器上，用 `config/framework_v7.yaml` 指向的 `model_best.pth` 和 asset config 运行 `python -m model.frameworks.v7.evaluate --config config/framework_v7.yaml`，如果 loader 通过身份检查但复现不出 `U/S/H/ZS=77.606910/83.639657/80.510432/88.473403`，或需要 V6 relation graph / Top-K / Laplacian 才能得到这些 logits，那么 V7 的“独立无图部署框架”主张失败。

最强论文边界反例是：如果后续文字把 V7 描述为超过内部正式 V5 或 matched online-V5 的准确率改进，或声称 blind-test / 未看 official test，则与 V6 原始结果冲突。当前账本没有这样改写，但必须继续保持这个边界。

## 关闭条件总表

- P1-1 关闭：V7 framework/diagram 绑定准确 promotion source、V6 training review commit、source RUN commit、V7 deployment/evaluator freeze commit，并移除 `pending promotion freeze`。
- P1-2 关闭：IDEA-201 顶部当前身份与 `promoted_framework_v7` 一致，同时保留程序 drop、低于 V5/matched control 和未完成成本/多数据集/新颖性对照事实。
- P2-1 关闭：记录 `config/framework_v7.yaml` SHA256。
- P2-2 关闭：如需可执行 S/V/I 复放，则新增 V7 off evaluator；否则保留 V6 账本证据即可。
- P2-3 关闭：注明服务器路径前提或增加非默认路径重映射。
- P2-4 关闭：补真实部署成本实验后再写效率/Pareto claim。

## 最终判断

`revise`。

V7 部署代码和 evaluator 没有发现 P0；但正式晋级身份与 Idea 当前状态存在两个 P1。修复后只需复核受影响账本/diagram/IDEA diff 与 V7 身份合同；未变化的部署代码、V6 RUN 结果和测试证据可直接复用。
