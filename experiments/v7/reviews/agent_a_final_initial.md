# Agent A final initial review — FRAMEWORK-V7

## 结论

- Reviewer：Agent A
- 阶段：FRAMEWORK-V7 最终审查阶段1复核
- 新冻结 commit：`7a7a4c1087b64aadb244a164da0e5955290711b1`
- 上一 commit：`568b01ec8a8e48ffe78336a6fc99f7708de03cbc`
- promotion source：`2f7837266f4077b3fb7e40927fc6571499a76747`
- config SHA256：`7c806382b6d1899a3639ed16cd287c7894b210efda58707358172b2224b943dd`
- 测试证据：主Agent声明 `32 passed`；本阶段未重跑测试、未运行服务器
- 最终判断：`pass`
- P0：0
- P1：0
- P2：4

我没有发现会阻断 FRAMEWORK-V7 最终晋级的代码或合同问题。上一轮 A 提出的两个 P1 已关闭：身份四分已落账，IDEA-201 顶部当前状态已改为 owner 晋级 V7，并且历史 program drop、低于内部 V5/matched online-V5、test-selected/nonblind、未完成成本和新颖性对照都继续保留。

按主Agent本轮说明，`framework/v7` 分支和 `v7` Tag 会在最终审查文件提交后的唯一正式 commit 上创建；当前 `refs_status: pending_promotion_freeze` 和 HTML 中 “Formal refs pending one-time freeze” 是诚实的流程状态，不作为代码错误。最终 ref 是否存在应留到创建后只读核验。

## 已读取范围

已读取并核对：

- `AGENTS.md`
- `README.md`
- `research/ideas/IDEA-201_compiled_pclr.md`
- `research/IDEA_TREE.md`
- `model/frameworks/v7/model.py`
- `model/frameworks/v7/evaluate.py`
- `config/framework_v7.yaml`
- `experiments/v7/FRAMEWORK.yaml`
- `experiments/v7/framework_diagram.html`
- `experiments/v7/EXPERIMENT_QUEUE.csv`
- `experiments/v7/innovation/INDEX.md`
- `experiments/v7/ablation/INDEX.md`
- `experiments/v7/tune/INDEX.md`
- `experiments/v7/confirmation/INDEX.md`
- V6 账本：`experiments/v6/EXPERIMENT_QUEUE.csv` 及 V6-TRY-006 相关账本事实
- `tests/frameworks/v7/test_v7_deployment.py`
- `568b01ec8a8e48ffe78336a6fc99f7708de03cbc..7a7a4c1087b64aadb244a164da0e5955290711b1` 的完整 diff

未读取 B final 文件。

## 复核要点

### 1. 身份四分

通过。

`experiments/v7/FRAMEWORK.yaml` 现在清楚拆分：

- `promotion_source_commit: 2f7837266f4077b3fb7e40927fc6571499a76747`
- `v6_reviewed_training_code_commit: b707b0c4671051244cebf4f8404299fc016b281e`
- `source_run_commit: 8de7cebda0235ab12e1b4b8f669134c8f4e2c075`
- `v7_deployment_code_commit: 568b01ec8a8e48ffe78336a6fc99f7708de03cbc`

同时记录了 `source_checkpoint_sha256: a551de9d...f2207c9`、`source_metrics_sha256: fbbd8e...0879` 和 `framework_v7_config_sha256: 7c8063...3dd`。这已经足够区分 V6 训练审查身份、V6 正式 RUN 产物身份、V7 晋级来源和 V7 部署代码身份。

说明：本轮 `7a7a4c1` 改动了 evaluator 身份校验和账本，不改变 V7 deployment model 的 `Reader/Q/b` 前向。最终 `framework/v7`/`v7` ref 将在审查文件提交后的正式 commit 上创建，因此当前 pending refs 是可接受状态。

### 2. IDEA-201 当前状态

通过。

IDEA-201 frontmatter 为 `status: promoted_framework_v7`、`performance_status: above_paper_parent_owner_promoted_v7`。顶部当前身份已改为：owner 将 S/V/I 统一方法晋级为 FRAMEWORK-V7，论文父基线固定 TG+GTD；历史 `proof_of_path/revise`、`drop_gate_b_contract_failed`、低于内部 V5/matched online-V5、test-selected 与未完成成本/多数据集/新颖性对照继续保留。

这关闭了上一轮“同一 Idea 同时说已 promoted 与未授权进入框架”的矛盾。

### 3. evaluator 字段校验

通过。

`model/frameworks/v7/evaluate.py` 在原有 `schema_version`、checkpoint path、asset config path、checkpoint SHA 和 asset config SHA 校验基础上，新增硬校验：

- `source_run_commit == 8de7cebda0235ab12e1b4b8f669134c8f4e2c075`
- `source_training_config_sha256 == 73a812268b18e9f46a2cedf59acdabb8ef0cdb13388ec83b5f23b73475e4239b`
- `promotion_source_commit == 2f7837266f4077b3fb7e40927fc6571499a76747`

`load_v7_checkpoint()` 仍检查 checkpoint 内部 `experiment_id`、`code_commit`、`config_sha256` 和 export dict。两层合起来能防止错误 checkpoint、错误 source config 或错误 promotion source 被静默用于正式评估。

### 4. 独立部署只用 Reader / Q / b

通过。

`V7DeploymentModel` 的持久状态只包含六个导出张量：`q`、`bias`、`reader_in_weight`、`reader_in_bias`、`reader_out_weight`、`reader_out_bias`。前向只执行：

```text
image = normalize(x)
u = normalize(x + W2 GELU(W1 x))
logits = concat(image, u) @ Q.T + b
```

模型代码不 import V6、不读取 relation edges、不执行 Top-K、不做 Laplacian solve、不保留 online graph/cap/std/late ensemble。测试也验证 state_dict 不包含 `edge_index`、`relation_embeddings`、`incidence`、`laplacian_map`。

### 5. U/S/H/ZS 评估口径

通过。

evaluator 使用 `train_labels` 得到 seen class，unseen class 由 200 类全集减 seen 得到。GZSL seen/unseen 均对完整 200 类 logits argmax；ZS 仅对 unseen 列 argmax 后映射回全局 class id；`per_class_accuracy()` 按类平均，`H=2*S*U/(S+U)`。`EXPECTED_METRICS` 与 V6 result、V7 config、V7 framework 中的正式数值一致。

未发现跨 checkpoint 拼接 best-H/best-ZS 或把 ZS 当 GZSL U 的问题。

### 6. 数据边界与论文父条件

通过。

V7 文档继续披露：

- `test_used_for_selection: true`
- `test_used_for_hyperparameter_selection: true`
- `nested_official_test_selection: true`
- `unseen_images_used_for_gradient: false`
- `strict_blind_claim: false`
- `llm_world_knowledge_used: true`

V6 账本继续保留 program decision：`drop_gate_b_contract_failed`，并明确 C-PCLR 低于内部 matched online-V5 和正式 V5。V7 同时把论文父基线固定为 TG+GTD / `TUNE-002-RUN-030` / `H=79.070015`。这种“论文父 TG+GTD + 内部 V5/matched online-V5 作为开发事实保留”的并存是诚实的，没有删除失败事实或把内部失败改写成准确率胜利。

### 7. pending refs

通过。

`README.md` 已从“冻结分支/Tag 已存在”的语气改为“正式 `framework/v7` 与 Tag `v7` 将在最终审查后一次性冻结”。`experiments/v7/FRAMEWORK.yaml` 使用 `refs_status: pending_promotion_freeze`，HTML 图写 `Formal refs pending one-time freeze`。这与主Agent本轮说明一致，不应作为代码错误。

最终正式 commit 创建后，仍需要只读核验 `framework/v7` 和 `v7` 是否指向同一 commit；这是后置核验，不阻断当前审查。

## P0

无。

## P1

无。

## P2

### P2-1：最终 ref 存在性仍需后置核验

证据：当前 `refs_status: pending_promotion_freeze`，README/HTML 也说明正式 refs 会在最终审查文件提交后创建。

影响：这是诚实 pending，不是代码错误；但最终交付前必须确认 `framework/v7` 分支和 `v7` Tag 指向同一正式 commit。

关闭条件：正式 commit 创建后，执行只读核验：`git rev-parse framework/v7` 与 `git rev-parse v7^{}` 一致，并与最终框架提交一致。

### P2-2：V7 deployment code commit 与最终 framework commit 会不同，需要后续签字说明

证据：`v7_deployment_code_commit` 记录的是 `568b01e...`，本轮冻结 commit `7a7a4c1...` 增加了 evaluator 字段校验和账本修复；最终正式 commit 还会包含 review 文件。

影响：不影响前向部署，因为 `model.py` 没变；但最终签字应说明 `568b01e...` 是 deployment model 代码身份，最终 framework commit 是账本/审查绑定身份。

关闭条件：最终汇总中记录 `final_framework_commit`，并说明它包含 `v7_deployment_code_commit=568b01e...` 的部署模型、`7a7a4c1...` 的 evaluator/config 合同修复和最终 review 文件。

### P2-3：测试未覆盖 evaluator 新增字段校验的负例

证据：`tests/frameworks/v7/test_v7_deployment.py` 覆盖 `hQ^T+b` 等价、checkpoint 身份和 graph-free state；但新增的 `source_run_commit/source_training_config_sha256/promotion_source_commit` evaluator config 校验没有专门负例测试。

影响：代码审查可直接读出校验存在，且主Agent声明测试 `32 passed`；这不是阻断项。

关闭条件：后续轻量测试可增加一个 mocked config/sha 场景，验证三个新增字段任一错误时 evaluator 拒绝。

### P2-4：效率 claim 仍缺真实延迟/显存/吞吐

证据：V7 framework、README、IDEA-201 均保留“deployment latency / AWA2 / SUN / multi-seed / irreducibility controls remain future confirmation evidence” 类限制。

影响：当前只能 claim “部署路径移除在线图推理，只执行 Reader 与 `hQ^T+b`”；不能 claim 已证明更快、Pareto 优势或跨数据集泛化。

关闭条件：按 IDEA-201 成本合同，同硬件、同 dtype、同预载特征，报告 Parent、V5/R4 online relation head、V7 exported head 的 batch=1 与评测 batch p50/p95 latency、吞吐、峰值显存和模型文件大小。

## 最强反例

最强运行反例：在含 `/data/...` warehouse 的服务器上，用 `config/framework_v7.yaml` 运行 V7 evaluator，若 checkpoint SHA、asset config SHA 和新增身份字段都通过，但复现不出 `U/S/H/ZS=77.606910/83.639657/80.510432/88.473403`，或需要 V6 online relation graph / Top-K / Laplacian 才能得到该结果，则 V7 独立部署合同失败。

最强论文边界反例：若后续论文把 V7 写成超过内部正式 V5 或 matched online-V5 的准确率提升，或写成 blind-test / 未使用 official test selection，则与当前 V6/V7 账本冲突。当前文件已保留这些限制，因此该反例未在本 diff 中发生。

## 最终判断

`pass`。

P0=0，P1=0。当前只剩 P2：最终 refs 后置核验、最终 framework commit 与 deployment code commit 的说明、evaluator 字段校验负例测试、真实成本指标。它们不阻断最终审查进入阶段2交叉交换。
