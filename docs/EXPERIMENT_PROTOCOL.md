# 实验与多 RUN 规范

## 核心对象

```text
Framework
└─ Experiment：一个研究问题
   ├─ Condition：基线、主方法、参数条件或控制条件
   │  └─ RUN：一套完整配置、一个 seed、一次执行
   └─ Condition
      └─ RUN
```

一个创新实验可以包含基线、主方法、多个参数、消融、多个 seed 和最终结果。不能把“一项创新实验”压缩成一个配置，也不能把每个参数值拆成新的创新编号。

## 探索实验清单

正式Experiment之前的快速尝试统一写入当前框架的`EXPERIMENT_QUEUE.csv`：

```text
一行 = 一个代码/配置条件 + 一个seed + 一次真实运行
```

快速尝试不创建实验目录，不写README、evidence、result或HTML图。每行必须绑定准确code commit、config、唯一改动、seed、U/S/H/ZS和仓库外输出URI。

状态只使用`planned / running / completed / failed`；运行代码或配置改变时，服务器启动前仍需有准确Git commit。

决策只使用：

- `drop`：失败或无收益，保留一行后停止；
- `keep`：有信号，继续少量尝试；
- `promote`：值得正式验证，随后创建正式Experiment目录。

只有`promote`候选进入下面的正式实验流程。

## 框架固定目录

每个正式框架必须始终提供四类实验入口：

```text
experiments/vX/
├─ tune/INDEX.md
├─ ablation/INDEX.md
├─ innovation/INDEX.md
└─ confirmation/INDEX.md
```

目录存在只表示分类入口存在，不代表已有实验。空目录的`INDEX.md`必须写明“当前无实验”和下一编号，不能伪造计划或结果。

## 实验目录

```text
INNOVATION-001_example/
├─ EXPERIMENT.yaml
├─ configs/
│  ├─ RUN-001.yaml
│  └─ RUN-002.yaml
├─ PARAMETER_MATRIX.csv
└─ result.md
```

这四类文件构成普通实验的最小闭环。只有修改代码结构、模块、forward、loss、数据流或评估语义时，才增加`framework_diagram.html`。`PARAMETER_MATRIX.md`、逐RUN evidence页、README、implementation和module_source均为按需文件；现有历史文件保留，但不要求新实验复制这些层级。

## 最短执行流程

```text
真实问题或证据
→ 一张可证伪Idea卡
→ EXPERIMENT_QUEUE.csv登记TRY
→ 代码/配置commit
→ 仓库外独立目录快速运行
→ 回填TRY结果与drop/keep/promote
→ promote后才建立正式Experiment
→ 正式pre-run / run / post-run闭环
```

不为快速尝试增加目录、审核线程、状态机、专用控制器、重复冻结或额外收据。

## 参数矩阵

`PARAMETER_MATRIX.csv` 是唯一机器事实源。每个真实训练对应一行，至少包含：

```text
run_id,stage,condition,code_commit,config_ref,config_sha256,seed,
dataset_split,evaluation_protocol,status,U,S,H,ZS,best_epoch,
test_used_for_selection,log_uri,model_uri,decision
```

## Experiment 与 RUN 的边界

- 只改变 learning rate、rank、gate、loss 权重、epoch、seed 或预注册开关：同一 Experiment 的新 RUN。
- 改变模块公式、输入信息、forward、loss、seen/unseen 边界或评估语义：新建 Experiment。
- 小规模参数选择可以留在 Innovation；模块成立后的系统性超参数搜索进入 Tune。

## HTML 框架图规则

- 每个 `FRAMEWORK-VX` 必须提供 `experiments/vX/framework_diagram.html`，并绑定该框架的准确 commit。
- 任何改变 module、forward、loss、数据流、输入输出、seen/unseen 边界或评估语义的 Experiment，必须提供实验目录内的 `framework_diagram.html`，展示相对 base commit 的实际差异。
- 参数、seed、epoch、纯文档和不改变计算语义的运行修复继续复用框架级 HTML 图，但必须在 `EXPERIMENT.yaml` 或 evidence 中链接该图并说明代码差异。
- HTML 图至少包含：输入、关键模块、主要张量/数据流、训练 loss、最终 logits、U/S/H/ZS 出口、配置开关或固定参数、baseline-off 行为及协议边界。
- 图必须是自包含 HTML，不依赖仓库外 CDN；修改后至少做一次浏览器打开检查。

## 推荐阶段

1. `baseline`：同 commit、同数据和同评估口径的基线。
2. `main`：创新模块默认条件。
3. `parameter`：少量预注册参数条件。
4. `control`：module-off、shuffle、wrong-role 等机制控制。
5. `repeat`：值得保留后再跑其他 seed。

新的论文主结果不得根据official test选择参数、epoch、模型或seed。开发RUN使用类别不相交validation并标记：

```yaml
evaluation_protocol: xlsa17_class_disjoint_gzsl_validation
validation_used_for_selection: true
test_used_for_selection: false
unseen_images_used_for_gradient: false
```

开发阶段使用`train_loc`训练、`val_loc`作为validation-unseen；若按H选择模型，还必须固定保留开发seen图像计算validation-seen。选定后在`trainval_loc`重训，official test只在最终checkpoint完成后评估。历史RUN若使用official test选模，继续记录为`test_selected_inductive_gzsl / test_used_for_selection: true`，只能作为探索结果。

## 多seed成绩口径

- seed必须在开发RUN前固定；最终seed集合在official test前一次性冻结，不得看到结果后改报最高seed。
- 追加seed用于稳定性判断时必须全部报告，并计算`mean / min / max / range`；最高seed只能作为分布描述，不能替代预注册主seed。
- 新论文核心创新原则上要求相对准确父条件在预注册主seed上`Delta H >= 0.20`个百分点，并由追加seed排除明显偶然性；更小增益只作为辅助模块或观察。
- 最终多seedRUN使用validation已选定的epoch或训练日程；official test结果必须全部报告，不得用于新一轮调参。
