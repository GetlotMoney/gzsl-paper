# 实验与多 RUN 规范

新模块必须先完成[`MODULE_WORKFLOW_CHECKLIST.md`](MODULE_WORKFLOW_CHECKLIST.md)；该清单是模块准备、硬门槛消融、止损和结果回填的唯一操作入口。

## 版本、分支与账本继承

```text
公共工作流规范、已接纳代码和轻量账本 → main
正式版本代码身份                         → framework/vX + Tag vX
版本探索账本                             → experiments/vX/
具体候选代码                             → exp/vX/<kind>/<module>
```

- Git分支是整个仓库的快照，不是只显示当前版本目录的过滤器。
- V2分支保留`experiments/v1/`，V3分支保留`experiments/v1/`和`experiments/v2/`；这些旧目录在后续分支中只读。
- 新版本不继承旧版本的TRY编号、状态或成绩，只继承Git历史可见性。
- 当前版本的实验只写当前`experiments/vX/EXPERIMENT_QUEUE.csv`；旧结果作为证据时引用原路径、原RUN、原commit、原配置和原协议。
- 不删除旧版本账本来制造“分支纯净”；代码版本由commit/branch/Tag隔离，账本版本由`experiments/vX/`路径隔离。
- 候选失败只更新当前版本账本和Idea状态，不改写旧版本结论。
- 候选失败后冻结该候选分支，不从它继续开发下一模块。下一候选必须先由owner确认准确父commit，再从该commit创建新的`exp/vX/<kind>/<module>`分支；旧失败分支只供复现和追溯。
- 候选被owner接纳后，先进入`main`的已接纳状态，再固定新的`framework/vY`与Tag `vY`；正式引用一经创建不移动。

## 代码两轮Agent对抗门槛

- 只对改变`module / forward / loss / 数据流 / 训练器 / 评估语义`的代码启动两轮审核；纯配置、队列、结果和文档只做确定性校验。
- 共享证据固定为五项：准确RUN/config、相关专项测试、一次完整测试、每种objective一次真实GPU micro-batch、一次正常checkpoint恢复。两名Agent直接复用，不重复生成。
- Round 1只审公式、初始化、梯度、数据边界和Full/Off；Round 2只审身份、服务器clean、真实评估出口和正常resume。并行预读，Round 1通过后Round 2签字。
- 正常resume只验证保存、`weights_only`加载、恢复后的下一batch与LR。默认禁止恶意篡改/fuzz、逐张量hash链、optimizer/RNG字节攻击和无穷字段枚举。
- 若有P0/P1，两名Agent先完成既定范围并一次性交齐清单；只做一批集中修复。复核只审最终diff和受影响合同，完整测试、资产和GPU证据只要语义未变就直接复用。
- 证据齐全时两轮目标5–10分钟。命令或自造入口连续失败两次必须停止并改走现有入口。
- 审核只写现有Experiment的`REVIEW.md`，不新增receipt、逐RUN证据页、状态机或额外目录。只有第二轮明确“无P0/P1，第2轮通过”才允许正式RUN。

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

## 实验目录

```text
INNOVATION-001_example/
├─ README.md
├─ EXPERIMENT.yaml
├─ module_source.md
├─ implementation.md
├─ framework_diagram.md
├─ configs/
│  ├─ RUN-001.yaml
│  └─ RUN-002.yaml
├─ PARAMETER_MATRIX.csv
├─ PARAMETER_MATRIX.md
├─ evidence/
│  ├─ RUN-001.md
│  └─ RUN-002.md
└─ result.md
```

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

## 推荐阶段

1. `baseline`：同 commit、同数据和同评估口径的基线。
2. `main`：创新模块默认条件。
3. `parameter`：少量预注册参数条件。
4. `control`：module-off、shuffle、wrong-role 等机制控制。
5. `repeat`：值得保留后再跑其他 seed。

项目允许根据 official test U/S/H/ZS 选择参数、epoch 和模型。每次 RUN 必须标记：

```yaml
evaluation_protocol: test_selected_inductive_gzsl
test_used_for_selection: true
unseen_images_used_for_gradient: false
```

这不是 blind-test 证据，任何论文数字或对外比较都必须如实说明。
