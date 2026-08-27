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

- 修改module、forward、loss、数据/资产生成、训练器或评估语义的Experiment，pre-run冻结前必须完成两轮不同Agent的独立只读对抗审查。
- 第一轮发现的P0/P1必须全部修复并通过直接相关测试；第二轮绑定准确post-fix commit、clean工作树和真实服务器资产重新审查。
- 只有第二轮明确“无P0/P1，第2轮通过”才允许服务器smoke或训练。签字绑定最终RUN commit、审查声明路径tree hash、config SHA、资产manifest SHA和环境/GPU fingerprint；其中任一语义身份变化即失效。若第二轮发现P0/P1，全部Agent先完成分工和清单，再集中修复并由新的独立Agent审查新commit。
- 审查记录至少绑定review round、reviewed commit、发现、修复commit、测试和结论；机器测试、同一Agent自审或聊天口头确认不能替代两轮审查。
- 两轮审核覆盖任何会改变计算或评估语义的代码和配置。已审schema内、未启用新forward/loss/objective/评估路径的参数值配置，以及纯队列、结果和文档，只走确定性contract；生成后资产走manifest/SHA/shape/dtype/count校验。
- 审核前先生成一次共享证据：准确diff、相关测试、本地完整测试、资产/config校验和服务器临时micro-batch。两名Agent复用该证据，禁止重复整仓测试和全量文件哈希。
- 审核正确性和完整性优先。共享证据齐全且无P0/P1时，两轮完整审核力争10分钟完成，但这不是强制截止线，禁止为了时长跳过检查或降低标准。
- 超过10分钟必须立即汇报剩余项和原因并继续审完；`证据不足`阻止运行，发现P0/P1时先完成全部并行分工并汇总完整清单，再集中修复和复核。
- Round 2可在仓库外临时目录重放micro-batch，但不得写正式资产或RUN。P0/P1修复后旧签字失效，以新代码身份重新开始完整审核。
- Round 2允许与Round 1并行预读同一冻结commit以节省时间，但只能在Round 1无P0/P1后签字；Round 1失败时签字资格作废，Agent仍须完成分工并提交完整问题清单。
- 冻结commit后先建立审查矩阵，多Agent并行覆盖静态语义、真实GPU、资产/评估和checkpoint；各Agent必须审完分工，不能发现一个问题就提前退出。
- 所有Agent完成后设置汇合点，主Agent去重完整P0/P1/P2清单；每个审核周期只提交一个集中修复批次。若复核发现新P0/P1，完成该周期全范围检查后再进入下一批。
- 共享证据按最终commit、审查声明路径tree hash、config SHA、资产manifest SHA和环境/GPU fingerprint复用；未变化的完整测试、父指标和大资产预载不得重复执行。
- 每个不同objective/forward/loss路径至少跑一次真实GPU micro-batch；相同路径不跨GPU重复，第二GPU只验证设备及特有差异。共享闭环同时覆盖梯度、ZS、动态停止、best选择和checkpoint roundtrip。

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
