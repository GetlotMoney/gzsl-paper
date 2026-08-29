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

## 一轮双Agent交叉审查门槛

- 修改module、forward、loss、数据/资产生成、训练器或评估语义的Experiment，pre-run冻结前必须完成一轮双Agent独立只读交叉审查。
- 两名Agent同时审查同一个冻结commit；交流前各自完成独立检查和完整`P0/P1/P2`初始清单，不得提前共享结论。
- 双方完成后相互交换一次完整清单，并各自回复一次补充、异议和最终结论。交叉交流完成后不再串行重复第二轮全量审核。
- 只有双方最终都明确`P0=0 / P1=0`并写出“双Agent交叉审查通过”才允许服务器smoke或训练。签字绑定最终RUN commit、审查声明路径tree hash、config SHA、资产manifest SHA和环境/GPU fingerprint；其中任一语义身份变化即失效。
- 若任一方发现P0/P1，双方仍须完成各自范围和一次交叉交流，主Agent汇总后制作一个集中修复批次；新commit重新执行一轮完整双Agent交叉审查，旧签字失效。
- 审查记录至少绑定双方Agent、同一reviewed commit、双方初始清单、一次交叉交流、修复commit、测试和最终双签；机器测试、单Agent自审或聊天口头确认不能替代双Agent审查。
- 双Agent审查覆盖任何会改变计算或评估语义的代码和配置。已审schema内、未启用新forward/loss/objective/评估路径的参数值配置，以及纯队列、结果和文档，只走确定性contract；生成后资产走manifest/SHA/shape/dtype/count校验。
- 审核前置全流程从共享证据准备开始计时，到独立检查、一次交叉交流和最终双签结束，硬性上限15分钟。默认只准备准确diff、最小相关测试、数据/test边界、一次GPU micro-batch和输出不存在证明。
- 超过15分钟，本次审核立即标记`failed_due_to_audit_timeout`并禁止启动；记录最慢步骤和重复劳动，删减流程后重新审核，不得超时后继续补证据或勉强签字。
- 已有manifest/资产SHA、未变化测试和父结果直接复用；禁止默认重复整仓测试、全量大文件SHA、双GPU重复预载或新增收据/证明文档。证据不足仅记`warning_evidence_incomplete`，不阻断15分钟内基于现有证据判断P0/P1。
- 两名Agent必须并行启动，完成初始清单后只交流一次；不得追加第二轮完整审核。发现P0/P1时仍汇总当前清单，但本次不运行，集中修复后重新开始新的15分钟审核。
- 任一Agent可在仓库外临时目录重放micro-batch，但不得写正式资产或RUN。P0/P1修复后以新代码身份重新开始完整的一轮双Agent审查。
- 冻结commit后先建立审查矩阵，多Agent并行覆盖静态语义、真实GPU、资产/评估和checkpoint；各Agent必须审完分工，不能发现一个问题就提前退出。
- 两名Agent完成独立清单与一次交叉交流后设置汇合点，主Agent去重完整P0/P1/P2清单；每个审核周期只提交一个集中修复批次。若复核发现新P0/P1，完成该周期全范围检查与交叉交流后再进入下一批。
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

owner选择的论文主结果采用Chen-style公开代码对齐协议，每次RUN标记：

```yaml
evaluation_protocol: chen_shiming_code_aligned_test_selected_gzsl
test_used_for_selection: true
unseen_images_used_for_gradient: false
strict_blind_claim: false
```

默认代码对齐条件为batch 50、200名义epoch、`niters=ntrain*epochs//batch_size`、`report_interval=niters//epochs`、每步独立随机抽样、每个report interval评估official test并按完整模型H保存best。端到端RUN只允许整模型选模；分阶段嵌套test选择必须另建Experiment并显式披露。现有validation-first RUN保留为严格协议对照。

V3快速候选可在独立探索Experiment中使用预注册动态筛选：最多150名义epoch，只在固定60/80/100/120/150轮边界依据累计best-H与U/S差停止。动态结果不得直接进入论文最终表；胜出累计条件和单模块移除必须重新固定200名义epoch。

## 多seed成绩口径

- 首个Chen-style主RUN固定使用TransZero CUB配置seed 5；追加seed必须全部报告并计算`mean / min / max / range`。
- owner内部主成绩可引用最高seed，但必须同时列出全部seed与波动，不得隐藏失败seed。
- 新论文核心创新原则上要求相对准确父条件`Delta H >= 0.20`个百分点；更小增益只作为辅助模块或观察。
