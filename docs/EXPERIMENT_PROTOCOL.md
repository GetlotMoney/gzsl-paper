# 实验操作规范

从第一次验证到正式接纳，始终用同一套身份：**一个 Idea，一项 Experiment，每次执行一行 RUN；代码按方法存放，成果按名称冻结。** 以下适用于新实验，旧队列和实验文件只读保留。

## 1. 文件放哪里

```text
research/ideas/IDEA-xxx_<slug>.md          # 假设、证据、演化
model/frameworks/<method_id>/            # 新方法的稳定实现目录，仅有代码时建立
experiments/<track_id>/<experiment_id>/
  EXPERIMENT.yaml                       # 问题、父条件、比较设计、入口与决定
  RUNS.csv                              # 每次执行一行，唯一运行事实源
  configs/RUN-xxx.yaml                   # 配置需要新建时放这里；已有准确配置可引用
  framework_diagram.html                # 计算/数据/评估语义变化时需要
```

`track_id` 是研究主线，例如 `triadic-native`；`method_id` 是稳定代码名称，使用小写字母、数字和下划线；`experiment_id` 在该主线内唯一。RUN 的完整身份是 `<track_id>/<experiment_id>/RUN-xxx`。均在出现真实内容时创建，不预建空目录。

独立方法放稳定方法目录，目录存在不代表正式接纳。改进已有框架时在实验分支修改必要文件，不复制整个模型；准确代码身份由 commit 确定，不能把修改后的目录继续冒称旧冻结框架。只有形成另一个需并存的完整方法时才建立独立方法目录。

历史 `model/frameworks/vX/` 与 `experiments/vX/` 不搬、不重编号；旧命名通过 [框架总账](../research/LEDGER_INDEX.md) 映射到具体方法。

## 2. 开工先确认三种起点

写入 EXPERIMENT.yaml，不另建合同文件：

| 字段 | 必须说明 |
|---|---|
| idea_ref / question | 对应假设与本次唯一问题 |
| research_mode | incremental 增量 / replacement 替换 / independent 独立方法 |
| base_commit / base_approval | owner 确认的40位代码起点，以及适用候选范围 |
| comparison_ref | 比较基线的代码、配置、数据、训练与评估条件 |
| init_ref | 预训练资产或 checkpoint 的 URI 与内容身份；新层随机初始化方式和 seed；不用权重则明确写无 |
| reuse_scope / change_scope | 复用的源 commit 与文件；本次唯一关键改动 |
| code_entry / config_ref | 实际代码入口与配置位置，不根据目录名称推断 |
| success_condition / failure_condition | 运行前的判断与停止条件 |
| management_ref | 开工时采用的 main 管理规范 commit，和代码父起点分开 |

代码从哪里来、和谁比较、权重从哪里初始化，必须分开。同代码父起点不意味着初始化相同。

同一组候选的父条件经 owner 确认后，按已授权范围持续沿用，不逐 RUN 重复询问；更换代码父条件、初始化或比较条件需明确登记，超出原授权时再确认。Idea 阶段可待定，写代码前必须落实。

## 3. 怎么切分支

1. 读取当前 main 的管理入口，记录 management_ref；切到旧父代码后仍遵守该已确认规范，不能重启旧 README 中的任务。
2. 检查当前分支和工作区。相关改动先提交到所属分支；不自动 stash、丢弃、reset 或携带无关修改进入新实验。
3. 从指定 base_commit 创建 `exp/<track_id>/<experiment_id>-<slug>`。分类可写实验字段，不再增加必选 kind 目录。历史分支原名保留。
4. 顺序实验默认在一个工作区切分支；只有需要并行保留活动现场时才建立工作树，工作树不作长期账本。

以下是模板，不是本项目已指定的新实验或父起点：

```powershell
git status --short
git branch --show-current
# 仅在修改已妥善保存、编号已登记、父起点已确认后：
git switch -c "exp/<track_id>/<experiment_id>-<slug>" "<40位base_commit>"
```

不同候选从同一已确认父条件独立分叉；失败候选不成为下一候选基线。同候选修 bug 可继续本分支，但实际执行代码必须产生新 commit。

## 4. 一次运行只记一次

新实验不再额外创建 TRY 队列或把简单实验晋级复制成另一份成绩。EXPERIMENT.yaml 保存共享条件与按日期追加的决定；RUNS.csv 保存每次执行的事实。

新 RUNS.csv 最少列：

```text
run_id,condition,code_commit,config_ref,config_sha256,seed,init_ref,asset_ref,
evaluation_protocol,status,U,S,H,ZS,history_uri,log_uri,model_uri,output_uri,
decision,code_archive_ref,artifact_check
```

共享初始化、资产和协议可显式引用 EXPERIMENT.yaml 中的字段，不能用不明确的空值暗示继承。配置只保存一份；诊断不适用的指标留空并从 output_uri 指向实际诊断量。完整披露 test 选择等协议字段，运行必须能解析到完整配置。

快速筛选后重新完整训练或换 seed 都是新 RUN，不能把同一输出改名算两次。修改公式、forward、loss、数据或评估语义为新 Experiment；核心假设改变时新 Idea。真正中断后恢复同一执行沿用 RUN 并记录恢复 checkpoint/时刻；重新初始化则新 RUN。

旧 TRY/参数矩阵/result 文件及编号保留，后续研究只引用它们；不在新 RUNS.csv 抄成另一次执行。result.md 不再是新实验强制文件，短解释直接回填 Idea 或 EXPERIMENT.yaml。

## 5. 冻结、运行、回填

- `base_commit`：开发起点。
- `code_commit`：实际运行的代码快照，由 pre-run 提交产生；在后续记录提交中绑定，不能要求提交包含自己的哈希。
- post-run commit：保存真实结果的记录快照，不能替代 code_commit。

运行前核对冻结代码、完整配置、数据资产、初始化、seed 和独立输出目录；需要的审查与框架图见 AGENTS。运行结束更新原 RUN 和 Idea 解释，再把记录指针和下一步汇入 main；不为回填账本合并候选代码。

代码、配置和轻量记录保存 Git；数据、缓存、权重和大日志保存仓库外。`code_archive_ref` 写可取回的远端引用，只有本地提交则明确 `local_only`；`artifact_check` 写 `unchecked` 或实际核对时间与范围，不能把路径存在当完整复现。

公开仓库不等于所有候选都已获准公开。缺少非公开备份去向时保留 local_only 并如实报告，不创建新仓库或偷偷上传。工作区/分支只有在无未提交内容、记录已归位、代码和产物可恢复且 owner 明确授权后才清理。

## 6. 框架怎样确定与命名

1. 完整机制及证据达到预先条件，只能成为框架候选；通过审查、smoke 或单次涨点均不自动接纳。
2. owner 明确接纳方法与准确实现。优先保留原代码路径和行为，不为晋级搬目录、改导入或改 checkpoint 格式。
3. 将接纳代码及必要记录合入 main；若集成改变运行实现，先做受影响验证，不能拿旧运行冒充新代码结果。
4. 在准确接纳 commit 建不可移动标签 `<method_id>/<commit前8位>`，例如 `tg_vpr_h1/3dc078c0`。若短码碰撞则延长；不另建同义正式分支，不续编全局 V 号。
5. 只在框架总账登记具体名称、method_id、标签、完整 commit、真实代码入口与来源证据。名称改进不改变稳定 method_id；机制变成独立新方法才新身份。

同一方法之后接纳新的实现，保留目录和 method_id，以新的 commit 标签区分；旧标签永久不移动。失败的候选实现留原分支，失败证据可进入 main；未接纳候选不会因为文件夹叫 frameworks 而晋级。

## 7. 必须保留的科学与执行边界

审查、训练协议、阈值和消融的唯一具体要求见 [AGENTS 第4、6、7节](../AGENTS.md)。不在多份规范重抄阈值，以免漂移。

- 新创新准入及代码进服务器前遵守双 Agent 规则；纯管理文档不启动独立 Agent 审核。
- module、forward、loss、数据或评估语义变化时，框架图注明准确 commit、输入输出、关键流、loss、logits/metric 与评估边界。
- 比较条件要匹配；U/S/H/ZS来自同一 best-H checkpoint，保存完整评估历史，明确 test 是否用于选择与 unseen 是否参与梯度。
- 初始化、预算或数据不同必须披露；推理关闭不冒充完整重训消融。
- 历史判定标准只解释历史结果，不用新规则重新宣判旧成绩。
