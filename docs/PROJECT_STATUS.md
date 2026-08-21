# 项目状态

## 当前版本

```yaml
repository: GetlotMoney/gzsl-paper
frameworks:
  - id: FRAMEWORK-V1
    branch: framework/v1
    tag: v1
    status: baseline_completed_with_runtime_device_fix
  - id: FRAMEWORK-V2
    branch: framework/v2
    tag: v2
    status: baseline_completed_single_seed
evaluation_protocol: test_selected_inductive_gzsl
paper_primary_framework: FRAMEWORK-V2
paper_baseline_H: 74.023182
paper_target_H: 77.023182
target_supported_innovations: 3
supported_innovations: 2
current_seed7_H: 76.984545
current_multiseed_mean_H: 76.866245
```

V1 来源于 `model/v5-template-v2@fb4b29b04087640890a532f105cb527d3a8c461b` 的必要运行代码，旧仓库历史、旧实验和旧账本没有迁入。

## FRAMEWORK-V2

owner已将来源身份`INNOVATION-MODULE-1 / TG-VPR-H1`提升为独立正式框架`FRAMEWORK-V2`。V2使用独立代码、配置和训练入口，不接入`FRAMEWORK-V1`。首个当前仓库正式基线已由`V2-CONFIRM-001 / RUN-001`完成：`U=72.655779%`、`S=75.443041%`、`H=74.023182%`、`ZS=81.534684%`。

owner已授权直接迁移H1旧实验的轻量证据。组件消融、多seed和参数收口证据位于`experiments/v2/evidence/legacy_h1/`；`IDEA-001 / TG-VPR-H1`现为论文核心创新1，状态`supported`。

## 当前待办

owner已选择`FRAMEWORK-V2`作为论文主框架。V2当前正式单seed基线为`H=74.023182%`，新的三个百分点目标为`H >= 77.023182%`。

固定10%保守unseen迁移在四seed均提升H，但它只在测试时生效，现降级为`test_time_observation`，不计入论文核心创新。

训练式ELPT已完成`V2-TRY-006`及全部3次方法级补救。最佳H达到`76.803085%`，但首次TRY的gate均值超过预注册上限；三个补救又持续出现gate饱和或S下降超过2个百分点，因此`IDEA-002`已标记`rejected`并强制止损。没有建立`V2-INNOVATION-002`。

`IDEA-003 / ICGR`已完成首次TRY与两次适用的补救。原始路由和增加语义余弦输入均未提高H；均匀KL消除了权重塌缩，但最终仍为`H=73.976174%`、`ΔH=-0.047008`。只适用于跨seed不稳定的RESCUE-3前提不成立，因此该方向已提前止损并标记`rejected`。

当前仍只有`IDEA-001 / TG-VPR-H1`一个supported核心创新。下一步必须建立新的独立训练式候选，不能把ELPT或ICGR失败条件晋级为正式创新。

`IDEA-004 / ACGR`使U和ZS出现正向信号，但H未提升；一次保守幅度补救仍失败且发生组权重塌缩，现已标记`rejected`。下一候选回到原型迁移主线，改用切空间方向迁移的新公式，不复用ELPT实验身份。

`IDEA-005 / TST`已在seed 5/6/7/8全部提高H，平均提升`3.013152`个百分点，候选H mean=`76.866245%`，已超过四seed目标`76.853093%`，现为论文核心创新2。seed7为`76.984545%`，距离单点目标仍差`0.038637`个百分点；项目还缺第3个supported创新，因此整体工作未完成。

完整执行顺序和完成条件见[`docs/PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md)。
