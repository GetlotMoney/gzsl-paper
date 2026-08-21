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
current_seed7_H: 75.587012
current_multiseed_mean_H: 75.237222
```

V1 来源于 `model/v5-template-v2@fb4b29b04087640890a532f105cb527d3a8c461b` 的必要运行代码，旧仓库历史、旧实验和旧账本没有迁入。

## FRAMEWORK-V2

owner已将来源身份`INNOVATION-MODULE-1 / TG-VPR-H1`提升为独立正式框架`FRAMEWORK-V2`。V2使用独立代码、配置和训练入口，不接入`FRAMEWORK-V1`。首个当前仓库正式基线已由`V2-CONFIRM-001 / RUN-001`完成：`U=72.655779%`、`S=75.443041%`、`H=74.023182%`、`ZS=81.534684%`。

owner已授权直接迁移H1旧实验的轻量证据。组件消融、多seed和参数收口证据位于`experiments/v2/evidence/legacy_h1/`；`IDEA-001 / TG-VPR-H1`现为论文核心创新1，状态`supported`。

## 当前待办

owner已选择`FRAMEWORK-V2`作为论文主框架。V2当前正式单seed基线为`H=74.023182%`，新的三个百分点目标为`H >= 77.023182%`。

`IDEA-002 / V2-INNOVATION-001`已完成：固定10%保守unseen迁移在seed 5/6/7/8全部提高H，平均`+1.384128`个百分点；seed 7从`74.023182%`提高到`75.587012%`。该方法无新增参数、无额外训练，但10%由official test下的快速TRY选择，必须披露test-exposed边界。

下一件事是建立`IDEA-003`，研究单图像如何动态选择local/unique/overall证据，并与创新1、创新2形成连续逻辑。

完整执行顺序和完成条件见[`docs/PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md)。
