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
```

V1 来源于 `model/v5-template-v2@fb4b29b04087640890a532f105cb527d3a8c461b` 的必要运行代码，旧仓库历史、旧实验和旧账本没有迁入。

## FRAMEWORK-V2

owner已将来源身份`INNOVATION-MODULE-1 / TG-VPR-H1`提升为独立正式框架`FRAMEWORK-V2`。V2使用独立代码、配置和训练入口，不接入`FRAMEWORK-V1`。首个当前仓库正式基线已由`V2-CONFIRM-001 / RUN-001`完成：`U=72.655779%`、`S=75.443041%`、`H=74.023182%`、`ZS=81.534684%`。

## 当前待办

owner已选择`FRAMEWORK-V2`作为论文主框架。V2当前正式单seed基线为`H=74.023182%`，新的三个百分点目标为`H >= 77.023182%`。

下一件事是重新核对论文来源并建立本仓库`PAPER-001`和`IDEA-001`，使TG-VPR-H1具备当前仓库可追溯的论文创新身份。之后才进入创新2和创新3的证据、代码与实验。

完整执行顺序和完成条件见[`docs/PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md)。
