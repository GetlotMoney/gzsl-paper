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
    status: formal_pending_new_repository_baseline
evaluation_protocol: test_selected_inductive_gzsl
```

V1 来源于 `model/v5-template-v2@fb4b29b04087640890a532f105cb527d3a8c461b` 的必要运行代码，旧仓库历史、旧实验和旧账本没有迁入。

## FRAMEWORK-V2

owner已将来源身份`INNOVATION-MODULE-1 / TG-VPR-H1`提升为独立正式框架`FRAMEWORK-V2`。V2使用独立代码、配置和训练入口，不接入`FRAMEWORK-V1`；迁入的多seed结果只作来源证据，不能替代本仓库正式V2基线。

## 当前待办

1. 首个真实基线已由 `V1-CONFIRM-001 / RUN-002` 完成：`U=72.3584%`、`S=76.2365%`、`H=74.2468%`、`ZS=81.4479%`。
2. 基线实际代码包含一个不改变模型与评估语义的 CUDA device validation 修复，准确提交为 `f8dd7c72465686cfe4aea8a0f37f658e1176386a`。
3. 下一目标是在相同数据和评估口径下达到 `H >= 77.2468%`。
4. 后续创新从 `V1-INNOVATION-001` 开始编号，并绑定本仓库 Idea 与论文证据。
5. V2下一步是按固定配置建立首个正式confirmation基线；在完成前不得把迁入结果写成V2 confirmed baseline。
