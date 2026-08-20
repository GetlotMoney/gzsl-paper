# 项目状态

## 当前版本

```yaml
repository: GetlotMoney/gzsl-paper
framework: FRAMEWORK-V1
branch: framework/v1
tag: v1
status: baseline_completed_with_runtime_device_fix
evaluation_protocol: test_selected_inductive_gzsl
```

V1 来源于 `model/v5-template-v2@fb4b29b04087640890a532f105cb527d3a8c461b` 的必要运行代码，旧仓库历史、旧实验和旧账本没有迁入。

## 当前待办

1. 首个真实基线已由 `V1-CONFIRM-001 / RUN-002` 完成：`U=72.3584%`、`S=76.2365%`、`H=74.2468%`、`ZS=81.4479%`。
2. 基线实际代码包含一个不改变模型与评估语义的 CUDA device validation 修复，准确提交为 `f8dd7c72465686cfe4aea8a0f37f658e1176386a`。
3. 下一目标是在相同数据和评估口径下达到 `H >= 77.2468%`。
4. 后续创新从 `V1-INNOVATION-001` 开始编号，并绑定本仓库 Idea 与论文证据。
