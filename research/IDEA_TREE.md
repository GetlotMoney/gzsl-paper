# GZSL 创意树

## 根目标

```yaml
objective_id: GZSL-H-PLUS-3PP
framework: FRAMEWORK-V1
baseline_code_commit: 7d842e5c0e5554409eedb3097fea5130a848c9e4
evaluation_protocol: test_selected_inductive_gzsl
primary_metric: H
target_improvement_percentage_points: 3.00
baseline_status: pending_first_run
baseline_H: null
target_H: null
```

这里的 `3.00` 表示 H 提高 3.00 个百分点，例如 `70.00 → 73.00`，不是相对增长 3%。只有 `V1-CONFIRM-001` 完成并取得真实基线后，才能填写 `baseline_H` 和 `target_H`。

## 当前树

```text
GZSL-H-PLUS-3PP
└─ pending_baseline
   └─ 等待 V1-CONFIRM-001 的真实 U/S/H/ZS 与 best epoch
```

## 节点规则

- 新节点使用 `IDEA-xxx`，并链接 `ideas/IDEA-xxx_<slug>.md`。
- 每个 Idea 必须引用本仓库的 `PAPER-xxx-Cxx` 证据，或明确标记为尚待论文证据支持的 `proposed`。
- 每个正式 Idea 只包含一个相对基线的核心改动，并预先写明成立条件和失败条件。
- 实验开始后链接准确 Experiment 路径；结束后回填真实结果和 Idea 状态。
- 失败节点保留在树中，不删除、不重编号，也不事后修改成功门槛。
