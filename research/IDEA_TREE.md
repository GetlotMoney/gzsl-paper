# GZSL 创意树

## 根目标

```yaml
objective_id: GZSL-H-PLUS-3PP
framework: FRAMEWORK-V1
baseline_code_commit: 7d842e5c0e5554409eedb3097fea5130a848c9e4
evaluation_protocol: test_selected_inductive_gzsl
primary_metric: H
target_improvement_percentage_points: 3.00
target_supported_innovations: 3
paper_method_requirement: one_coherent_framework
naming_requirement: one_method_name_plus_three_consistent_subnames
baseline_status: completed_with_runtime_device_fix
baseline_H: 74.2468
target_H: 77.2468
```

这里的 `3.00` 表示 H 提高 3.00 个百分点，不是相对增长 3%。当前真实基线为 `74.2468%`，目标为 `H >= 77.2468%`。

## 当前树

```text
GZSL-H-PLUS-3PP
└─ baseline_completed
   ├─ V1-CONFIRM-001 / RUN-002：H=74.2468%
   └─ paper_method_goal
      ├─ shared_research_question：待证据确定
      ├─ innovation_slot_1：待筛选与验证
      ├─ innovation_slot_2：待筛选与验证
      ├─ innovation_slot_3：待筛选与验证
      ├─ integration_gate：三者必须形成连续或互补逻辑
      ├─ naming_gate：一个总方法名 + 三个统一风格子名称
      └─ metric_target：H>=77.2468%
```

## 节点规则

- 新节点使用 `IDEA-xxx`，并链接 `ideas/IDEA-xxx_<slug>.md`。
- 每个 Idea 必须引用本仓库的 `PAPER-xxx-Cxx` 证据，或明确标记为尚待论文证据支持的 `proposed`。
- 每个正式 Idea 只包含一个相对基线的核心改动，并预先写明成立条件和失败条件。
- 每个候选必须说明它在三创新论文主线中的角色、与其他候选的接口关系和命名候选。
- 只有获得证据与实验支持且通过整体逻辑、接口和命名检查的节点，才能标记为 `paper_core_innovation`；最终只选择三个。
- 三个最终节点必须共享一个核心研究问题，并能在同一 HTML 框架图中形成自然的数据流或机制关系；互不相关的模块不得强行并列。
- 实验开始后链接准确 Experiment 路径；结束后回填真实结果和 Idea 状态。
- 失败节点保留在树中，不删除、不重编号，也不事后修改成功门槛。
